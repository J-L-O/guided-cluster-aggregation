import logging

import torch
from torch import nn


log = logging.getLogger(__name__)


class VITB16Backbone(nn.Module):
    def __init__(
        self,
        dino_pretrained: bool = False,
        checkpoint_path: str = None,
        grad_from_block: int = 11,
    ):
        super(VITB16Backbone, self).__init__()

        self.grad_from_block = grad_from_block

        load_checkpoint = checkpoint_path is not None
        assert not (
            dino_pretrained and load_checkpoint
        ), "Cannot use both dino_pretrained and checkpoint_path"

        # Need to be careful with module collisions here
        # See https://github.com/ultralytics/yolov5/issues/2414
        self.model = torch.hub.load(
            "facebookresearch/dino:main", "dino_vitb16", pretrained=dino_pretrained
        )

        if load_checkpoint:
            log.info(f"Loading checkpoint from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")

            if "state_dict" in state_dict:
                # We are loading checkpoint from this repo
                state_dict = state_dict["state_dict"]

                # Remove the "model.backbone.model" prefix from the keys
                state_dict = {
                    k.replace("model.backbone.model.", ""): v
                    for k, v in state_dict.items()
                }
                missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

                # Unexpected keys are okay, missing keys are not
                assert len(missing) == 0, f"Missing keys: {missing}"
            else:
                # We are loading checkpoint from DINO or GCD repo

                if "student" in state_dict:
                    # We are loading checkpoint from DINO repo
                    state_dict = state_dict["student"]

                    state_dict = {
                        k.replace("module.backbone.", ""): v for k, v in state_dict.items()
                    }

                if "module" in list(state_dict.keys())[0]:
                    # Remove the "module." prefix from the keys
                    state_dict = {
                        k.replace("module.", ""): v for k, v in state_dict.items()
                    }

                missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

                # Unexpected keys are okay, missing keys are not
                assert len(missing) == 0, f"Missing keys: {missing}"

        self.partial_freeze()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def partial_freeze(self):
        self.freeze()

        # From https://github.com/sgvaze/generalized-category-discovery/blob/main/methods/contrastive_training/contrastive_training.py
        # ----------------------
        # HOW MUCH OF BASE MODEL TO FINETUNE
        # ----------------------
        for name, param in self.model.named_parameters():
            if "block" in name:
                block_num = int(name.split(".")[1])
                if block_num >= self.grad_from_block:
                    param.requires_grad = True

    def forward(self, x):
        return self.model(x)
