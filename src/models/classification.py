from typing import List

from torch import nn


class HierarchicalClassificationModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        backbone_feature_dim: int,
        class_hierarchy: List[int],
        heads_per_level: List[int],
    ):
        super(HierarchicalClassificationModel, self).__init__()

        self.backbone = backbone
        self.class_hierarchy = class_hierarchy

        hierarchy_list = []

        for i in range(len(class_hierarchy)):
            num_classes = class_hierarchy[i]
            num_heads = heads_per_level[i]
            heads_list = []

            for _ in range(num_heads):
                head = nn.Linear(
                    in_features=backbone_feature_dim,
                    out_features=num_classes,
                    bias=True,
                )

                heads_list.append(head)

            heads = nn.ModuleList(heads_list)

            hierarchy_list.append(heads)

        self.heads = nn.ModuleList(hierarchy_list)

    def forward(self, x):
        features_backbone = self.backbone(x)

        logits = []

        # Get logits for each head
        for hierarchy_level, heads in enumerate(self.heads):

            features_head = features_backbone

            level_logits = [head(features_head) for head in heads]
            logits.append(level_logits)

        result = {
            "features": features_backbone,
            "logits": logits,
        }

        return result
