from typing import Optional

import torch
import torchmetrics
from scipy.optimize import linear_sum_assignment
from torchmetrics.functional.classification import multiclass_confusion_matrix


def compute_purity(targets: torch.Tensor, preds: torch.Tensor) -> float:
    num_clusters = preds.max() + 1

    main_cls_count = 0
    for i in range(num_clusters):
        if len(targets[preds == i]) == 0:
            continue
        main_cls = targets[preds == i].mode().values.item()
        main_occurences = (targets[preds == i] == main_cls).float().sum()
        main_cls_count += main_occurences

    purity = main_cls_count / len(targets)

    return purity


class ClusterAccuracy(torchmetrics.Metric):
    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = True

    def __init__(self, num_classes: int, subset: str):
        super().__init__()
        self.num_classes = num_classes
        self.subset = subset
        self.add_state(
            "conf_matrix_old",
            default=torch.zeros((num_classes, num_classes)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "conf_matrix_new",
            default=torch.zeros((num_classes, num_classes)),
            dist_reduce_fx="sum",
        )

    def update(self, logits: torch.Tensor, targets: torch.Tensor, is_old_mask: torch.Tensor):
        # Padding for NCD case. doesn't change the result
        if logits.shape[1] < self.num_classes:
            logits = torch.nn.functional.pad(logits, (0, self.num_classes - logits.shape[1]))

        logits_new = logits[~is_old_mask]
        targets_new = targets[~is_old_mask]
        if len(targets_new) > 0:
            conf_matrix_new = multiclass_confusion_matrix(logits_new, targets_new, self.num_classes)
            self.conf_matrix_new += conf_matrix_new

        logits_old = logits[is_old_mask]
        targets_old = targets[is_old_mask]
        if len(targets_old) > 0:
            conf_matrix_old = multiclass_confusion_matrix(logits_old, targets_old, self.num_classes)
            self.conf_matrix_old += conf_matrix_old

    def compute(self):
        conf_matrix_old = self.conf_matrix_old.cpu().numpy()
        conf_matrix_new = self.conf_matrix_new.cpu().numpy()
        conf_matrix_all = conf_matrix_old + conf_matrix_new

        if conf_matrix_all.sum() == 0:
            return torch.tensor(0)

        row_ind, col_ind = linear_sum_assignment(conf_matrix_all, maximize=True)

        if self.subset == "all":
            conf_matrix = conf_matrix_all
        elif self.subset == "old":
            conf_matrix = conf_matrix_old
        elif self.subset == "new":
            conf_matrix = conf_matrix_new
        else:
            raise ValueError(f"Subset {self.subset} not supported")

        correct = conf_matrix[row_ind, col_ind].sum()
        total = conf_matrix.sum()

        if total == 0:
            return torch.tensor(0)
        else:
            return torch.tensor(correct / total)
