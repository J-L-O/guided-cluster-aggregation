from abc import ABC
from pathlib import Path
from typing import List, Dict

import faiss
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import numpy as np


class Evaluator(ABC):
    def evaluate(self, outputs: Dict):
        pass


class ClusterAccuracyEvaluator(Evaluator):
    def evaluate(self, outputs: Dict):
        logits = outputs["logits"][-1][0]
        targets = outputs["targets"]
        is_labeled = outputs["is_labeled"]
        predictions = logits.argmax(dim=1)

        # Only evaluate unlabeled samples
        unlabeled_predictions = predictions[~is_labeled]
        unlabeled_targets = targets[~is_labeled]

        conf_matrix = confusion_matrix(unlabeled_targets, unlabeled_predictions)
        row_ind, col_ind = linear_sum_assignment(conf_matrix, maximize=True)
        correct = conf_matrix[row_ind, col_ind].sum()
        total = conf_matrix.sum()

        cluster_accuracy = correct / total

        print(f"{cluster_accuracy=}")


class KNNEvaluator(Evaluator):
    def __init__(
        self,
        k_list: List[int],
        k_save: int,
        save_path: str,
        metric: str = "euclidean",
    ):
        assert metric in ["euclidean", "cosine"], f"Invalid metric {metric}"

        self.k_list = k_list
        self.k_save = k_save
        self.metric = metric

        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

    def evaluate(self, outputs: Dict):
        features = outputs["features"]
        targets = outputs["targets"]

        if self.metric == "cosine":
            features = torch.nn.functional.normalize(features, dim=1)

        # Create faiss index
        feature_dim = features.shape[1]
        features = features.cpu().numpy()

        if self.metric == "cosine":
            index = faiss.IndexFlatIP(feature_dim)
        else:
            index = faiss.IndexFlatL2(feature_dim)
        index = faiss.index_cpu_to_all_gpus(index)

        search_k = max(self.k_list)
        index.add(features)
        distances, indices = index.search(features, search_k)

        for k in self.k_list:
            # Get k nearest neighbors of unlabeled samples
            indices_k = indices[:, :k]

            # Compute accuracy of k-nearest neighbors
            knn_targets = targets[indices_k]
            accuracy = torch.eq(knn_targets, targets[:, None]).float().mean()

            print(f"KNN accuracy ({k=}): {accuracy}")

            if k == self.k_save:
                save_path = self.save_path / f"knn_{k}.npy"
                np.save(save_path, indices_k)
