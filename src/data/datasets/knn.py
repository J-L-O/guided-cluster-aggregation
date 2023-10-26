import logging

from numpy.lib.npyio import NpzFile
from torch.utils.data import Dataset
import numpy as np

log = logging.getLogger(__name__)


class KNNDataset(Dataset):
    def __init__(
        self,
        dataset,
        knn_path,
        num_neighbors,
    ):
        self.dataset = dataset
        self.num_neighbors = num_neighbors

        # Use dummy knn if knn_path is None
        if knn_path is None:
            log.warning("Using dummy knn")
            self.knns = np.arange(len(dataset))[:, None]
        else:
            self.knns = np.load(knn_path, allow_pickle=True)

            if isinstance(self.knns, NpzFile):
                self.knns = self.knns["indices"]

    def __getitem__(self, index):
        anchor, target, uq_idx, is_labeled = self.dataset[index]

        # Pick a random neighbor
        neighbor_indices = self.knns[index][: self.num_neighbors]
        neighbor_index = np.random.choice(neighbor_indices)
        neighbor, neighbor_target, neighbor_uq_idx, _ = self.dataset[neighbor_index]

        return {
            "anchors": anchor,
            "neighbors": neighbor,
            "targets": target,
            "neighbor_targets": neighbor_target,
            "indices": index,
            "neighbor_indices": neighbor_index,
            "uq_indices": uq_idx,
            "neighbor_uq_indices": neighbor_uq_idx,
            "is_labeled": is_labeled,
        }

    def __len__(self):
        return len(self.dataset)
