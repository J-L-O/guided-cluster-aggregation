import logging
from functools import partial
from typing import List, Dict, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from numpy.lib.npyio import NpzFile
from torch import nn
from torch.nn import BCELoss, ModuleDict

from util.metrics import ClusterAccuracy, compute_purity

log = logging.getLogger(__name__)


# Loss that maximizes the entropy of the class distribution
class EntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp to avoid NaNs
        x = torch.clamp(x, 1e-8, 1.0)
        return torch.sum(-x * torch.log(x))


class GCAModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        base_k: int,
        hierarchy_k: List[int],
        hierarchy_schedule: List[int],
        knn_file: str,
        num_known_classes: int,
        entropy_weight: float,
        optimizer: partial,
        scheduler: partial = None,
    ):
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.num_classes = num_classes
        self.base_k = base_k
        self.hierarchy_schedule = hierarchy_schedule
        self.hierarchy_k = hierarchy_k
        self.num_known_classes = num_known_classes

        # The current stage in the clustering hierarchy
        self.clustering_stage = self.get_current_stage()
        num_hierarchy_levels = len(hierarchy_schedule) + 1

        # We still load the model if no knn file is supplied, but can only do inference
        if knn_file is not None:
            knn = np.load(knn_file, allow_pickle=True)

            if isinstance(knn, NpzFile):
                confidences = torch.tensor(knn["confidences"])
                knn = knn["indices"]
            elif knn.dtype != object:
                knn = knn[:, :base_k]
                confidences = None
            else:
                confidences = None
            num_train_samples = knn.shape[0]

            self.knn_graphs = []

            for i in range(num_hierarchy_levels):
                if i == 0:
                    # Construct sparse adjacency matrix for kNN graph
                    num_knn = torch.tensor([len(neighbors) for neighbors in knn])

                    x_coords = torch.arange(num_train_samples).repeat_interleave(num_knn)

                    if knn.dtype != object:
                        y_coords = torch.tensor(knn.flatten())
                    else:
                        y_coords = torch.tensor([item for sublist in knn for item in sublist])
                    indices = torch.stack([x_coords, y_coords])

                    if confidences is not None:
                        values = confidences.flatten()
                    else:
                        values = torch.ones(x_coords.shape[0])
                else:
                    # Construct empty sparse adjacency matrix
                    indices = torch.zeros((2, 0))
                    values = torch.zeros(0)

                knn_graph = torch.sparse_coo_tensor(
                    indices,
                    values,
                    (num_train_samples, num_train_samples),
                    dtype=torch.float32,
                )

                self.knn_graphs.append(knn_graph)
        else:
            self.knn_graphs = None
            log.warning(f"knn_file is None, only inference is possible.")

        self.lowest_loss_heads = torch.zeros(num_hierarchy_levels, dtype=torch.int)

        self.bce_loss = BCELoss(reduction="none")
        self.entropy_loss = EntropyLoss()
        self.entropy_weight = entropy_weight
        self.val_splits = ["train", "val"]

        self.train_acc_unlabeled = ModuleDict(
            {
                "train/acc_unlabeled": ClusterAccuracy(num_classes, subset="all"),
                "train/acc_unlabeled_old": ClusterAccuracy(num_classes, subset="old"),
                "train/acc_unlabeled_new": ClusterAccuracy(num_classes, subset="new"),
            }
        )
        self.val_acc_train_unlabeled = ModuleDict(
            {
                "val/train_acc_unlabeled": ClusterAccuracy(num_classes, subset="all"),
                "val/train_acc_unlabeled_old": ClusterAccuracy(num_classes, subset="old"),
                "val/train_acc_unlabeled_new": ClusterAccuracy(num_classes, subset="new"),
            }
        )
        self.val_acc_val_unlabeled = ModuleDict(
            {
                "val/val_acc_unlabeled": ClusterAccuracy(num_classes, subset="all"),
                "val/val_acc_unlabeled_old": ClusterAccuracy(num_classes, subset="old"),
                "val/val_acc_unlabeled_new": ClusterAccuracy(num_classes, subset="new"),
            }
        )

    def on_save_checkpoint(self, checkpoint) -> None:
        """Customize checkpointing to save knn graphs and lowest loss heads"""

        checkpoint["knn_graphs"] = self.knn_graphs
        checkpoint["lowest_loss_heads"] = self.lowest_loss_heads

    def on_load_checkpoint(self, checkpoint) -> None:
        """Retrieve knn graphs and lowest loss heads from checkpoint"""

        self.knn_graphs = checkpoint["knn_graphs"]
        self.lowest_loss_heads = checkpoint["lowest_loss_heads"]

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.model.parameters())

        if self.scheduler is None:
            return optimizer

        scheduler = self.scheduler(optimizer=optimizer)
        scheduler_config = {"scheduler": scheduler}

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }

    def forward(self, batch: Dict):
        images = batch["anchors"]

        output = self.model(images)
        output["targets"] = batch["targets"]
        output["is_labeled"] = batch["is_labeled"]
        output["indices"] = batch["indices"]
        output["uq_indices"] = batch["uq_indices"]

        return output

    def get_current_stage(self) -> int:
        for i, epoch in enumerate(self.hierarchy_schedule):
            if self.current_epoch < epoch:
                return i
        return len(self.hierarchy_schedule)

    def compute_scan_loss(
        self,
        anchor_logits: torch.Tensor,
        neighbor_logits: torch.Tensor,
        indices: torch.Tensor,
        neighbor_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchors_probs = torch.softmax(anchor_logits, dim=1)
        neighbors_probs = torch.softmax(neighbor_logits, dim=1)

        same_class_probs = torch.mul(anchors_probs, neighbors_probs).sum(dim=1)

        # Fetch the certainty from the kNN graph
        knn_graph = self.knn_graphs[0].to(self.device)
        sample_weights = torch.index_select(knn_graph, dim=0, index=indices)
        sample_weights = torch.index_select(sample_weights, dim=1, index=neighbor_indices)
        sample_weights = sample_weights.float().to_dense().diagonal()
        same_class_targets = torch.ones_like(same_class_probs)

        clustering_loss = self.bce_loss(same_class_probs, same_class_targets)
        clustering_loss = clustering_loss * sample_weights
        clustering_loss = clustering_loss.sum() / sample_weights.sum()

        mean_anchors_probs = torch.mean(anchors_probs, dim=0)
        regularization_loss = -self.entropy_loss(mean_anchors_probs)

        loss = clustering_loss + self.entropy_weight * regularization_loss

        return loss, clustering_loss, regularization_loss

    def compute_hierarchical_loss(
        self,
        logits: torch.Tensor,
        indices: torch.Tensor,
        stage: int,
    ) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        same_class_probs = torch.matmul(probs, probs.T)
        indices = indices.cpu()

        # Create BCE targets based on KNN graph
        knn_graph = self.knn_graphs[stage]
        same_class_targets = torch.index_select(knn_graph, dim=0, index=indices)
        same_class_targets = torch.index_select(same_class_targets, dim=1, index=indices)
        same_class_targets = same_class_targets.float().to_dense().to(logits.device)

        # Compute the loss
        loss = self.bce_loss(same_class_probs, same_class_targets)
        loss = torch.mean(loss)

        return loss

    def on_train_start(self) -> None:
        """Determine the current stage in the clustering hierarchy"""
        self.clustering_stage = self.get_current_stage()

    def on_validation_start(self) -> None:
        """Determine the current stage in the clustering hierarchy"""
        self.clustering_stage = self.get_current_stage()

    def on_train_epoch_start(self) -> None:
        self.clustering_stage = self.get_current_stage()

    def compute_stage_losses(
        self,
        anchors_output: Dict,
        neighbors: torch.Tensor,
        indices: torch.Tensor,
        neighbor_indices: torch.Tensor,
    ) -> List[torch.Tensor]:

        stage_losses = []
        first_bce_stage = 1

        # SCAN loss for the first stage
        neighbors_output = self.model(neighbors)

        # First compute the standard SCAN loss for the first stage
        anchors_logits_base = anchors_output["logits"][0]
        neighbors_logits_base = neighbors_output["logits"][0]

        clustering_losses = []
        entropy_losses = []
        total_losses = []

        for anchor_head_logits, neighbor_head_logits in zip(anchors_logits_base, neighbors_logits_base):
            loss_head = self.compute_scan_loss(
                anchor_head_logits,
                neighbor_head_logits,
                indices,
                neighbor_indices,
            )
            total_losses.append(loss_head[0])
            clustering_losses.append(loss_head[1])
            entropy_losses.append(loss_head[2])

        clustering_losses = torch.stack(clustering_losses)
        entropy_losses = torch.stack(entropy_losses)
        total_losses = torch.stack(total_losses)

        stage_losses.append((total_losses, clustering_losses, entropy_losses))

        # Then compute the loss for the other stages
        for stage in range(first_bce_stage, self.clustering_stage + 1):
            logits = anchors_output["logits"][stage]

            head_losses = []

            for head_logits in logits:
                loss_head = self.compute_hierarchical_loss(
                    head_logits,
                    indices,
                    stage,
                )
                head_losses.append(loss_head)

            head_losses = torch.stack(head_losses)
            stage_losses.append(head_losses)

        return stage_losses

    def pad_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Pad logits with zeros if the model uses fewer classes than the dataset.
        This can happen if the number of classes is estimated.
        """
        if logits.shape[1] < self.num_classes:
            padded_logits = torch.cat(
                [
                    logits,
                    torch.zeros(
                        logits.shape[0],
                        self.num_classes - logits.shape[1],
                        ).to(logits.device),
                ],
                dim=1,
            )
        else:
            padded_logits = logits

        return padded_logits

    def training_step(self, batch: dict, batch_idx: int):
        anchors = batch["anchors"]
        neighbors = batch["neighbors"]
        indices = batch["indices"]
        neighbor_indices = batch["neighbor_indices"]
        targets = batch["targets"]
        is_labeled = batch["is_labeled"]

        anchors_output = self.model(anchors)

        stage_losses = self.compute_stage_losses(
            anchors_output,
            neighbors,
            indices,
            neighbor_indices,
        )

        loss_per_level = []

        mean_scan_loss = stage_losses[0][0].mean()
        mean_clustering_loss = stage_losses[0][1].mean()
        mean_entropy_loss = stage_losses[0][2].mean()

        self.log(f"train/clustering_loss_base", mean_clustering_loss)
        self.log(f"train/entropy_loss_base", mean_entropy_loss)
        self.log(f"train/loss_base", mean_scan_loss)
        loss_per_level.append(mean_scan_loss)

        for stage in range(1, len(stage_losses)):
            mean_stage_loss = stage_losses[stage].mean()

            self.log(f"train/loss_stage_{stage}", mean_stage_loss)
            loss_per_level.append(mean_stage_loss)

        # Compute the total loss
        total_loss = torch.mean(torch.stack(loss_per_level))

        # Compute cluster accuracy
        final_logits_unlabeled = anchors_output["logits"][-1][0][~is_labeled]
        targets_unlabeled = targets[~is_labeled]
        is_old_mask = targets_unlabeled < self.num_known_classes

        final_logits_unlabeled = self.pad_logits(final_logits_unlabeled)

        for tag, metric in self.train_acc_unlabeled.items():
            metric.update(final_logits_unlabeled, targets_unlabeled, is_old_mask)
            self.log(
                tag,
                metric,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        return total_loss

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        split = self.val_splits[dataloader_idx]

        anchors = batch["anchors"]
        targets = batch["targets"]
        indices = batch["indices"]
        is_labeled = batch["is_labeled"]

        anchors_output = self.model(anchors)

        features = anchors_output["features"]

        # Compute cluster accuracy
        val_acc = self.val_acc_train_unlabeled if split == "train" else self.val_acc_val_unlabeled
        final_logits_unlabeled = anchors_output["logits"][-1][0][~is_labeled]
        targets_unlabeled = targets[~is_labeled]
        is_old_mask = targets_unlabeled < self.num_known_classes

        final_logits_unlabeled = self.pad_logits(final_logits_unlabeled)

        for tag, metric in val_acc.items():
            metric.update(final_logits_unlabeled, targets_unlabeled, is_old_mask)
            self.log(
                tag,
                metric,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                add_dataloader_idx=False,
            )

        predictions = []
        probabilities = []
        for stage in range(len(self.hierarchy_schedule) + 1):
            predictions_stage = []
            probabilities_stage = []
            logits_stage = anchors_output["logits"][stage]

            for head_logits in logits_stage:
                preds = torch.argmax(head_logits, dim=1)
                probs = torch.softmax(head_logits, dim=1)
                predictions_stage.append(preds)
                probabilities_stage.append(probs)

            predictions.append(predictions_stage)
            probabilities.append(probabilities_stage)

        output = {
            "features": features,
            "predictions": predictions,
            "probabilities": probabilities,
            "targets": targets,
            "indices": indices,
            "is_labeled": is_labeled,
        }

        if split == "train":
            neighbors = batch["neighbors"]
            neighbor_indices = batch["neighbor_indices"]

            stage_losses = self.compute_stage_losses(
                anchors_output,
                neighbors,
                indices,
                neighbor_indices,
            )

            # Only keep total loss
            stage_losses[0] = stage_losses[0][0]

            output["stage_losses"] = stage_losses

        # TODO: Refactor this
        # Move the output to the CPU
        for key in output.keys():
            if isinstance(output[key], torch.Tensor):
                output[key] = output[key].cpu()
            elif isinstance(output[key], list):
                if isinstance(output[key][0], torch.Tensor):
                    for i, head in enumerate(output[key]):
                        output[key][i] = head.cpu()
                elif isinstance(output[key][0], list):
                    for i in range(len(output[key])):
                        for j, head in enumerate(output[key][i]):
                            output[key][i][j] = head.cpu()

        return output

    @staticmethod
    def merge_dl_output(dl_output: List[Dict]) -> Dict:
        output = {}
        keys = dl_output[0].keys()

        for key in keys:
            dl_output_key = [batch[key] for batch in dl_output]

            if isinstance(dl_output_key[0], torch.Tensor):
                output[key] = torch.cat(dl_output_key, dim=0)
            elif isinstance(dl_output_key[0], list):
                num_batches = len(dl_output_key)
                hierarchy_levels = len(dl_output_key[0])
                merged_list = []

                for level in range(hierarchy_levels):
                    num_heads = len(dl_output_key[0][level])
                    merged_heads = []

                    for head in range(num_heads):
                        output_head = [dl_output_key[i][level][head] for i in range(num_batches)]

                        # Stack if the head is a scalar, otherwise concatenate
                        if output_head[0].ndim == 0:
                            merged_output_head = torch.stack(output_head, dim=0)
                        else:
                            merged_output_head = torch.cat(output_head, dim=0)

                        merged_heads.append(merged_output_head)

                    merged_list.append(merged_heads)

                output[key] = merged_list
            else:
                raise NotImplementedError

        return output

    def compute_next_stage_targets(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        is_labeled: torch.Tensor,
        indices: torch.Tensor,
        stage: int,
    ) -> torch.Tensor:
        base_knn_graph = self.knn_graphs[0]
        ordered_indices = torch.argsort(indices, dim=0)
        ordered_probs = probs[ordered_indices]
        ordered_preds = ordered_probs.argmax(dim=1)
        ordered_targets = targets[ordered_indices]
        ordered_is_labeled = is_labeled[ordered_indices]

        num_clusters = self.model.class_hierarchy[stage]

        knn_graph = base_knn_graph.float().to_dense()

        # Aggregate the kNN graph
        pooled_knn_graph = ordered_probs.transpose(0, 1) @ knn_graph
        pooled_knn_graph = pooled_knn_graph @ ordered_probs

        stage_k = self.hierarchy_k[stage]

        # Include the supervision
        ordered_labeled_preds = ordered_preds[ordered_is_labeled]
        ordered_labeled_targets = ordered_targets[ordered_is_labeled]
        known_classes_per_cluster = torch.zeros((pooled_knn_graph.shape[0], self.num_known_classes))
        for i in range(num_clusters):
            classes, counts = torch.unique(
                ordered_labeled_targets[ordered_labeled_preds == i],
                return_counts=True,
            )
            known_classes_per_cluster[i].index_add_(0, classes, counts.float())
        samples_per_cluster = known_classes_per_cluster.sum(dim=1)
        possible_matchings = samples_per_cluster.unsqueeze(1) @ samples_per_cluster.unsqueeze(0)
        equal = known_classes_per_cluster @ known_classes_per_cluster.T
        not_equal = possible_matchings - equal

        pooled_knn_graph = pooled_knn_graph + equal - not_equal

        # Indentify the k nearest clusters
        k_nearest_clusters = pooled_knn_graph.topk(stage_k, dim=1).indices

        # Construct the sparse cluster adjacency matrix
        x_coords = torch.arange(num_clusters).repeat_interleave(stage_k)
        y_coords = k_nearest_clusters.flatten()
        adjacency_indices = torch.stack([x_coords, y_coords])

        values = torch.ones(adjacency_indices.shape[1])

        cluster_adjacency = torch.sparse_coo_tensor(
            adjacency_indices,
            values,
            (num_clusters, num_clusters),
            dtype=torch.float,
        )

        # Create a sparse tensor from the ordered predictions
        x_coords = torch.arange(ordered_preds.shape[0])
        y_coords = ordered_preds
        values = torch.ones(ordered_preds.shape[0])

        preds_graph = torch.sparse_coo_tensor(
            torch.stack([x_coords, y_coords]),
            values,
            size=(ordered_preds.shape[0], num_clusters),
        )

        preds_graph = preds_graph.cpu()

        # Generate the targets for the next stage
        next_stage_targets = preds_graph @ cluster_adjacency
        next_stage_targets = next_stage_targets @ preds_graph.transpose(0, 1)

        return next_stage_targets

    def should_compute_aggregation_for_stage(self, stage: int) -> bool:
        for i, epoch in enumerate(self.hierarchy_schedule):
            correct_stage = i == stage
            correct_epoch = self.current_epoch + 1 >= epoch

            if correct_stage and correct_epoch:
                return True
        return False

    def validation_epoch_end(self, outputs: List):
        for dataloader_idx in range(len(outputs)):
            split = self.val_splits[dataloader_idx]
            dl_output = outputs[dataloader_idx]

            merged_output = self.merge_dl_output(dl_output)
            targets = merged_output["targets"]
            indices = merged_output["indices"]
            is_labeled = merged_output["is_labeled"]

            for stage in range(self.clustering_stage + 1):

                # Find lowest loss head
                if split == "train":
                    stage_losses = merged_output["stage_losses"][stage]
                    stage_losses = torch.stack(stage_losses, dim=0)

                    mean_loss_per_head = torch.mean(stage_losses, dim=1)
                    lowest_loss_head = torch.argmin(mean_loss_per_head)

                    self.lowest_loss_heads[stage] = lowest_loss_head

                preds_stage = merged_output["predictions"][stage]
                probs_stage = merged_output["probabilities"][stage]
                lowest_loss_preds = preds_stage[self.lowest_loss_heads[stage]]

                # Compute targets for the next stage
                if split == "train" and self.should_compute_aggregation_for_stage(stage):
                    aggregated_targets = self.compute_next_stage_targets(
                        probs_stage[self.lowest_loss_heads[stage]],
                        targets,
                        is_labeled,
                        indices,
                        stage,
                    )

                    self.knn_graphs[stage + 1] = aggregated_targets.to(torch.uint8)

                purity_all = compute_purity(targets, lowest_loss_preds)
                purity_unlabeled = compute_purity(targets[~is_labeled], lowest_loss_preds[~is_labeled])

                self.log(
                    f"{split}/purity_stage_{stage}",
                    purity_all,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
                self.log(
                    f"{split}/purity_unlabeled_stage_{stage}",
                    purity_unlabeled,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
