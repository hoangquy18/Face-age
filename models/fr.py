import os.path as osp

import torch
import torch.distributed as dist
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
from torch import amp

from common.sampler import RandomSampler
from common.data_prefetcher import DataPrefetcher
from common.ops import (
    convert_to_ddp,
    get_dex_age,
    age2group,
    apply_weight_decay,
    reduce_loss,
)
from common.grl import GradientReverseLayer
from . import BasicTask
from backbone.aifr import backbone_dict, AgeEstimationModule
from head.cosface import CosFace
from common.dataset import TrainImageDataset


class FR(BasicTask):

    def set_loader(self):
        opt = self.opt

        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize([opt.image_size, opt.image_size]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[
                        0.5,
                    ],
                    std=[
                        0.5,
                    ],
                ),
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.Resize([opt.image_size, opt.image_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        train_dataset = TrainImageDataset(
            opt.dataset_name,
            train_transform,
            data_root=opt.dataset_root,
            list_path=opt.dataset_list,
        )

        # Deterministic train/val split. Val set is a hold-out subset of
        # the training list (same class space, unseen samples) so we can
        # measure id_acc / age_group_acc / age_mae on data not used for SGD.
        val_split = float(getattr(opt, "val_split", 0.0) or 0.0)
        train_indices = None
        val_indices = None
        if val_split > 0.0:
            n_total = len(train_dataset)
            rng = np.random.RandomState(int(getattr(opt, "val_seed", 12345)))
            perm = rng.permutation(n_total)
            n_val = max(1, int(round(n_total * val_split)))
            n_val = min(n_val, n_total - 1)
            val_indices = perm[:n_val].tolist()
            train_indices = perm[n_val:].tolist()

        weights = None
        sampler = RandomSampler(
            train_dataset,
            batch_size=opt.batch_size,
            num_iter=opt.num_iter,
            restore_iter=opt.restore_iter,
            weights=weights,
            indices_pool=train_indices,
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            sampler=sampler,
            pin_memory=True,
            num_workers=opt.num_worker,
            drop_last=True,
        )
        self.prefetcher = DataPrefetcher(train_loader)

        self.val_loader = None
        if val_indices is not None:
            # Use the same image list but with deterministic, val-friendly
            # transforms (no random flip). We share the underlying file index
            # via Subset over a parallel dataset that uses val_transform.
            val_base = TrainImageDataset(
                opt.dataset_name,
                val_transform,
                data_root=opt.dataset_root,
                list_path=opt.dataset_list,
            )
            val_subset = torch.utils.data.Subset(val_base, val_indices)
            val_batch_size = int(getattr(opt, "val_batch_size", 0) or 0) or opt.batch_size
            if dist.is_initialized() and dist.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(
                    val_subset, shuffle=False, drop_last=False
                )
            else:
                val_sampler = torch.utils.data.SequentialSampler(val_subset)
            self.val_loader = torch.utils.data.DataLoader(
                val_subset,
                batch_size=val_batch_size,
                sampler=val_sampler,
                pin_memory=True,
                num_workers=max(2, opt.num_worker // 4),
                drop_last=False,
                persistent_workers=True,
            )

    def set_model(self):
        opt = self.opt
        backbone = backbone_dict[opt.backbone_name](input_size=opt.image_size)
        head = CosFace(
            in_features=512,
            out_features=len(self.prefetcher.__loader__.dataset.classes),
            s=opt.head_s,
            m=opt.head_m,
        )

        estimation_network = AgeEstimationModule(
            input_size=opt.image_size, age_group=opt.age_group
        )

        da_discriminator = AgeEstimationModule(
            input_size=opt.image_size, age_group=opt.age_group
        )

        optimizer = torch.optim.SGD(
            list(backbone.parameters())
            + list(head.parameters())
            + list(estimation_network.parameters())
            + list(da_discriminator.parameters()),
            momentum=opt.momentum,
            lr=opt.learning_rate,
        )

        backbone, head, estimation_network, da_discriminator = convert_to_ddp(
            backbone, head, estimation_network, da_discriminator
        )
        scaler = amp.GradScaler("cuda")
        self.optimizer = optimizer
        self.backbone = backbone
        self.head = head
        self.estimation_network = estimation_network
        self.da_discriminator = da_discriminator
        self.grl = GradientReverseLayer()
        self.scaler = scaler

        self.logger.modules = [
            optimizer,
            backbone,
            head,
            estimation_network,
            da_discriminator,
            scaler,
        ]
        if opt.restore_iter > 0:
            self.logger.load_checkpoints(opt.restore_iter)

    def validate(self, n_iter):
        opt = self.opt
        metrics = {}

        if self.val_loader is not None:
            metrics.update(self._validate_holdout())

        if getattr(opt, "val_test_root", None):
            metrics.update(self._validate_verification())

        if metrics:
            self.logger.msg(metrics, n_iter, tag="VAL")

        if dist.is_initialized():
            dist.barrier()

    def _validate_holdout(self):
        """Forward the hold-out val subset once and report id_acc / age_group_acc / age_mae.

        Runs on all ranks (each shard via DistributedSampler), then sums counters
        with all_reduce so the printed metric is over the full val set.
        """
        opt = self.opt
        self.backbone.eval()
        self.estimation_network.eval()
        self.head.eval()

        head_module = self.head.module if hasattr(self.head, "module") else self.head

        device = next(self.backbone.parameters()).device
        counters = torch.zeros(4, dtype=torch.float64, device=device)
        # [id_correct, age_group_correct, age_abs_err_sum, total]

        try:
            with torch.no_grad():
                for batch in self.val_loader:
                    images, labels, ages, genders = batch
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    ages = ages.to(device, non_blocking=True)

                    with amp.autocast("cuda", enabled=opt.amp):
                        embedding, _x_id, x_age = self.backbone(
                            images, return_age=True
                        )
                        x_age_logits, x_group = self.estimation_network(x_age)
                    embedding = embedding.float()
                    x_age_logits = x_age_logits.float()
                    x_group = x_group.float()

                    cosine = F.linear(
                        F.normalize(embedding), F.normalize(head_module.weight)
                    )
                    counters[0] += (cosine.argmax(dim=1) == labels).sum().double()

                    age_pred = get_dex_age(x_age_logits)
                    counters[2] += (age_pred - ages).abs().sum().double()

                    age_group_gt = age2group(ages, age_group=opt.age_group).long()
                    counters[1] += (
                        x_group.argmax(dim=1) == age_group_gt
                    ).sum().double()

                    counters[3] += labels.numel()
        finally:
            self.backbone.train()
            self.estimation_network.train()
            self.head.train()

        if dist.is_initialized():
            dist.all_reduce(counters, op=dist.ReduceOp.SUM)

        total = counters[3].item()
        if total <= 0:
            return {}
        return {
            "id_acc": counters[0].item() / total,
            "age_group_acc": counters[1].item() / total,
            "age_mae": counters[2].item() / total,
        }

    def _validate_verification(self):
        """Run InsightFace-style verification benchmarks (LFW / AGEDB-30 / ...)
        located under --val_test_root. Returns {<name>_acc, <name>_tar1e4}.

        Only rank 0 does the actual encoding/scoring (benchmarks are small,
        ~10K images each); other ranks wait at the barrier in `validate()`.
        Embeddings come from the bare backbone (no margin head needed).
        """
        opt = self.opt
        if self.logger.local_rank != 0:
            return {}

        from glob import glob

        # Lazy import to avoid pulling test code into normal training.
        import sys

        repo_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from evaluate_arcface_test_set import (  # type: ignore
            parse_pair_file,
            encode_all,
            accuracy_at_best_threshold,
            tar_at_far,
        )

        if not hasattr(self, "_val_benchmarks"):
            test_root = osp.abspath(opt.val_test_root)
            pair_files = sorted(glob(osp.join(test_root, "*.txt")))
            self._val_benchmarks = []
            for p in pair_files:
                name = osp.splitext(osp.basename(p))[0]
                image_dir = osp.join(test_root, name)
                if osp.isdir(image_dir):
                    self._val_benchmarks.append((name, p, image_dir))
            self._val_tfm = transforms.Compose(
                [
                    transforms.Resize([opt.image_size, opt.image_size]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ]
            )

        self.backbone.eval()
        try:
            backbone = (
                self.backbone.module if hasattr(self.backbone, "module") else self.backbone
            )
            device = next(backbone.parameters()).device
            val_bs = int(getattr(opt, "val_batch_size", 0) or 0) or opt.batch_size

            results = {}
            for name, pair_path, image_dir in self._val_benchmarks:
                pairs, unique_names = parse_pair_file(pair_path)
                if not pairs:
                    continue
                emb_map = encode_all(
                    backbone, image_dir, unique_names, self._val_tfm, device, val_bs
                )
                scores_list, y_list = [], []
                for a, b, lab in pairs:
                    ea = emb_map[a].numpy()
                    eb = emb_map[b].numpy()
                    scores_list.append(float(np.dot(ea, eb)))
                    y_list.append(1 if lab == 1 else 0)
                scores = np.asarray(scores_list, dtype=np.float64)
                y_true = np.asarray(y_list, dtype=np.int32)
                acc, _ = accuracy_at_best_threshold(scores, y_true)
                tar, _ = tar_at_far(scores, y_true, far=1e-4)
                results[f"{name}_acc"] = acc * 100.0
                results[f"{name}_tar1e4"] = tar * 100.0
        finally:
            self.backbone.train()

        return results

    def adjust_learning_rate(self, step):
        assert step > 0, "batch index should large than 0"
        opt = self.opt
        if step > opt.warmup:
            lr = opt.learning_rate * (
                opt.gamma ** np.sum(np.array(opt.milestone) < step)
            )
        else:
            lr = step * opt.learning_rate / opt.warmup
        lr = max(1e-4, lr)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def compute_age_loss(self, x_age, x_group, ages):
        opt = self.opt
        age_loss = F.mse_loss(get_dex_age(x_age), ages) + F.cross_entropy(
            x_group, age2group(ages, age_group=opt.age_group).long()
        )
        return age_loss

    def forward_da(self, x_id, ages):
        x_age, x_group = self.da_discriminator(self.grl(x_id))
        loss = self.compute_age_loss(x_age, x_group, ages)
        return loss

    def train(self, inputs, n_iter):
        opt = self.opt

        images, labels, ages, genders = inputs
        self.backbone.train()
        self.head.train()
        self.da_discriminator.train()
        self.estimation_network.train()

        if opt.amp:
            with amp.autocast("cuda"):
                embedding, x_id, x_age = self.backbone(images, return_age=True)
            embedding = embedding.float()
            x_id = x_id.float()
            x_age = x_age.float()
        else:
            embedding, x_id, x_age = self.backbone(images, return_age=True)

        ######## Train Face Recognition
        head_logits = self.head(embedding, labels)
        id_loss = F.cross_entropy(head_logits, labels)
        x_age, x_group = self.estimation_network(x_age)
        age_loss = self.compute_age_loss(x_age, x_group, ages)
        da_loss = self.forward_da(x_id, ages)
        loss = (
            id_loss
            + age_loss * opt.fr_age_loss_weight
            + da_loss * opt.fr_da_loss_weight
        )

        total_loss = loss
        if opt.amp:
            total_loss = self.scaler.scale(loss)
        self.optimizer.zero_grad()
        total_loss.backward()
        apply_weight_decay(
            self.backbone,
            self.head,
            self.estimation_network,
            weight_decay_factor=opt.weight_decay,
            wo_bn=True,
        )
        if opt.amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        # Training-time metrics (no grad). id_acc is computed from pure
        # cosine logits (no CosFace margin), so it reflects the true top-1
        # rate rather than the harder margin-adjusted decision used by the
        # loss.
        with torch.no_grad():
            head_module = self.head.module if hasattr(self.head, "module") else self.head
            cosine = F.linear(
                F.normalize(embedding.detach()), F.normalize(head_module.weight)
            )
            id_acc = (cosine.argmax(dim=1) == labels).float().mean()

            age_pred = get_dex_age(x_age.detach())
            age_mae = (age_pred - ages).abs().mean()

            age_group_gt = age2group(ages, age_group=opt.age_group).long()
            age_group_acc = (
                x_group.detach().argmax(dim=1) == age_group_gt
            ).float().mean()

        id_loss, da_loss, age_loss, id_acc, age_group_acc, age_mae = reduce_loss(
            id_loss, da_loss, age_loss, id_acc, age_group_acc, age_mae
        )
        lr = self.optimizer.param_groups[0]["lr"]
        self.logger.msg(
            [id_loss, da_loss, age_loss, id_acc, age_group_acc, age_mae, lr],
            n_iter,
        )
