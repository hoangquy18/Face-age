import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler


class RandomSampler(Sampler):

    def __init__(
        self,
        dataset,
        batch_size=0,
        num_iter=None,
        restore_iter=0,
        weights=None,
        replacement=True,
        seed=0,
        indices_pool=None,
    ):
        # PyTorch 2.2+ Sampler no longer uses data_source in the base __init__;
        # passing dataset can raise TypeError (object.__init__ gets extra args).
        super().__init__()
        self.dist = dist.is_initialized()
        if self.dist:
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0
        self.dataset = dataset
        self.batch_size = batch_size * self.num_replicas
        self.num_samples = num_iter * self.batch_size
        self.restore = restore_iter * self.batch_size
        self.weights = weights
        self.replacement = replacement
        self.seed = seed
        # Optional restriction to a subset of dataset indices (e.g. train split
        # when a hold-out val split is used). When None, samples from the full
        # dataset range [0, len(dataset)).
        if indices_pool is not None:
            self.indices_pool = torch.as_tensor(indices_pool, dtype=torch.long)
        else:
            self.indices_pool = None

    def __iter__(self):
        # deterministically shuffle
        g = torch.Generator()
        g.manual_seed(self.seed)
        if self.weights is None:
            if self.indices_pool is not None:
                pool = self.indices_pool
            else:
                pool = torch.arange(len(self.dataset), dtype=torch.long)
            n = pool.numel()
            n = n - n % self.batch_size
            pool = pool[:n] if n > 0 else pool
            epochs = self.num_samples // max(n, 1) + 1
            indices = []
            for e in range(epochs):
                g = torch.Generator()
                g.manual_seed(self.seed + e)
                # drop last
                perm = torch.randperm(n, generator=g)
                indices.extend(pool[perm].tolist())
            indices = indices[: self.num_samples]
        else:
            sampled = torch.multinomial(
                self.weights, self.num_samples, self.replacement, generator=g
            ).tolist()
            if self.indices_pool is not None:
                pool_list = self.indices_pool.tolist()
                indices = [pool_list[i] for i in sampled]
            else:
                indices = sampled

        # subsample
        indices = indices[
            self.restore + self.rank : self.num_samples : self.num_replicas
        ]

        return iter(indices)

    def __len__(self):
        return self.num_samples - self.restore
