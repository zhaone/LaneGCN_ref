import json
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from dataset.ArgoDatasetLaneGCN import ArgoDatasetLaneGCN
from dataset.ArgoDatasetLaneGCN import collate_fn as argo_collate_fn_lh
from torch.utils.data.distributed import DistributedSampler

local_rank = 0

_dataset_cfg = edict({
    'lanegcn': {
        'debug': {
            'path': '/workspace/datasets/argo/LaneGCN_debug',
            'meta_path': None
        },
        'full': {
            'path': '/workspace/datasets/argo/LaneGCN',
            'meta_path': None
        }
    }
})


def get_dataset_cfg(name, version):
    print(name, version)
    cfg = _dataset_cfg[name][version]
    if cfg.meta_path is not None:
        with open(cfg.meta_path, 'r') as f:
            cfg.META = edict(json.load(f))
    return cfg


def get_dataset(args, mode="train"):
    cfg = get_dataset_cfg(args.data_name, args.data_version)
    if args.data_name == 'lanegcn':
        dataset = ArgoDatasetLaneGCN(cfg, mode=mode)
        collate_fn = argo_collate_fn_lh
    else:
        raise NotImplementedError

    return dataset, collate_fn

def _worker_init_fn(pid):
    pass

def get_dataloader(args, mode="train"):
    dataset, collate_fn = get_dataset(args, mode)
    sampler = None
    if args.is_dist:
        sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=args.local_rank)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        shuffle=args.shuffle if mode=='train' else False,
        pin_memory=args.pin_memory,
        worker_init_fn=_worker_init_fn,
        drop_last=True if mode == 'train' else False
    )
    return dataloader