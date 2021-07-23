from torch.utils.data import Dataset
from pathlib import Path
import json
import pickle
import mxnet.recordio as mxrec

from util.util import ref_copy
from util.dc import from_numpy, to_long


def collate_fn(batch):
    batch = from_numpy(batch)
    batch = to_long(batch) # id to long
    return_batch = dict()
    # Batching by use a list for non-fixed size
    for key in batch[0].keys():
        return_batch[key] = [x[key] for x in batch]
    return return_batch


class ArgoDatasetLaneGCN(Dataset):
    def __init__(self, cfg, mode='train'):
        self.prefix = Path(cfg.path)
        self.mode = mode
        meta_path = self.prefix / f'{mode}.meta'
        with meta_path.open('r') as f:
            meta_info = json.load(f)
            self.num_sample = meta_info['num_sample']
        print('num samples:', self.num_sample)
        self.record_reader = mxrec.MXIndexedRecordIO(str(self.prefix / f'{mode}.idx'), str(self.prefix / f'{mode}.rec'), 'r')

    def __len__(self):
        return self.num_sample
        # return 1000

    def __getitem__(self, idx):

        data = pickle.loads(self.record_reader.read_idx(idx))
        new_data = ref_copy(data)
        for key in ['file_id', 'hist_traj', 'fur_traj', 'city', 'orig', 'gt_preds', 'has_preds', 'theta', 'rot', 'feats', 'ctrs', 'graph']:
            if key in data:
                new_data[key] = ref_copy(data[key])
        return new_data

    def __del__(self):
        self.record_reader.close()

def get_toy_dataloader():
    from dataset import get_dataloader
    from easydict import EasyDict as edict

    args = edict({
        'data_name': 'lanegcn_heter',
        'data_version': 'debug',
        'mode': 'train',
        'is_dist': False,
        'batch_size': 2,
        'shuffle': False,
        'num_workers': 2,
        'pin_memory': False
    })
    dataloader = get_dataloader(args)
    return dataloader


if __name__ == '__main__':
    dl = get_toy_dataloader()
    for sample in dl:
        print(sample.keys())
        print(sample['graph'][0].keys())
        break