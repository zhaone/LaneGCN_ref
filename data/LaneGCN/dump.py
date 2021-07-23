from pathlib import Path
import pickle
import time
import json
import gc
import numpy as np
from multiprocessing import Process
from tqdm import tqdm
from data.LaneGCN.traj import TrajExtractor
from data.LaneGCN.hdmap import GraphExtractor
from data.LaneGCN.config import data_config

DEBUG = True
FORCE_OVERWRITE = False
sources_dirs = {
    'train': '/workspace/datasets/argo/forecasting/train/data',
    'val': '/workspace/datasets/argo/forecasting/val/data',
    'test': '/workspace/datasets/argo/forecasting/test_obs/data',
}

def check_nan(data):
    if isinstance(data, dict):
        for k, v in data.items():
            if check_nan(v):
                print(k)
                return True
    elif isinstance(data, list):
        for d in data:
            if check_nan(d):
                return True
    elif isinstance(data, np.ndarray):
        return np.isnan(data).any()
    return False

def processor(pid, source_dir, output_dir, start, end, mode):
    te = TrajExtractor(root_dir=source_dir, config=data_config, mode=mode)
    ge = GraphExtractor(config=data_config, mode=mode)
    for i in tqdm(range(start, end)):
        if not FORCE_OVERWRITE:
            to_process_path = te.avl[i].current_seq
            to_out_path = output_dir/'{:06d}.pkl'.format(int(to_process_path.stem))
            if to_out_path.exists():
                continue
        sample = te.extract(i)
        graph = ge.extract(sample) # TODO add idx key
        sample['graph'] = graph
        # if check_nan(sample):
        #     print(f'there is nan in: {sample["file_id"]}!!!')
        output_path = output_dir/'{:06d}.pkl'.format(int(sample['file_id'].stem))
        with output_path.open('wb') as f:
            pickle.dump(sample, f)
        gc.collect()
    print(f'sub process {pid} finish!!!')
    return pid


def permodeProcessor(output_root_dir, mode, num_processor=16):
    output_dir = output_root_dir / f'{mode}'
    output_json = output_root_dir / f'{mode}.json'
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    start_time = time.time()
    num_p = num_processor
    processes = []
    source_dir = sources_dirs[mode]
    num_seq_file = len(list(Path(source_dir).glob('*.csv')))
    if DEBUG:
        num_seq_file = 1000
    num_per_task = num_seq_file // num_p
    for i in range(num_p):
        start = i * num_per_task
        end = start+num_per_task if i != num_p - 1 else num_seq_file
        t = Process(target=processor, args=(i, source_dir, output_dir, start, end, mode))
        t.daemon = True
        processes.append(t)
        t.start()

    for t in processes:
        t.join()
    # json
    seq_path = []
    for seq_pkl in output_dir.glob('*.pkl'):
        seq_path.append(str(seq_pkl.stem))

    with output_json.open('w') as f:
        json.dump(seq_path, f)
    print(f"finished {mode}")
    print("--- %s seconds ---" % (time.time() - start_time))


def generate_dataset(output_dir, modes=['train', 'val', 'test']):
    output_dir = Path(output_dir)
    for mode in modes:
        permodeProcessor(output_dir, mode)

if __name__ == '__main__':
    generate_dataset(
        output_dir='/home/lgcn/datasets/argo/LaneGCN_pkl',
        modes=['train', 'val', 'test'])
