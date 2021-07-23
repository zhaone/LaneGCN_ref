import mxnet.recordio as mxrec
import pickle
from pathlib import Path
from tqdm import tqdm
import json

source_root = Path('/workspace/datasets/argo/LaneGCN_pkl')
output_root = Path('/workspace/datasets/argo/LaneGCN')

if not output_root.exists():
    output_root.mkdir(parents=True)

splits = ['train', 'val', 'test']
max_sample = 1000

def dump_mxrec(data_splits):
    for dsp in data_splits:
        num_sample = 0
        source_path = source_root / dsp
        output_meta = output_root / f'{dsp}.meta'
        write_record = mxrec.MXIndexedRecordIO(str(output_root / f'{dsp}.idx'), str(output_root / f'{dsp}.rec'), 'w')
        for pkl_path in tqdm(source_path.glob("*.pkl")):
            with pkl_path.open('rb') as pf:
                data = pickle.load(pf)
                data = pickle.dumps(data)
                write_record.write_idx(num_sample, data)
            num_sample+=1
            # if num_sample > max_sample:
            #     break
        with output_meta.open('w') as f:
            json.dump({'num_sample': num_sample}, f)
        write_record.close()


if __name__ == '__main__':
    dump_mxrec(splits)
