import numpy as np
import pickle
from pathlib import Path
from argoverse.evaluation.competition_util import generate_forecasting_h5
import torch.distributed as dist

class ArgoResultCollectionHook(object):
    def __init__(self, every_n_steps, output_dir, mode='test', args=None):
        self._every_n_steps = every_n_steps
        self._output_dir = output_dir
        self.args = args
        if self.args.main_process:
            if not self._output_dir.exists():
                self._output_dir.mkdir(parents=True)
        self._traj_all = {}
        self._cls_all = {}

    def trigger(self, global_step, global_epoch):
        if global_step % self._every_n_steps == 0:
            return True
        return False

    def run(self, global_step, global_epoch, example, results, write=False):
        """
        Helper function to generate the result h5 file for argoverse forecasting challenge

        Args:
            data: a dictionary of trajectory, with the key being the sequence ID. For each sequence, the
                  trajectory should be stored in a (9,30,2) np.ndarray
            output_path: path to the output directory to store the output h5 file
            filename: to be used as the name of the file
            probabilities (optional) : normalized probability for each trajectory

        Returns:

        """
        batch_features = example
        preds = np.concatenate([x[0:1].detach().cpu().numpy() for x in results["reg"]], axis=0)
        cls = np.concatenate([x[0:1].detach().cpu().numpy() for x in results["cls"]], axis=0)

        for i, file_id in enumerate(batch_features['file_id']):
            seq_id = int(file_id.stem)
            pred_trajs = preds[i]
            agent_cls = cls[i]
            _, idx = np.unique(pred_trajs, axis=0, return_index=True)
            pred_trajs = pred_trajs[np.sort(idx)][:6]
            agent_cls = agent_cls[np.sort(idx)][:6]
            self._traj_all[seq_id] = pred_trajs
            self._cls_all[seq_id] = agent_cls

        if write: # each process write a pickle
            output_pkl_path = Path(self._output_dir) / f'res_{self.args.local_rank}.pkl'
            with output_pkl_path.open('wb') as f:
                pickle.dump({
                    'traj': self._traj_all,
                    'cls': self._cls_all
                }, f)
            dist.barrier() # make sure that ever process has finished file writing
            if self.args.main_process:
                trajs = {}
                cls = {}
                for i in range(self.args.world_size):
                    source_pkl_path = Path(self._output_dir) / f'res_{i}.pkl'
                    with source_pkl_path.open('rb') as f2:
                        res = pickle.load(f2)
                        trajs.update(res['traj'])
                        cls.update(res['cls'])
                generate_forecasting_h5(trajs, self._output_dir, filename='res_mgpu', probabilities=cls)
                print(f'------dump finish, num of samples:{len(cls)}--------')
