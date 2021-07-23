import numpy as np
import copy
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
import logging
from pathlib import Path

class TrajExtractor:
    def __init__(self, root_dir, config, mode):
        self.avl = ArgoverseForecastingLoader(root_dir)
        self.avl.seq_list.sort()
        self.config = config
        self.train = mode == 'train'
        logging.info(f'root_dir: {root_dir}, num of squence: {len(self.avl)}')

    def __del__(self):
        del self.avl

    def extract(self, index=None, seq_path=None):
        assert index is not None or Path(seq_path).exists(), 'cannot get sequence: no index is provided and no seq_path file'
        if index is not None:
            seq = self.avl[index]
        else:
            seq = self.avl.get(seq_path)

        data = self.read_argo_data(seq)
        data = self.get_obj_feats(data)
        return data

    def read_argo_data(self, seq):
        city = copy.deepcopy(seq.city)

        """TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,CITY_NAME"""
        df = copy.deepcopy(seq.seq_df)

        agt_ts = np.sort(np.unique(df['TIMESTAMP'].values))
        mapping = dict()
        for i, ts in enumerate(agt_ts):
            mapping[ts] = i

        trajs = np.concatenate((
            df.X.to_numpy().reshape(-1, 1),
            df.Y.to_numpy().reshape(-1, 1)), 1)

        steps = [mapping[x] for x in df['TIMESTAMP'].values]
        steps = np.asarray(steps, np.int64)

        objs = df.groupby(['TRACK_ID', 'OBJECT_TYPE']).groups
        keys = list(objs.keys())
        obj_type = [x[1] for x in keys]

        agt_idx = obj_type.index('AGENT')
        idcs = objs[keys[agt_idx]]

        agt_traj = trajs[idcs]
        agt_step = steps[idcs]

        del keys[agt_idx]
        ctx_trajs, ctx_steps = [], []
        # ctx means neighbor (in my mind)
        for key in keys:
            idcs = objs[key]
            ctx_trajs.append(trajs[idcs])
            ctx_steps.append(steps[idcs])

        data = dict()
        data['file_id'] = self.avl.current_seq
        data['city'] = city
        data['trajs'] = [agt_traj] + ctx_trajs  # [agent, other neighbor] traj
        data['steps'] = [agt_step] + ctx_steps  # [agent, other nrighbor] traj corresponding step index
        return data

    def get_obj_feats(self, data):
        orig = data['trajs'][0][19].copy().astype(np.float32)  # last point of agent.

        # add noise at train phase
        if self.train and self.config['rot_aug']:
            theta = np.random.rand() * np.pi * 2.0
        else:
            pre = data['trajs'][0][18] - orig
            theta = np.pi - np.arctan2(pre[1], pre[0])  # theta is the last 2 frame heading

        rot = np.asarray([  # a sequence has the same rot mat
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]], np.float32)

        feats, ctrs, gt_preds, has_preds, hist_trajs = [], [], [], [], []
        for traj, step in zip(data['trajs'], data['steps']):
            if 19 not in step:
                continue  # if an neighbor is not present at the last frame, omit it.

            gt_pred = np.zeros((30, 2), np.float32)
            has_pred = np.zeros(30, np.bool)
            future_mask = np.logical_and(step >= 20, step < 50)
            post_step = step[future_mask] - 20
            post_traj = traj[future_mask]
            gt_pred[post_step] = post_traj
            has_pred[post_step] = 1

            obs_mask = step < 20
            step = step[obs_mask]
            traj = traj[obs_mask]
            idcs = step.argsort()  # why argsort
            step = step[idcs]
            traj = traj[idcs]

            # valid history step mask: must be continueous till last observed point
            for i in range(len(step)):  # clip gap
                if step[i] == 19 - (len(step) - 1) + i:
                    break
            step = step[i:]
            traj = traj[i:]

            feat = np.zeros((20, 3), np.float32)
            feat[step, :2] = np.matmul(rot, (traj - orig.reshape(-1, 2)).T).T  # route history
            feat[step, 2] = 1.0

            x_min, x_max, y_min, y_max = self.config['pred_range']  # clip extent
            if feat[-1, 0] < x_min or feat[-1, 0] > x_max or feat[-1, 1] < y_min or feat[-1, 1] > y_max:
                continue

            ctrs.append(feat[-1, :2].copy())  # ctrs: abs xy coor (rotated, each agent's base pt)
            feat[1:, :2] -= feat[:-1, :2]
            feat[step[0], :2] = 0
            feats.append(feat)  # offset, rotated; valid flag
            gt_preds.append(gt_pred)  # abs xy coor, (no rotated, no based on base pt)
            has_preds.append(has_pred)  # loss flag
            hist_trajs.append(traj)

        feats = np.asarray(feats, np.float32)
        ctrs = np.asarray(ctrs, np.float32)
        gt_preds = np.asarray(gt_preds, np.float32)
        has_preds = np.asarray(has_preds, np.bool)

        data['feats'] = feats
        data['ctrs'] = ctrs
        data['orig'] = orig
        data['theta'] = theta
        data['rot'] = rot
        data['gt_preds'] = gt_preds
        data['has_preds'] = has_preds
        # add by zhaoyi
        data['hist_traj'] = hist_trajs
        data['fur_traj'] = gt_preds
        return data

'''data dict format
neighbor filter:
1. has point at frame 19
2. within range self.config['pred_range']

-------------------------seq level---------------------
name; type; shape; meaning

orig: ndarray; (2,); agent base_pt, center of the whole sequence
theta: float; 1; agent history last direction angle, (-pi, pi)
rot: ndarray; (2,2); rot matrix by theta
------------------agent(neighbor) level----------------
suppose we have N agents in the sequence
valid history step mask: must be continueous till last observed point

name; type; shape; meaning

feats: ndarray; (N, 20, 3); step offset (:2) | step valid indicator (2), (rotated, bias)
ctrs: ndarray; (N, 2); agent 20th coor, each agent's base pt (rotated, bias)
gt_preds: ndarray; (N, 30, 2); agent gt furture traj, (no roated, no bias)
has_preds: ndarray; (N, 30); bool ndarray, i-th is True means i-th gt_pred coor is valid, False means that is valid 
history_traj: [ndarray]; N * (valid_history_step_num, 2); history trajectory (no roated, no bias, continueous till last observed point)
fur_traj: identical with gt_preds
'''