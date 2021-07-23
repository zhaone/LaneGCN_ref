from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader

_ZORDER = {"AGENT": 15, "AV": 10, "OTHERS": 5}
_DRAW_EXTENT = 50

'''
{
    'neighbor_alphas'
    'path_alphas'
    'pred_offset'
    'pred_delta_dist'
    'pred_delta_sin_heading'
    'path_mask'
}
'''


class ArgoVisHook(object):
    def __init__(self, every_n_steps, output_dir,
                 width=100, height=100, scale=10, mode='train',
                 source_dir='/workspace/datasets/argo/forecasting/val/data/'):
        self._every_n_steps = every_n_steps
        self.last_epoch = -1

        self.source_dir = source_dir
        self.avm = ArgoverseMap()
        self.seq_lane_props = {}
        self.seq_lane_props['PIT'] = self.avm.city_lane_centerlines_dict['PIT']
        self.seq_lane_props['MIA'] = self.avm.city_lane_centerlines_dict['MIA']
        self.afl = ArgoverseForecastingLoader(source_dir)
        self.seq_file = None

        self._width = width
        self._height = height
        self._min_alpha = 0.1

        self._output_dir = Path(output_dir)
        if not self._output_dir.exists():
            self._output_dir.mkdir(parents=True)

        self.mode = mode

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

    def trigger(self, global_step, global_epoch):
        if global_step % self._every_n_steps == 0:
            return True
        return False


    def draw_map(self, file_id, city_name, base_pt):
        # TODO solve the api avm problem
        seq_lane_props = deepcopy(self.seq_lane_props[city_name])
        # seq_lane_props = self.avm.city_lane_centerlines_dict[city_name]

        # draw all nearby lanes
        x_min = base_pt[0] - _DRAW_EXTENT
        x_max = base_pt[0] + _DRAW_EXTENT
        y_min = base_pt[1] - _DRAW_EXTENT
        y_max = base_pt[1] + _DRAW_EXTENT

        lane_centerlines = []
        # Get lane centerlines which lie within the range of trajectories
        for lane_id, lane_props in seq_lane_props.items():
            lane_cl = lane_props.centerline
            if (
                    np.min(lane_cl[:, 0]) < x_max
                    and np.min(lane_cl[:, 1]) < y_max
                    and np.max(lane_cl[:, 0]) > x_min
                    and np.max(lane_cl[:, 1]) > y_min
            ):
                # lane_cl -= base_pt
                lane_centerlines.append(lane_cl)
        for lane_cl in lane_centerlines:
            self.ax.plot(lane_cl[:, 0], lane_cl[:, 1], "-.", color="grey", alpha=0.5, linewidth=3, zorder=1)

    '''
    map; agent history; agent future gt; agent mutiple prediction; other histroy; other future;
    '''
    def run(self, global_step, global_epoch, example, results, unmerge_metrics=None, write=False):

        # get interest agent
        preds = np.concatenate([x[0:1].detach().cpu().numpy() for x in results["reg"]], axis=0)
        # todo adde metric

        for i in range(len(example['file_id'])):
            # set feature size
            self.fig.set_figheight(self._height / 5)
            self.fig.set_figwidth(self._width / 5)
            self.ax.clear()

            file_id = example['file_id'][i]
            city_name = example['city'][i]
            orig = example['orig'][i].numpy()
            hist_trajs = example['hist_traj'][i]
            fur_trajs = example['fur_traj'][i]
            pred_trajs = preds[i]
            # print(hist_trajs[0])
            _, idx = np.unique(pred_trajs, axis=0, return_index=True)
            pred_trajs = pred_trajs[np.sort(idx)][:6]

            self.draw_map(file_id, city_name, orig)

            # compute metrics
            err = np.sqrt(((pred_trajs - np.expand_dims(fur_trajs[0], axis=0)) ** 2).sum(2))
            ades = np.mean(err, axis=1)
            fdes = err[:, -1]
            ade1, fde1, ade, fde = ades[0], fdes[0], np.min(ades), np.min(fdes)

            ## agent
            self.ax.scatter(hist_trajs[0][:, 0], hist_trajs[0][:, 1], marker="o",
                            facecolors='none', edgecolors=(0, 0, 0), s=60, zorder=3)
            if len(fur_trajs) != 0:
                self.ax.scatter(fur_trajs[0][:, 0], fur_trajs[0][:, 1], marker="o",
                                facecolors='none', edgecolors=(0, 0, 1), s=60, zorder=3)
            for p_idx, pred_traj in enumerate(pred_trajs):
                alpha = 1
                self.ax.scatter(pred_traj[:, 0], pred_traj[:, 1], marker="o",
                                facecolors='none', edgecolors=(0, 1, 0), s=60 * alpha, zorder=4)
                if p_idx >=5:
                    break

            self.ax.axis('off')
            fde_str = str(np.round(fde, 3))[:6]
            self.fig.savefig(
                str(self._output_dir / f"{fde_str}_{Path(file_id).stem}.svg"), format='svg', bbox_inches='tight', dpi=200)
