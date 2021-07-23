import json
from util.util import sum_dict
from util.util import mean_dict
from copy import deepcopy
import torch
from mpi4py import MPI

comm = MPI.COMM_WORLD

class Metrics:

    def __init__(self, compare_fn):
        self._best_step = -1
        self._latest_step = -1

        self._best_epoch = -1
        self._latest_epoch = -1

        self._best_metrics = None
        self._latest_metrics = None

        self._is_updated = False
        self._best_is_updated = False
        self._metrics_dict = None
        self._acc_metrics_dict = None
        self._acc_count = 0
        self._compare_fn = compare_fn

    def reset_acc_metrics(self):
        self._acc_metrics_dict = None
        self._acc_count = 0

    def update_acc_metrics(self, metric_dict, this_batch_size):
        self._acc_metrics_dict = sum_dict(metric_dict, self._acc_metrics_dict)
        self._acc_count += this_batch_size

    def avg_acc_metrics(self):
        return mean_dict(deepcopy(self._acc_metrics_dict), self._acc_count)

    def restore_from_meta(self, meta_info):
        self._best_step = meta_info['best_model']['step']
        self._latest_step = meta_info['latest_eval']['step']

        self._best_epoch = meta_info['best_model']['epoch']
        self._latest_epoch = meta_info['latest_eval']['epoch']

        self._best_metrics = meta_info['best_model']['metrics']
        self._latest_metrics = meta_info['latest_eval']['metrics']

    def update(self, step, epoch, new_metrics):
        self._is_updated = True
        is_better = self._compare_fn(self._best_metrics, new_metrics)
        if is_better:
            self._best_metrics = new_metrics
            self._best_step = step
            self._best_epoch = epoch
            self._best_is_updated = True
        self._latest_metrics = new_metrics
        self._latest_step = step
        self._latest_epoch = epoch

    def stuck_step(self):
        if self._latest_epoch == -1 or self._best_epoch == -1:
            return 0
        return self._latest_epoch - self._best_epoch

    def is_updated(self):
        flag = self._is_updated
        self._is_updated = False
        return flag

    def best_is_updated(self):
        flag = self._best_is_updated
        self._best_is_updated = False
        return flag

    def latest(self):
        result = {
            'step': self._latest_step,
            'epoch': self._latest_epoch,
            'metrics': self._latest_metrics
        }
        return result

    def best(self):
        result = {
            'step': self._best_step,
            'epoch': self._best_epoch,
            'metrics': self._best_metrics
        }
        return result

    def __str__(self):
        repr_s = json.dumps(
            {
                "latest": {
                    "step": self._latest_step,
                    "metrics": self._latest_metrics
                },
                "best": {
                    "step": self._best_step,
                    "metrics": self._best_metrics
                },
            }, indent=4)
        return repr_s



def reduce_metrics(data, world_size):
    '''
    :param data: value must be scalar
    :return: reuced metrics
    '''
    if world_size < 2:
        return data
    data_list = comm.allgather(data)
    reduced_metrics = {}
    for key in data_list[0]:
        reduced_metrics[key] = 0
        for i in range(len(data_list)):
            reduced_metrics[key] += data_list[i][key]
        reduced_metrics[key] /= world_size
    return reduced_metrics