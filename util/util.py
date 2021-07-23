import numpy as np
import copy
import torch
import time

def ref_copy(data):
    if isinstance(data, list):
        return [ref_copy(x) for x in data]
    if isinstance(data, dict):
        d = dict()
        for key in data:
            d[key] = ref_copy(data[key])
        return d
    return data

def sum_dict(dict1, dict2):
    if dict1 is None:
        return dict2
    elif dict2 is None:
        return dict1
    ret_dict = {}
    keys = set(list(dict1.keys()))
    keys.update(list(dict2.keys()))
    for key in keys:
        if key in dict1 and key in dict2:
            if (isinstance(dict1[key], (int, float)) or np.isscalar(dict1[key])):
                ret_dict[key] = dict1[key] + dict2[key]
            elif isinstance(dict1[key], dict):
                ret_dict[key] = sum_dict(dict1[key], dict2[key])
            else:
                raise ValueError("dict1 {}, dict2 {}".format(type(dict1[key]), type(dict2[key])))
        elif key in dict1:
            ret_dict[key] = dict1[key]
        else:
            ret_dict[key] = dict2[key]
    return ret_dict


def mean_dict(sdict, count):
    for key in sdict:
        if (isinstance(sdict[key], (int, float)) or np.isscalar(sdict[key])):
            sdict[key] = sdict[key] / count
        elif isinstance(sdict[key], dict):
            sdict[key] = mean_dict(sdict[key], count)
        else:
            raise ValueError
    return sdict

def get_run_time(name, start_time):
    torch.cuda.synchronize()
    t = time.time()
    print(f'{name} last time:', t - start_time)
    return t