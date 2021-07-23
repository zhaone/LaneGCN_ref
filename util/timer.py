import functools
import time
from collections import Counter


def func_timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer


class Timer(object):

    def __init__(self):
        self._total_time = Counter()
        self._total_count = Counter()
        self._start_time = {}
        self._tick_time = {}

    def contains(self, name):
        return name in self._total_time

    def get_avg(self, name):
        if name not in self._total_time:
            raise ValueError("{} not found".format(name))

        if self._total_count[name] == 0:
            return "{} Avg:0.000s".format(name)

        return "{} Avg:{:.3f}s".format(
            name, self._total_time[name] / self._total_count[name])

    def __str__(self):
        return "\n".join([
            self.get_avg(name) for name in self._total_count
        ])

    def create(self, *args):
        start_time = time.time()
        for arg in args:
            self._start_time[arg] = start_time
            self._tick_time[arg] = start_time
            self._total_time[arg] = 0
            self._total_count[arg] = 0

    def tick(self, *args):
        cur_time = time.time()
        for arg in args:
            if arg not in self._start_time:
                raise ValueError("{} not found ".format(arg))
            self._tick_time[arg] = cur_time

    def update_from_tick(self, *args):
        cur_time = time.time()
        for arg in args:
            if arg not in self._start_time:
                raise ValueError("{} not found ".format(arg))
            delta = cur_time - self._tick_time[arg]
            self._total_count[arg] += 1
            self._total_time[arg] += delta

    def update_from_start(self, *args):
        cur_time = time.time()
        for arg in args:
            if arg not in self._start_time:
                raise ValueError("{} not found ".format(arg))
            delta = cur_time - self._start_time[arg]
            self._total_count[arg] += 1
            self._total_time[arg] = delta
