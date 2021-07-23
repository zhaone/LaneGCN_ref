import os
import logging
from datetime import datetime
import random
from pathlib import Path
import tarfile
import argparse
import shutil
import json
import hashlib
import GPUtil
import numpy as np
import torch
import horovod.torch as hvd
from easydict import EasyDict as edict
from main import main as m

SEED = 1
os.environ['PYTHONHASHSEED'] = str(SEED)

# Argument Parser
#-----------------key----------------
key_parser = argparse.ArgumentParser(
    description='Trajectory Prediction, Important param', add_help=False)
key_parser.add_argument(
    '--model',
    type=str,
    required=False,
)
key_parser.add_argument('--optimizer', type=str, default='adam')
key_parser.add_argument('--hparams_path', required=False, type=str)
key_parser.add_argument('--seed', type=int, default=SEED)

all_parser = argparse.ArgumentParser(parents=[key_parser])

#-----------------basic-------------
all_parser.add_argument('--run_id',
                        type=str,
                        required=False,
                        default=None,
                        help="run id if only export or eval")
all_parser.add_argument('--name',
                        type=str,
                        default='traj',
                        help='A name to attach to the training session')
all_parser.add_argument('--force_clear', action='store_true', default=False)
all_parser.add_argument('--mixed_precision',
                        action='store_true',
                        default=False)
#----------------distribution-------------
all_parser.add_argument(
    "--is_dist", action="store_true", default=False
)
all_parser.add_argument(
    "--local_rank", type=int
)
all_parser.add_argument(
    "--world_size", type=int, default=1
)
all_parser.add_argument(
    '--gpu_id', type=str, required=False, default=0, help='IDs of GPUs to use'
)
all_parser.add_argument(
    "--main_process", action="store_true", default=True
)

#-----------------mode--------------------
all_parser.add_argument('--mode',
                        type=str,
                        choices=["train_eval", "train", "eval", "test"],
                        default="train_eval")
all_parser.add_argument('--enable_initial_evaluate',
                        action='store_true',
                        default=False)
all_parser.add_argument('--debug',
                        action='store_true',
                        default=False)

#-----------------optimize-----------------------
# optimization
all_parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
all_parser.add_argument('--lr_decay', type=float, default=1, help='Learning rate decay')
all_parser.add_argument('--min_decay', type=float, default=0.001, help='decay will stop if lr has reached min_decay * lr minimum value')
all_parser.add_argument('--momentum',
                        type=float,
                        default=0,
                        required=False)
# stop
all_parser.add_argument('--epochs',
                        type=int,
                        default=20,
                        help='Number of epochs')
all_parser.add_argument('--enable_early_stop',
                        action='store_true',
                        default=False)
all_parser.add_argument('--early_stop_patience',
                        type=int,
                        default=20,
                        required=False)
all_parser.add_argument('--early_stop_min_delta',
                        type=int,
                        default=0.5,
                        required=False)
#----------------dataset----------------------
all_parser.add_argument('--data_name',
                        type=str,
                        required=False,
                        help="dataset name, like: argo, lyft")
all_parser.add_argument('--data_version',
                        type=str,
                        required=False,
                        help="dataset version, like: debug, full")
all_parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
all_parser.add_argument('--shuffle', action='store_true', default=False)
all_parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='num parallel for dataset')
all_parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False)
#---------------log---------------------------
all_parser.add_argument('--reload',
                        type=str,
                        choices=["best", "latest", "none"],
                        default="latest")
all_parser.add_argument('--save_path',
                        type=str,
                        default='/tmp/intention_outputs')
all_parser.add_argument('--print_steps', type=int, default=1000)
all_parser.add_argument('--board_steps', type=int, default=1000)
all_parser.add_argument('--save_checkpoints_steps', type=int, default=1000)
all_parser.add_argument('--eval_steps', type=int, default=1000)
all_parser.add_argument('--keep_checkpoint_max', type=int, default=6)
all_parser.add_argument('--save_summary_steps', type=int, default=100)

#---------------hook--------------------------
all_parser.add_argument('--enable_hook', action='store_true', default=False)

def dump_src(save_dir):
    with tarfile.open(str(Path(save_dir) / 'env' / 'src.tar.gz'),
                      'w:gz') as tar:
        tar.add('model')
    pass


def dump_env(pydict, save_dir, name):
    json.dump(pydict, (Path(save_dir) / 'env' / name).open(mode='w'),
              indent=4,
              sort_keys=True)


def init_logger(logger_name, log_dir, filename, fmt=None, datefmt=None, disable_stdout=False):
    # Set color for warning and error(only in terminal)
    logging.addLevelName(logging.WARNING, "\033[1;31mW\033[1;0m")
    logging.addLevelName(logging.ERROR, "\033[1;41mE\033[1;0m")
    logging.addLevelName(logging.INFO, "I")
    log_file_path = log_dir / filename

    FORMAT = "[%(levelname)s %(asctime)s] %(message)s" if fmt is None else fmt
    # DATEFMT = '%Y-%m-%d %H:%M:%S'
    DATEFMT = '%m-%d %H:%M:%S' if datefmt is None else datefmt

    logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATEFMT)

    logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATEFMT)
    logger = logging.getLogger(logger_name)

    file_handler = logging.FileHandler(str(log_file_path),
                                       mode='a',
                                       encoding=None,
                                       delay=False)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(FORMAT, DATEFMT))

    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    return logger


def load_saved_env(args, saved_path):
    args_json_path = saved_path / "env" / "args.json"
    args_json = json.load(args_json_path.open(mode="r"))

    hparams_json_path = saved_path / "env" / "hparams.json"
    hparams_json = json.load(hparams_json_path.open(mode="r"))

    # dataset_cfg_json_path = saved_path / "env" / "dataset_cfg.json"
    # dataset_cfg_json = json.load(dataset_cfg_json_path.open(mode="r"))
    args_json["reload"] = args.reload
    args_json["mode"] = args.mode
    args_json["gpu_id"] = args.gpu_id
    args_json["enable_hook"] = args.enable_hook
    return edict(args_json), hparams_json


def prepare_env(args):

    key_args = key_parser.parse_known_args()[0]
    if args.run_id is not None:
        run_id = args.run_id
        name, identity = run_id.split("-")
        save_dir = Path(args.save_path) / run_id
        assert save_dir.exists()
    else:
        name = args.name
        md5 = hashlib.md5(
            json.dumps(vars(key_args), sort_keys=True).encode('utf-8')).hexdigest()
        identity = str(md5)

        # run_id = f"{name}-{identity}"
        run_id = f"{name}"

        save_dir = Path(args.save_path) / run_id
        if save_dir.exists() and args.force_clear:
            shutil.rmtree(save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / 'env').mkdir(parents=True, exist_ok=True)
    (save_dir / 'log').mkdir(parents=True, exist_ok=True)
    (save_dir / 'params').mkdir(parents=True, exist_ok=True)
    (save_dir / 'eval').mkdir(parents=True, exist_ok=True)

    (save_dir /
     "{}".format(datetime.now().strftime(format="%Y-%m-%d_%H-%M"))).touch()

    fmt = f"[{name}-{identity[:5]} %(levelname)s%(asctime)s] %(message)s"
    # DATEFMT = '%Y-%m-%d %H:%M:%S'
    datefmt = '%m-%d %H:%M'

    init_logger(None,
                save_dir / 'log',
                "log.txt",
                fmt=fmt,
                datefmt=datefmt)
    init_logger('important',
                save_dir / 'log',
                "key.txt",
                disable_stdout=True)
    logging.info(datetime.now().strftime('%Y-%m-%d-%H_%M_%S'))
    logging.info(f"Output Dir {save_dir}")

    return run_id, save_dir


def main():
    key_args = key_parser.parse_known_args()[0]
    print(key_args)
    args = all_parser.parse_args()
    args.name = args.name.replace("-", "_")

    if args.mode == "train_eval":
        assert args.data_name is not None
        assert args.hparams_path is not None

    run_id, save_dir = prepare_env(args)
    # set distribution env
    if torch.cuda.is_available():
        if args.is_dist:
            hvd.init()
            args.local_rank = hvd.local_rank()
            args.world_size = hvd.size()
            print('local rank:', args.local_rank, 'world size', args.world_size)
            if args.local_rank != 0:
                args.main_process = False
            torch.cuda.set_device(args.local_rank)
            seed = args.local_rank
        else:
            torch.cuda.set_device('cuda:%d' % int(args.gpu_id))
            seed = args.seed
            # seed = args.gpu_id
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)  # numpy pseudo-random generator
        random.seed(seed) # `python` built-in pseudo-random generator
    else:
        raise RuntimeError('gpu is not available')

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    if args.run_id is None:
        # hparams = edict(json.loads(open(args.hparams_path, "r").read()))
        hparams = json.loads(open(args.hparams_path, "r").read())
        if args.main_process:
            dump_src(save_dir)
            dump_env(vars(args), save_dir, "args.json")
            dump_env(hparams, save_dir, "hparams.json")
    else:
        args, hparams = load_saved_env(args, save_dir) # TODO some process may not have $savedir
    m(args, str(save_dir), hparams=hparams)


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    main()