import logging
import torch
import json
from experiment import Experiment
from dataset import get_dataloader
from model import get_model

def dump_moudle(local_rank, module, dump_path, other):
    from pathlib import Path
    dump_path = Path(dump_path) / f'gpu_{local_rank}-{other}.pkl'
    torch.save(module.state_dict(), dump_path)

def main(args, output_dir, hparams=None):
    """
    :param args: argument
    :param output_dir: save dir
    :return:
    """

    torch.autograd.set_detect_anomaly(True)

    if args.main_process:
        logging.info("Args {}".format(json.dumps(
            vars(args), indent=2, sort_keys=True)))
        logging.info("Hparams {}".format(json.dumps(
            hparams, indent=2, sort_keys=True)))

    model = get_model(args, hparams)

    train_dataloader = None
    validation_dataloader = None
    test_dataloader = None

    if 'train' in args.mode:
        train_dataloader = get_dataloader(
            args, mode="train")
    if 'val' in args.mode:
        validation_dataloader = get_dataloader(
            args, mode="val")
    if 'test' in args.mode:
        test_dataloader = get_dataloader(
            args, mode="test")

    if args.main_process: logging.info("Creating Experiment Instance...")
    ex = Experiment(model=model,
                    train_dataloader=train_dataloader,
                    validation_dataloader=validation_dataloader,
                    test_dataloader=test_dataloader,
                    output_dir=output_dir,
                    device=None,
                    args=args)

    try:
        if args.mode == 'train_eval':
            if args.main_process: logging.info("Start training...")
            ex.train_eval()
        elif args.mode == 'eval':
            if args.main_process: logging.info("Start evalating...")
            ex.evaluate()
        elif args.mode == 'test':
            if args.main_process: logging.info("Start inferring...")
            ex.infer()
        else:
            raise NotImplementedError("Not implemented")
    except RuntimeError as e:
        raise e
    except IOError as e:
        raise e
    except ValueError as e:
        raise e
    except KeyboardInterrupt:
        if args.main_process: logging.info("Exit by keyboard interrupt ")
    logging.info(f"Done {output_dir}")