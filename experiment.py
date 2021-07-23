import pdb
import pathlib
import logging
import json
from time import time
import pickle
import sys
import datetime
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch import autograd
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR
import horovod.torch as hvd
from tensorboardX import SummaryWriter
from util.metrics import Metrics
# from lib.metrics.metrics import Metrics
from util.timer import Timer
from util.dc import to_gpu
from util.metrics import reduce_metrics
from torch.cuda import amp


def get_optimizer(args, parameters):
    name = args.optimizer.lower()
    learning_rate = args.lr
    optimizer = None
    if name == "sgd":
        optimizer = optim.SGD(parameters,
                              lr=learning_rate,
                              momentum=args.momentum)
    elif name == "adam":
        optimizer = optim.Adam(parameters, lr=learning_rate)
    elif name == "rmsprop":
        optimizer = optim.RMSprop(parameters, lr=learning_rate)
    else:
        raise NotImplementedError
    logging.info("Using optimizer -> {}:{}".format(name, learning_rate))
    logging.info(f"{optimizer}")
    return optimizer


def get_scheduler(args, optimizer):
    if args.lr_decay is None:
        scheduler = None
    else:
        # lambda1 = lambda epoch: args.lr_decay ** epoch if args.lr_decay ** epoch > args.min_decay else args.min_decay
        # scheduler = LambdaLR(optimizer, lr_lambda=lambda1)
        # scheduler = StepLR(optimizer, step_size=32, gamma=.1)
        scheduler = MultiStepLR(optimizer, milestones=[32], gamma=.1)

    return scheduler


class Experiment:
    def __init__(self, model, train_dataloader, validation_dataloader, test_dataloader,
                 output_dir, device, args):

        self._args = args
        self.main_process = self._args.main_process
        # output dir
        output_dir = pathlib.Path(output_dir)
        self._checkpoint_dir = output_dir / 'params'
        self._eval_dir = output_dir / 'eval'
        if self.main_process:
            self._writer = SummaryWriter(output_dir / 'log')

        self._criterion = model.get_criterion()
        self._get_metrics = model.get_metrics_func()

        self._train_hooks = model.get_train_hooks(args, output_dir)
        self._eval_hooks = model.get_eval_hooks(args, output_dir)
        self._test_hooks = model.get_test_hooks(args, output_dir)

        self._batch_size = args.batch_size
        self._val_batch_size = args.batch_size
        self._test_batch_size = args.batch_size
        self._global_step = 0
        self._global_epoch = 0
        self._current_loss = 0
        self._train_loss_history = []
        self._val_loss_history = []
        self._last_eval_time = None
        self._device = device
        self._timer = Timer()

        self._n_train_batches = len(train_dataloader) if train_dataloader is not None else 0
        self._n_val_batches = len(validation_dataloader) if validation_dataloader is not None else 0
        self._n_test_batches = len(test_dataloader) if test_dataloader is not None else 0

        self._train_dataloader = train_dataloader
        self._validation_dataloader = validation_dataloader
        self._test_dataloader = test_dataloader

        self._metrics = Metrics(compare_fn=model.metrics_compare_fn)

        self.scaler = amp.GradScaler()

        if torch.cuda.is_available():
            logging.info("Using GPU")
            self._model = model.cuda()
        else:
            print('-------------------not distribution-------------')
            self._model = model

        self._optimizer = get_optimizer(args, model.parameters())
        if self._args.is_dist:
            self._optimizer = hvd.DistributedOptimizer(self._optimizer, named_parameters=self._model.named_parameters())
        self._scheduler = get_scheduler(args, self._optimizer)

        self.maybe_load()
        hvd.broadcast_parameters(self._model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self._optimizer, root_rank=0)

        for state in self._optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        self._n_epochs = args.epochs
        # plot windows in visdom
        self._stuck_train_loss_epoch = 0
        self._stuck_val_loss_epoch = 0
        if args.main_process:
            self.summary()

    # TODO change hook code
    def run_train_hooks(self, ex, results):
        for hook in self._train_hooks:
            if hook.trigger(self._global_step, self._global_epoch):
                hook.run(self._global_step, self._global_epoch, ex, results)

    def run_eval_hooks(self, eid, ex, results, unmerge_metrics):
        for hook in self._eval_hooks:
            if hook.trigger(eid, self._global_epoch):
                hook.run(eid, self._global_epoch, ex, results, unmerge_metrics)

    def run_test_hooks(self, eid, ex, results, write=False):
        for hook in self._test_hooks:
            if hook.trigger(eid, self._global_epoch):
                hook.run(eid, self._global_epoch, ex, results, write=write)

    def summary(self):
        logging.info(
            "----------------------------------------------------------------")
        serialized_size = 0
        for name, param in self._model.named_parameters():
            byte_size = param.numel() * param.element_size()
            serialized_size += byte_size
            logging.info("Var: {} {}, type: {}, {}".format(
                name, param.size(), param.dtype, byte_size))
        logging.info("----")
        logging.info(f"Serialized model size: "
                     f"{serialized_size / (1024 * 1024):.3f} MB")
        logging.info("Total number of train samples: {} * {}".format(
            self._n_train_batches, self._batch_size))
        logging.info("Total number of val samples: {} * {}".format(
            self._n_val_batches, self._val_batch_size))
        logging.info("Total number of test samples: {} * {}".format(
            self._n_test_batches, self._test_batch_size))
        logging.info("Optimizer {}:{}".format(self._args.optimizer,
                                              self._args.lr))
        logging.info("LR decay {}".format(self._args.lr_decay))
        logging.info("Reload {}:{}".format(
            self._args.reload, self._metrics))
        logging.info(
            "----------------------------------------------------------------")

    def maybe_load(self):
        meta_info_file = self._checkpoint_dir / "meta.json"
        if meta_info_file.exists() and self._args.reload != "none":
            self.load_state_dict()
        if self._global_step > 0 and self._n_train_batches > 0:
            # make step align with dataset loader
            self._global_step -= self._global_step % self._n_train_batches

    def get_info(self):
        return {'epoch': self._global_epoch, 'step': self._global_step}

    def train_step(self, batch_features, batch_labels):
        # torch.cuda.synchronize()
        # st = time.time()

        self._timer.tick("forward")

        self._model.train()
        self._optimizer.zero_grad()

        with amp.autocast(enabled=self._args.mixed_precision):
            network_results = self._model(batch_features)
            loss_dict = self._criterion(batch_features,
                               batch_labels,
                               network_results)

        backward_loss = loss_dict['total_loss']
        # ns = get_run_time('loss', ns)
        self._timer.update_from_tick("forward")
        self._timer.tick("opt")
        # ns = get_run_time('before backward', ns)
        if self._args.mixed_precision:
            self.scaler.scale(backward_loss).backward()
            self._optimizer.synchronize()
            # self._optimizer.step()
            with self._optimizer.skip_synchronize():
                self.scaler.step(self._optimizer)
                self.scaler.update()
        else:
            backward_loss.backward()
            self._optimizer.step()

        # loss_val = loss.item()
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                loss_dict[k] = v.item()
        self._timer.update_from_tick("opt")
        self._current_loss = loss_dict
        self._global_step += 1

        return network_results, loss_dict

    def should_stop(self):
        if (self._stuck_train_loss_epoch > self._args.early_stop_patience
                and self._stuck_val_loss_epoch > self._args.early_stop_patience
                and
                self._metrics.stuck_step() > self._args.early_stop_patience):
            logging.info("Stop. I have no more patience. {} {} {}".format(
                self._stuck_train_loss_epoch, self._stuck_val_loss_epoch,
                self._metrics.stuck_step()))
            return True
        else:
            return False

    def train_epoch(self):
        losses = []
        self._timer.create("train_step", "forward",
                           "opt", "dataloader", "metrics", "others")
        self._timer.tick("dataloader", "train_step")
        for bid, ex in enumerate(self._train_dataloader):
            data = to_gpu(ex)
            self._timer.update_from_tick("dataloader")

            results, loss_dict = self.train_step(data, data)
            self._timer.update_from_tick("train_step")

            # print(f'-------------{bid}-{self._args.local_rank}-----------\n{loss_dict}')
            # if bid > 100:
            #     sys.exit(0)
            new_metric_dict = self._get_metrics(data, data, results, loss_dict)
            new_metric_dict = reduce_metrics(new_metric_dict, self._args.world_size)
            # print(f'------------metrics-{bid}-{self._args.local_rank}-----------\n{new_metric_dict}')

            self._timer.tick("others")
            if self._args.enable_hook:
                self.run_train_hooks(ex, results)

            if loss_dict is not None:
                losses.append(loss_dict['total_loss'])

            self._timer.tick("metrics")
            self._timer.update_from_tick("metrics")

            self._timer.update_from_tick("others")
            if self.main_process:
                print(f'-------------{bid}-------------')
                if self._global_step % self._args.print_steps == 0:
                    train_timer = self._timer.get_avg("train_step")
                    forward_timer = self._timer.get_avg("forward")
                    opt_timer = self._timer.get_avg("opt")
                    data_timer = self._timer.get_avg("dataloader")
                    others_timer = self._timer.get_avg("others")
                    metrics_timer = self._timer.get_avg("metrics")

                    epoch_percent = (float(self._global_step %
                                           self._n_train_batches) /
                                     self._n_train_batches * 100)

                    logging.info(
                        f"{self._args.name} Epoch {self._global_epoch}"
                        f"/{self._args.epochs} "
                        f"Step:{self._global_step}/"
                        f"{epoch_percent:.2f}% "
                        f"loss: {loss_dict} \n"
                        f"{train_timer} {forward_timer} {opt_timer} "
                        f"{data_timer} {others_timer} {metrics_timer}")

                    for k, v in new_metric_dict.items():
                        logging.info(f'{k}: {v}')


                if self._global_step % self._args.board_steps == 0:
                    self.write_metrics(new_metric_dict, prefix="train")
                    epoch_lr = self._scheduler.get_last_lr()
                    self._writer.add_scalar(
                        'lr/lr', epoch_lr, self._global_epoch)

                if self._global_step % self._args.save_checkpoints_steps == 0:
                    self.save("latest")

            if self._global_step % self._args.eval_steps == 0:
                self.maybe_evaluate()

            self._timer.tick("dataloader", "train_step")

        # learning rate decay if assigned
        if self._scheduler is not None:
            self._scheduler.step()
        loss_avg = np.mean(losses)
        if len(self._train_loss_history) > 0:
            not_update = (self._args.early_stop_min_delta >=
                          self._train_loss_history[-1])
            self._stuck_train_loss_epoch += int(
                loss_avg + not_update)
        self._train_loss_history.append(loss_avg)
        logging.info("Epoch {}: loss.avg: {} {}".format(self._global_epoch,
                                                        loss_avg, len(losses)))
        self._global_epoch += 1


    def train_eval(self):
        n_epochs = self._n_epochs

        # initial evaluation
        if self._args.enable_initial_evaluate:
            self.evaluate()

        # training
        self._model.train()
        if n_epochs is not None and n_epochs > 0:
            logging.info(
                "Start training with total n_epochs = {}".format(n_epochs))
            start_epoch = self._global_epoch
            for e in range(start_epoch, n_epochs):
                if self._args.enable_early_stop and self.should_stop():
                    logging.info("Early stop at {}".format(self._global_epoch))
                    break
                if self._args.is_dist:
                    self._train_dataloader.sampler.set_epoch(int(e))
                self.train_epoch()
        else:
            logging.error("epochs  required.")
        if self.main_process:
            self._writer.close()

            logging.info(
                f"Train ends(Step {self._global_step}, "
                f"Epoch {self._global_epoch}/ {self._args.epochs}). "
                f"Best metrics: {json.dumps(self._metrics.best(), indent=2)}")

    def maybe_evaluate(self):
        self.evaluate()

    def evaluate(self):
        if self.main_process:
            logging.info("Start evalation at step {}...".format(self._global_step))
        self._model.eval()
        self._timer.create("inference_step")
        if self._args.is_dist:
            self._validation_dataloader.sampler.set_epoch(int(self._global_epoch))
        pbar = tqdm(enumerate(self._validation_dataloader),
                    total=self._n_val_batches)
        losses = []
        self._metrics.reset_acc_metrics()
        with torch.no_grad():
            # torch.cuda.synchronize()
            for eid, ex in pbar:
                # pdb.set_trace()
                data = to_gpu(ex)
                torch.cuda.synchronize()
                # Some evaluate need labels
                results = self._model(data)
                torch.cuda.synchronize()
                loss_dict = self._criterion(data,
                                           data,
                                           results,
                                           mode='eval')
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        loss_dict[k] = v.item()
                losses.append(loss_dict['total_loss'])

                # already averaged on one gpu batch size
                new_metric_dict = self._get_metrics(data, data, results, loss_dict, mode='eval')
                new_metric_dict = reduce_metrics(new_metric_dict, self._args.world_size)
                self._metrics.update_acc_metrics(new_metric_dict, 1) # TODO update count

                if self._args.enable_hook:
                    # self.run_eval_hooks(eid, ex, results, unmerge_metrics)
                    self.run_eval_hooks(eid, ex, results, eid == self._n_val_batches - 1)

                pbar.update(1)
                self._timer.update_from_start("inference_step")
            # torch.cuda.synchronize()
            loss_avg = np.mean(losses)
            # self._writer.add_scalar('eval/loss', loss_avg, self._global_step)
            if len(self._val_loss_history) > 0:
                self._stuck_val_loss_epoch += int(
                    loss_avg +
                    self._args.early_stop_min_delta >= self._val_loss_history[-1])
            self._val_loss_history.append(loss_avg)
        if self.main_process:
            last_metrics = self._metrics.latest()
            old_best_metrics = self._metrics.best()

            self._writer.add_scalar('eval/epoch', self._global_epoch, self._global_step)
            self.write_metrics(self._metrics.avg_acc_metrics(), prefix="eval")

            self._metrics.update(self._global_step, self._global_epoch,
                                 self._metrics.avg_acc_metrics())

            logging.info("Inference performance {}s ({} * {})".format(
                self._timer.get_avg("inference_step"), self._val_batch_size,
                self._n_val_batches))
            logging.info(f"Step {self._global_step}: "
                         f"new_metrics {self._metrics.latest()}, "
                         f"last_metrics {last_metrics}")

            eval_json = {
                "step": self._global_step,
                "metrics": self._metrics.latest()
            }
            eval_file_name = "{}".format(self._global_step).zfill(9)
            eval_output_file = (
                self._eval_dir /
                "{}-{}-{}-{}.json".format(self._args.model, self._args.mode,
                                          self._args.name, eval_file_name))
            with eval_output_file.open('w') as f:
                f.write(json.dumps(eval_json, indent=4))
            # eval_output_file.open('w').write(json.dumps(eval_json, indent=4))

            if self._metrics.best_is_updated():
                logging.info("\n!!!!Get Best metrics!!!!\n")
                logging.info("NEW:{}".format(self._metrics.best()))
                logging.info("OLD:{}".format(old_best_metrics))

                best_output_file = (self._eval_dir / "BEST-{}-{}-{}.json".format(
                    self._args.mode, self._args.model, self._args.name))
                with best_output_file.open('w') as f:
                    f.write(json.dumps(eval_json, indent=4))
                # best_output_file.open('w').write(json.dumps(eval_json, indent=4))
                self.save(prefix="best")

            self._model.train()

    def infer(self):
        """
        only work for 1 GPU
        :return:
        """
        logging.info("Start test at step {}...".format(self._global_step))
        self._model.eval()
        self._timer.create("test_step")
        pbar = tqdm(enumerate(self._test_dataloader),
                    total=self._n_test_batches)
        with torch.no_grad():
            for eid, ex in pbar:
                # pdb.set_trace()
                torch.cuda.synchronize()
                data = to_gpu(ex)
                # Some evaluate need labels
                results = self._model(data)
                torch.cuda.synchronize()
                # if self._args.enable_hook:
                #     self.run_test_hooks(eid, ex, results, eid == self._n_test_batches-1)
                self.run_test_hooks(eid, ex, results, eid == self._n_test_batches - 1)
                pbar.update(1)
                self._timer.update_from_start("test_step")
                # print(results)

        logging.info("Test performance {}s ({} * {})".format(
            self._timer.get_avg("test_step"), self._test_batch_size,
            self._n_test_batches))
        return None

    def save(self, prefix="latest"):
        checkpoint_dir = pathlib.Path(self._checkpoint_dir)
        existed_paths = sorted(list(
            checkpoint_dir.glob("**/{}-*.params".format(prefix))),
            key=lambda path: path.stat().st_mtime)
        for path in existed_paths[:-self._args.keep_checkpoint_max]:
            logging.info("Try to remove {}".format(path))
            path.unlink()

        ckp_path = self._checkpoint_dir / "{}-{:09}.params".format(
            prefix, self._global_step)
        torch.save({
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "lr_scheduler_state_dict": self._scheduler.state_dict() if self._scheduler is not None else None,
            "epoch": self._global_epoch,
            "step": self._global_step,
        },
            ckp_path)
        logging.info("Save to {}".format(ckp_path))
        current_time = datetime.datetime.today().strftime('%Y-%m-%d_%H_%M_%S')

        meta_info_file = self._checkpoint_dir / "meta.json"
        if meta_info_file.exists():
            with meta_info_file.open() as f:
                meta_info = json.load(f)
            # meta_info = json.load(meta_info_file.open())
        else:
            meta_info = {}
        meta_info["time"] = current_time
        meta_info["latest_loss"] = self._current_loss
        meta_info['best_model'] = self._metrics.best()
        meta_info['latest_eval'] = self._metrics.latest()
        meta_info["train_loss_history"] = self._train_loss_history
        meta_info["val_loss_history"] = self._val_loss_history
        meta_info["latest_step"] = self._global_step
        meta_info["latest_epoch"] = self._global_epoch
        with meta_info_file.open('w') as f:
            f.write(json.dumps(meta_info, indent=4))
        # meta_info_file.open('w').write(json.dumps(meta_info, indent=4))

    def load_state_dict(self):
        """
        load params to model if exists.
        """
        meta_info_file = self._checkpoint_dir / "meta.json"
        with meta_info_file.open() as f:
            meta_info = json.load(f)
        if self._args.reload == 'latest':
            filename = "latest-{:09}".format(meta_info['latest_step'])
            self._global_step = meta_info['latest_step']
            self._global_epoch = meta_info['latest_epoch']
        else:
            filename = "best-{:09}".format(meta_info['best_model']['step'])
            self._global_step = meta_info['best_model']['step']
            self._global_epoch = meta_info['best_model']['epoch']
        self._metrics.restore_from_meta(meta_info)
        logging.info("=============================================")
        logging.info(
            f"Loading exist param from {filename}"
            f"(S{self._global_step}/E{self._global_epoch})")
        logging.info("Loaded metrics {}".format(str(self._metrics)))
        logging.info("=============================================")

        filename = self._checkpoint_dir / "{}.params".format(filename)
        if self._args.is_dist:
            saved_data = torch.load(filename, map_location={'cuda:0': f'cuda:{self._args.local_rank}'}) # TODO, check map_location
        else:
            saved_data = torch.load(filename)
        if "model_state_dict" not in saved_data:
            self._model.load_state_dict(saved_data)
        else:
            self._model.load_state_dict(saved_data["model_state_dict"])
            self._optimizer.load_state_dict(
                saved_data["optimizer_state_dict"])
            if self._scheduler is not None:
                self._scheduler.load_state_dict(
                    saved_data["lr_scheduler_state_dict"])
            self._global_epoch = saved_data["epoch"]
            self._global_step = saved_data["step"]

    def write_metrics(self, raw_metrics, prefix):
        def iter_dict(metrics, prefix):
            for key in metrics:
                if (isinstance(metrics[key], (int, float))
                        or np.isscalar(metrics[key])):
                    self._writer.add_scalar('{}/{}'.format(prefix, key),
                                            metrics[key], self._global_step)
                elif metrics[key] is not None:
                    iter_dict(metrics[key], prefix="{}/{}".format(prefix, key))

        if raw_metrics is not None:
            iter_dict(raw_metrics, prefix)
