# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import time
import weakref
from typing import List, Mapping, Optional

import numpy as np
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

import MultiModalGraph.train_utils.comm as comm
from MultiModalGraph.utils.events import EventStorage, get_event_storage
from MultiModalGraph.utils.logger import _log_api_usage

__all__ = ["HookBase", "TrainerBase", "SimpleTrainer", "AMPTrainer"]


class HookBase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.

    Each hook can implement 4 methods. The way they are called is demonstrated
    in the following snippet:
    ::
        hook.before_train()
        for iter in range(start_iter, max_iter):
            hook.before_step()
            trainer.run_step()
            hook.after_step()
        iter += 1
        hook.after_train()

    Notes:
        1. In the hook method, users can access ``self.trainer`` to access more
           properties about the context (e.g., model, current iteration, or config
           if using :class:`DefaultTrainer`).

        2. A hook that does something in :meth:`before_step` can often be
           implemented equivalently in :meth:`after_step`.
           If the hook takes non-trivial time, it is strongly recommended to
           implement the hook in :meth:`after_step` instead of :meth:`before_step`.
           The convention is that :meth:`before_step` should only take negligible time.

           Following this convention will allow hooks that do care about the difference
           between :meth:`before_step` and :meth:`after_step` (e.g., timer) to
           function properly.

    """

    trainer: "TrainerBase" = None
    """
    A weak reference to the trainer object. Set by the trainer when the hook is registered.
    """

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        pass

    def state_dict(self):
        """
        Hooks are stateless by default, but can be made checkpointable by
        implementing `state_dict` and `load_state_dict`.
        """
        return {}


class TrainerBase:
    """
    Base class for iterative trainer with hooks.

    The only assumption we made here is: the training runs in a loop.
    A subclass can implement what the loop is.
    We made no assumptions about the existence of dataloader, optimizer, model, etc.

    Attributes:
        iter(int): the current iteration.

        start_iter(int): The iteration to start with.
            By convention the minimum possible value is 0.

        max_iter(int): The iteration to end training.

        storage(EventStorage): An EventStorage that's opened during the course of training.
    """

    def __init__(self) -> None:
        self._hooks: List[HookBase] = []
        self.iter: int = 0
        self.start_iter: int = 0
        self.max_iter: int
        self.storage: EventStorage
        _log_api_usage("trainer." + self.__class__.__name__)

    def register_hooks(self, hooks: List[Optional[HookBase]]) -> None:
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def train(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
                # self.iter == max_iter can be used by `after_train` to
                # tell whether the training successfully finished or failed
                # due to exceptions.
                self.iter += 1
            except StopIteration:
                logger.exception("Early Stopping happened.")
                # raise
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        self.storage.iter = self.iter
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        # Maintain the invariant that storage.iter == trainer.iter
        # for the entire execution of each step
        self.storage.iter = self.iter

        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()

    def run_step(self):
        raise NotImplementedError

    def state_dict(self):
        ret = {"iteration": self.iter}
        hooks_state = {}
        for h in self._hooks:
            sd = h.state_dict()
            if sd:
                name = type(h).__qualname__
                if name in hooks_state:
                    # TODO handle repetitive stateful hooks
                    continue
                hooks_state[name] = sd
        if hooks_state:
            ret["hooks"] = hooks_state
        return ret

    def load_state_dict(self, state_dict):
        logger = logging.getLogger(__name__)
        self.iter = state_dict["iteration"]
        for key, value in state_dict.get("hooks", {}).items():
            for h in self._hooks:
                try:
                    name = type(h).__qualname__
                except AttributeError:
                    continue
                if name == key:
                    h.load_state_dict(value)
                    break
            else:
                logger.warning(
                    f"Cannot find the hook '{key}', its state_dict is ignored."
                )


class SimpleTrainer(TrainerBase):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization,
    optionally using data-parallelism.
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.

    All other tasks during training (checkpointing, logging, evaluation, LR schedule)
    are maintained by hooks, which can be registered by :meth:`TrainerBase.register_hooks`.

    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(self, model, data_loader, optimizer):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
        """
        super().__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        model.train()

        self.model = model
        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)
        self.optimizer = optimizer

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict, metric = self.model(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        self._write_metrics(loss_dict, data_time)
        # just support mAP and accuracy, to add go to events line 466
        self._write_metrics(metric, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()

    def _write_metrics(
        self, loss_dict: Mapping[str, torch.Tensor], data_time: float, prefix: str = "",
    ) -> None:
        SimpleTrainer.write_metrics(loss_dict, data_time, prefix)

    @staticmethod
    def write_metrics(
        loss_dict: Mapping[str, torch.Tensor], data_time: float, prefix: str = "",
    ) -> None:
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
            prefix (str): prefix for logging keys
        """
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={storage.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            for k, v in metrics_dict.items():
                if "loss" in k:
                    storage.put_scalar(
                        "{}total_loss".format(prefix), total_losses_reduced
                    )
                    break
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)

    def state_dict(self):
        ret = super().state_dict()
        ret["optimizer"] = self.optimizer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.optimizer.load_state_dict(state_dict["optimizer"])


class AMPTrainer(SimpleTrainer):
    """
    Like :class:`SimpleTrainer`, but uses PyTorch's native automatic mixed precision
    in the training loop.
    """

    def __init__(self, model, data_loader, optimizer, grad_scaler=None):
        """
        Args:
            model, data_loader, optimizer: same as in :class:`SimpleTrainer`.
            grad_scaler: torch GradScaler to automatically scale gradients.
        """
        unsupported = (
            "AMPTrainer does not support single-process multi-device training!"
        )
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        super().__init__(model, data_loader, optimizer)

        if grad_scaler is None:
            from torch.cuda.amp import GradScaler

            grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler

    def run_step(self):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert (
            torch.cuda.is_available()
        ), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        with autocast():
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        self.grad_scaler.scale(losses).backward()

        self._write_metrics(loss_dict, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

    def state_dict(self):
        ret = super().state_dict()
        ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.grad_scaler.load_state_dict(state_dict["grad_scaler"])


# class DefaultTrainer(TrainerBase):
#     """
#     A trainer with default training logic. It does the following:
#
#     1. Create a :class:`SimpleTrainer` using model, optimizer, dataloader
#        defined by the given config. Create a LR scheduler defined by the config.
#     2. Load the last checkpoint or `cfg.MODEL.WEIGHTS`, if exists, when
#        `resume_or_load` is called.
#     3. Register a few common hooks defined by the config.
#
#     It is created to simplify the **standard model training workflow** and reduce code boilerplate
#     for users who only need the standard training workflow, with standard features.
#     It means this class makes *many assumptions* about your training logic that
#     may easily become invalid in a new research. In fact, any assumptions beyond those made in the
#     :class:`SimpleTrainer` are too much for research.
#
#     The code of this class has been annotated about restrictive assumptions it makes.
#     When they do not work for you, you're encouraged to:
#
#     1. Overwrite methods of this class, OR:
#     2. Use :class:`SimpleTrainer`, which only does minimal SGD training and
#        nothing else. You can then add your own hooks if needed. OR:
#     3. Write your own training loop similar to `tools/plain_train_net.py`.
#
#     See the :doc:`/tutorials/training` tutorials for more details.
#
#     Note that the behavior of this class, like other functions/classes in
#     this file, is not stable, since it is meant to represent the "common default behavior".
#     It is only guaranteed to work well with the standard models and training workflow in detectron2.
#     To obtain more stable behavior, write your own training logic with other public APIs.
#
#     Examples:
#     ::
#         trainer = DefaultTrainer(cfg)
#         trainer.resume_or_load()  # load last checkpoint or MODEL.WEIGHTS
#         trainer.train()
#
#     Attributes:
#         scheduler:
#         checkpointer (DetectionCheckpointer):
#         cfg (CfgNode):
#     """
#
#     def __init__(self, cfg):
#         """
#         Args:
#             cfg (CfgNode):
#         """
#         super().__init__()
#         logger = logging.getLogger("detectron2")
#         if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
#             setup_logger()
#         cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
#
#         # Assume these objects must be constructed in this order.
#         model = self.build_model(cfg)
#         optimizer = self.build_optimizer(cfg, model)
#         data_loader = self.build_train_loader(cfg)
#
#         model = create_ddp_model(model, broadcast_buffers=False)
#         self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
#             model, data_loader, optimizer
#         )
#
#         self.scheduler = self.build_lr_scheduler(cfg, optimizer)
#         self.checkpointer = DetectionCheckpointer(
#             # Assume you want to save checkpoints together with logs/statistics
#             model,
#             cfg.OUTPUT_DIR,
#             trainer=weakref.proxy(self),
#         )
#         self.start_iter = 0
#         self.max_iter = cfg.SOLVER.MAX_ITER
#         self.cfg = cfg
#
#         self.register_hooks(self.build_hooks())
#
#     def resume_or_load(self, resume=True):
#         """
#         If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
#         a `last_checkpoint` file), resume from the file. Resuming means loading all
#         available states (eg. optimizer and scheduler) and update iteration counter
#         from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
#
#         Otherwise, this is considered as an independent training. The method will load model
#         weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
#         from iteration 0.
#
#         Args:
#             resume (bool): whether to do resume or not
#         """
#         self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
#         if resume and self.checkpointer.has_checkpoint():
#             # The checkpoint stores the training iteration that just finished, thus we start
#             # at the next iteration
#             self.start_iter = self.iter + 1
#
#     def build_hooks(self):
#         """
#         Build a list of default hooks, including timing, evaluation,
#         checkpointing, lr scheduling, precise BN, writing events.
#
#         Returns:
#             list[HookBase]:
#         """
#         cfg = self.cfg.clone()
#         cfg.defrost()
#         cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN
#
#         ret = [
#             hooks.IterationTimer(),
#             hooks.LRScheduler(),
#             hooks.PreciseBN(
#                 # Run at the same freq as (but before) evaluation.
#                 cfg.TEST.EVAL_PERIOD,
#                 self.model,
#                 # Build a new data loader to not affect training
#                 self.build_train_loader(cfg),
#                 cfg.TEST.PRECISE_BN.NUM_ITER,
#             )
#             if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
#             else None,
#         ]
#
#         # Do PreciseBN before checkpointer, because it updates the model and need to
#         # be saved by checkpointer.
#         # This is not always the best: if checkpointing has a different frequency,
#         # some checkpoints may have more precise statistics than others.
#         if comm.is_main_process():
#             ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))
#
#         def test_and_save_results():
#             self._last_eval_results = self.test(self.cfg, self.model)
#             return self._last_eval_results
#
#         # Do evaluation after checkpointer, because then if it fails,
#         # we can use the saved checkpoint to debug.
#         ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))
#
#         if comm.is_main_process():
#             # Here the default print/log frequency of each writer is used.
#             # run writers in the end, so that evaluation metrics are written
#             ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
#         return ret
#
#     def build_writers(self):
#         """
#         Build a list of writers to be used using :func:`default_writers()`.
#         If you'd like a different list of writers, you can overwrite it in
#         your trainer.
#
#         Returns:
#             list[EventWriter]: a list of :class:`EventWriter` objects.
#         """
#         return default_writers(self.cfg.OUTPUT_DIR, self.max_iter)
#
#     def train(self):
#         """
#         Run training.
#
#         Returns:
#             OrderedDict of results, if evaluation is enabled. Otherwise None.
#         """
#         super().train(self.start_iter, self.max_iter)
#         if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
#             assert hasattr(
#                 self, "_last_eval_results"
#             ), "No evaluation results obtained during training!"
#             verify_results(self.cfg, self._last_eval_results)
#             return self._last_eval_results
#
#     def run_step(self):
#         self._trainer.iter = self.iter
#         self._trainer.run_step()
#
#     @classmethod
#     def build_model(cls, cfg):
#         """
#         Returns:
#             torch.nn.Module:
#
#         It now calls :func:`detectron2.modeling.build_model`.
#         Overwrite it if you'd like a different model.
#         """
#         model = build_model(cfg)
#         logger = logging.getLogger(__name__)
#         logger.info("Model:\n{}".format(model))
#         return model
#
#     @classmethod
#     def build_optimizer(cls, cfg, model):
#         """
#         Returns:
#             torch.optim.Optimizer:
#
#         It now calls :func:`detectron2.solver.build_optimizer`.
#         Overwrite it if you'd like a different optimizer.
#         """
#         return build_optimizer(cfg, model)
#
#     @classmethod
#     def build_lr_scheduler(cls, cfg, optimizer):
#         """
#         It now calls :func:`detectron2.solver.build_lr_scheduler`.
#         Overwrite it if you'd like a different scheduler.
#         """
#         return build_lr_scheduler(cfg, optimizer)
#
#     @classmethod
#     def build_train_loader(cls, cfg):
#         """
#         Returns:
#             iterable
#
#         It now calls :func:`detectron2.data.build_detection_train_loader`.
#         Overwrite it if you'd like a different data loader.
#         """
#         return build_detection_train_loader(cfg)
#
#     @classmethod
#     def build_test_loader(cls, cfg, dataset_name):
#         """
#         Returns:
#             iterable
#
#         It now calls :func:`detectron2.data.build_detection_test_loader`.
#         Overwrite it if you'd like a different data loader.
#         """
#         return build_detection_test_loader(cfg, dataset_name)
#
#     @classmethod
#     def build_evaluator(cls, cfg, dataset_name):
#         """
#         Returns:
#             DatasetEvaluator or None
#
#         It is not implemented by default.
#         """
#         raise NotImplementedError(
#             """
# If you want DefaultTrainer to automatically run evaluation,
# please implement `build_evaluator()` in subclasses (see train_net.py for example).
# Alternatively, you can call evaluation functions yourself (see Colab balloon tutorial for example).
# """
#         )
#
#     @classmethod
#     def test(cls, cfg, model, evaluators=None):
#         """
#         Args:
#             cfg (CfgNode):
#             model (nn.Module):
#             evaluators (list[DatasetEvaluator] or None): if None, will call
#                 :meth:`build_evaluator`. Otherwise, must have the same length as
#                 ``cfg.DATASETS.TEST``.
#
#         Returns:
#             dict: a dict of result metrics
#         """
#         logger = logging.getLogger(__name__)
#         if isinstance(evaluators, DatasetEvaluator):
#             evaluators = [evaluators]
#         if evaluators is not None:
#             assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
#                 len(cfg.DATASETS.TEST), len(evaluators)
#             )
#
#         results = OrderedDict()
#         for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
#             data_loader = cls.build_test_loader(cfg, dataset_name)
#             # When evaluators are passed in as arguments,
#             # implicitly assume that evaluators can be created before data_loader.
#             if evaluators is not None:
#                 evaluator = evaluators[idx]
#             else:
#                 try:
#                     evaluator = cls.build_evaluator(cfg, dataset_name)
#                 except NotImplementedError:
#                     logger.warn(
#                         "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
#                         "or implement its `build_evaluator` method."
#                     )
#                     results[dataset_name] = {}
#                     continue
#             results_i = inference_on_dataset(model, data_loader, evaluator)
#             results[dataset_name] = results_i
#             if comm.is_main_process():
#                 assert isinstance(
#                     results_i, dict
#                 ), "Evaluator must return a dict on the main process. Got {} instead.".format(
#                     results_i
#                 )
#                 logger.info("Evaluation results for {} in csv format:".format(dataset_name))
#                 print_csv_format(results_i)
#
#         if len(results) == 1:
#             results = list(results.values())[0]
#         return results
#
#     @staticmethod
#     def auto_scale_workers(cfg, num_workers: int):
#         """
#         When the config is defined for certain number of workers (according to
#         ``cfg.SOLVER.REFERENCE_WORLD_SIZE``) that's different from the number of
#         workers currently in use, returns a new cfg where the total batch size
#         is scaled so that the per-GPU batch size stays the same as the
#         original ``IMS_PER_BATCH // REFERENCE_WORLD_SIZE``.
#
#         Other config options are also scaled accordingly:
#         * training steps and warmup steps are scaled inverse proportionally.
#         * learning rate are scaled proportionally, following :paper:`ImageNet in 1h`.
#
#         For example, with the original config like the following:
#
#         .. code-block:: yaml
#
#             IMS_PER_BATCH: 16
#             BASE_LR: 0.1
#             REFERENCE_WORLD_SIZE: 8
#             MAX_ITER: 5000
#             STEPS: (4000,)
#             CHECKPOINT_PERIOD: 1000
#
#         When this config is used on 16 GPUs instead of the reference number 8,
#         calling this method will return a new config with:
#
#         .. code-block:: yaml
#
#             IMS_PER_BATCH: 32
#             BASE_LR: 0.2
#             REFERENCE_WORLD_SIZE: 16
#             MAX_ITER: 2500
#             STEPS: (2000,)
#             CHECKPOINT_PERIOD: 500
#
#         Note that both the original config and this new config can be trained on 16 GPUs.
#         It's up to user whether to enable this feature (by setting ``REFERENCE_WORLD_SIZE``).
#
#         Returns:
#             CfgNode: a new config. Same as original if ``cfg.SOLVER.REFERENCE_WORLD_SIZE==0``.
#         """
#         old_world_size = cfg.SOLVER.REFERENCE_WORLD_SIZE
#         if old_world_size == 0 or old_world_size == num_workers:
#             return cfg
#         cfg = cfg.clone()
#         frozen = cfg.is_frozen()
#         cfg.defrost()
#
#         assert (
#             cfg.SOLVER.IMS_PER_BATCH % old_world_size == 0
#         ), "Invalid REFERENCE_WORLD_SIZE in config!"
#         scale = num_workers / old_world_size
#         bs = cfg.SOLVER.IMS_PER_BATCH = int(round(cfg.SOLVER.IMS_PER_BATCH * scale))
#         lr = cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR * scale
#         max_iter = cfg.SOLVER.MAX_ITER = int(round(cfg.SOLVER.MAX_ITER / scale))
#         warmup_iter = cfg.SOLVER.WARMUP_ITERS = int(round(cfg.SOLVER.WARMUP_ITERS / scale))
#         cfg.SOLVER.STEPS = tuple(int(round(s / scale)) for s in cfg.SOLVER.STEPS)
#         cfg.TEST.EVAL_PERIOD = int(round(cfg.TEST.EVAL_PERIOD / scale))
#         cfg.SOLVER.CHECKPOINT_PERIOD = int(round(cfg.SOLVER.CHECKPOINT_PERIOD / scale))
#         cfg.SOLVER.REFERENCE_WORLD_SIZE = num_workers  # maintain invariant
#         logger = logging.getLogger(__name__)
#         logger.info(
#             f"Auto-scaling the config to batch_size={bs}, learning_rate={lr}, "
#             f"max_iter={max_iter}, warmup={warmup_iter}."
#         )
#
#         if frozen:
#             cfg.freeze()
#         return cfg
