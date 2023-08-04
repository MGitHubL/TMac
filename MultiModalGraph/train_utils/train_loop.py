import logging
import os
import weakref
from collections import OrderedDict
from typing import Optional

import torch
from torch.nn.parallel import DistributedDataParallel

import MultiModalGraph.utils.comm as comm
import MultiModalGraph.utils.hooks as hooks
from MultiModalGraph.data.build import build_AudioSet_data_loaders
from MultiModalGraph.evaluation.AudioSet_evaluation import AudioSetEvaluator
from MultiModalGraph.evaluation.evaluator import (
    DatasetEvaluator,
    DatasetEvaluators,
    inference_on_dataset,
)
from MultiModalGraph.model.build import build_model
from MultiModalGraph.solver.build import build_lr_scheduler, build_optimizer
from MultiModalGraph.train_utils.training_utils import (
    AMPTrainer,
    SimpleTrainer,
    TrainerBase,
)
from MultiModalGraph.utils.events import (
    CommonMetricPrinter,
    JSONWriter,
    TensorboardXWriter,
)
from MultiModalGraph.utils.logger import print_csv_format, setup_logger
from MultiModalGraph.utils.utils import DetectionCheckpointer, verify_results

__all__ = ["DefaultTrainer", "create_ddp_model"]


def create_ddp_model(model, *, fp16_compression=False, **kwargs):
    """
    Create a DistributedDataParallel model if there are >1 processes.

    Args:
        model: a torch.nn.Module
        fp16_compression: add fp16 compression hooks to the ddp object.
            See more at https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """  # noqa
    if comm.get_world_size() == 1:
        return model
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [comm.get_local_rank()]
    ddp = DistributedDataParallel(model, **kwargs)
    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp


class DefaultTrainer(TrainerBase):
    """
    A trainer with default training logic. It does the following:

    1. Create a :class:`SimpleTrainer` using model, optimizer, dataloader
       defined by the given config. Create a LR scheduler defined by the config.
    2. Load the last checkpoint or `cfg.MODEL.WEIGHTS`, if exists, when
       `resume_or_load` is called.
    3. Register a few common hooks defined by the config.

    It is created to simplify the **standard model training workflow** and reduce code boilerplate
    for users who only need the standard training workflow, with standard features.
    It means this class makes *many assumptions* about your training logic that
    may easily become invalid in a new research. In fact, any assumptions beyond those made in the
    :class:`SimpleTrainer` are too much for research.

    The code of this class has been annotated about restrictive assumptions it makes.
    When they do not work for you, you're encouraged to:

    1. Overwrite methods of this class, OR:
    2. Use :class:`SimpleTrainer`, which only does minimal SGD training and
       nothing else. You can then add your own hooks if needed. OR:
    3. Write your own training loop similar to `tools/plain_train_net.py`.

    See the :doc:`/tutorials/training` tutorials for more details.

    Note that the behavior of this class, like other functions/classes in
    this file, is not stable, since it is meant to represent the "common default behavior".
    It is only guaranteed to work well with the standard models and training workflow in detectron2.
    To obtain more stable behavior, write your own training logic with other public APIs.

    Examples:
    ::
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load()  # load last checkpoint or MODEL.WEIGHTS
        trainer.train()

    Attributes:
        scheduler:
        checkpointer (DetectionCheckpointer):
        cfg (CfgNode):
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__()
        logger = logging.getLogger("MultiModalGraph")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader_tr = self.build_data_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)

        with torch.no_grad():  # Initialize lazy modules.
            data = iter(data_loader_tr).next()
            out = model(data)

        # check how the random hyperparameters are distributed for graph construction
        # from collections import defaultdict
        # Graph_param = defaultdict(list)
        # for i in range(10):
        #     data = iter(data_loader_tr).next()
        #     Graph_param['span_over_time_between'].extend(data['span_over_time_between'].numpy())
        #     Graph_param['span_over_time_audio'].extend(data['span_over_time_audio'].numpy())
        #     Graph_param['audio_dilation'].extend(data['audio_dilation'].numpy())
        #     Graph_param['span_over_time_video'].extend(data['span_over_time_video'].numpy())
        #     Graph_param['video_dilation'].extend(data['video_dilation'].numpy())
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.hist(Graph_param['span_over_time_between'])
        # plt.title('span_over_time_between')
        # plt.ylabel('Number of occurrence for 10 epochs')
        # plt.savefig('span_over_time_between.png')
        # plt.figure()
        # plt.hist(Graph_param['span_over_time_audio'])
        # plt.title('span_over_time_audio')
        # plt.ylabel('Number of occurrence for 10 epochs')
        # plt.savefig('span_over_time_audio.png')
        # plt.figure()
        # plt.hist(Graph_param['audio_dilation'])
        # plt.title('audio_dilation')
        # plt.ylabel('Number of occurrence for 10 epochs')
        # plt.savefig('audio_dilation.png')
        # plt.figure()
        # plt.hist(Graph_param['span_over_time_video'])
        # plt.title('span_over_time_video')
        # plt.ylabel('Number of occurrence for 10 epochs')
        # plt.savefig('span_over_time_video.png')
        # plt.figure()
        # plt.hist(Graph_param['video_dilation'])
        # plt.title('video_dilation')
        # plt.ylabel('Number of occurrence for 10 epochs')
        # plt.savefig('video_dilation.png')

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if params > 1e6:
            print("Number of trainbale params: %.2f M" % (params / 1e6))
        else:
            print("Number of trainbale params: %.2f K" % (params / 1e3))

        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader_tr, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """
        # self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        #
        # if resume and self.checkpointer.has_checkpoint():
        #     # The checkpoint stores the training iteration that just finished, thus we start
        #     # at the next iteration
        #     self.start_iter = self.iter + 1
        ### Added by me to load the early stopping checkpoint
        if resume:
            early_stop_file = os.path.join(
                self.cfg.OUTPUT_DIR, "EarlyStopping_checkpoint.pt"
            )
            if os.path.exists(early_stop_file):
                self.model.load_state_dict(torch.load(early_stop_file))
                logger = logging.getLogger(__name__)
                logger.info("Loaded EarlyStopping_checkpoint.pt")
                self.start_iter = self.max_iter + 1
        ###

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        def load_best_model():
            self.load_state_dict(
                state_dict_path="checkpoints/EarlyStopping_checkpoint.pt"
            )

        ret = [
            hooks.LoadEarlyStoppingModel(load_best_model),
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            # hooks.EarlyStopping(max_patience=cfg.TEST.MAX_PATIENCE),
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results():
            self._last_test_results = self.test(self.cfg, self.model, cfg.DATASETS.TEST)
            return self._last_test_results

        def evaluate_and_save_results():
            self._last_eval_results = self.eval(self.cfg, self.model, cfg.DATASETS.TEST)
            return self.model, self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(
            hooks.EvalHook(
                cfg.TEST.EVAL_PERIOD,
                evaluate_and_save_results,
                max_patience=cfg.TEST.MAX_PATIENCE,
                verbose=True,
                path_to_save_best_model="checkpoints/",
            )
        )
        ret.append(hooks.TestHook(test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def build_writers(self):
        """
        Build a list of writers to be used using :func:`default_writers()`.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        """
        return default_writers(self.cfg.OUTPUT_DIR, self.max_iter)

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        super().train(self.start_iter, self.max_iter)
        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def run_step(self):
        self._trainer.iter = self.iter
        self._trainer.run_step()

    def load_state_dict(self, state_dict_path):
        logger = logging.getLogger(__name__)
        if os.path.exists(state_dict_path):
            logger.info(f"Loading state_dict from {state_dict_path}")
            self.model.state_dict().update(torch.load(state_dict_path))
        else:
            logger.info(f"{state_dict_path} not found")

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:

        It now calls :func:`detectron2.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_data_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """

        return build_AudioSet_data_loaders(cfg, "train")

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        # return build_detection_test_loader(cfg, dataset_name)
        return build_AudioSet_data_loaders(cfg, "test")

    @classmethod
    def build_eval_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        # return build_detection_test_loader(cfg, dataset_name)
        return build_AudioSet_data_loaders(cfg, "eval")

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, prefix="test"):
        """
        Returns:
            DatasetEvaluator or None

        It is not implemented by default.
        """
        evaluator_list = []
        for evaluator in cfg.DATASETS.TEST:
            if evaluator == "AudioSet":
                evaluator_list.append(
                    AudioSetEvaluator(
                        evaluator,
                        cfg.MODEL.OUT_DIM,
                        prefix=prefix,
                        loss=cfg.TRAINING.LOSS,
                        classes_name=cfg.DATALOADER.DISERED_CLASSES,
                        cfg=cfg,
                    )
                )
        if len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                if evaluators[idx] == "AudioSet":
                    try:
                        evaluator = cls.build_evaluator(
                            cfg, dataset_name, prefix="test"
                        )
                    except NotImplementedError:
                        logger.warn(
                            "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                            "or implement its `build_evaluator` method."
                        )
                    results[dataset_name] = {}
                    # continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Test results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    @classmethod
    def eval(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_eval_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                if evaluators[idx] == "AudioSet":
                    try:
                        evaluator = cls.build_evaluator(
                            cfg, dataset_name, prefix="eval"
                        )
                    except NotImplementedError:
                        logger.warn(
                            "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                            "or implement its `build_evaluator` method."
                        )
                    results[dataset_name] = {}
                    # continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info(
                    "Evaluation results for {} in csv format:".format(dataset_name)
                )
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    @staticmethod
    def auto_scale_workers(cfg, num_workers: int):
        """
        When the config is defined for certain number of workers (according to
        ``cfg.SOLVER.REFERENCE_WORLD_SIZE``) that's different from the number of
        workers currently in use, returns a new cfg where the total batch size
        is scaled so that the per-GPU batch size stays the same as the
        original ``IMS_PER_BATCH // REFERENCE_WORLD_SIZE``.

        Other config options are also scaled accordingly:
        * training steps and warmup steps are scaled inverse proportionally.
        * learning rate are scaled proportionally, following :paper:`ImageNet in 1h`.

        For example, with the original config like the following:

        .. code-block:: yaml

            IMS_PER_BATCH: 16
            BASE_LR: 0.1
            REFERENCE_WORLD_SIZE: 8
            MAX_ITER: 5000
            STEPS: (4000,)
            CHECKPOINT_PERIOD: 1000

        When this config is used on 16 GPUs instead of the reference number 8,
        calling this method will return a new config with:

        .. code-block:: yaml

            IMS_PER_BATCH: 32
            BASE_LR: 0.2
            REFERENCE_WORLD_SIZE: 16
            MAX_ITER: 2500
            STEPS: (2000,)
            CHECKPOINT_PERIOD: 500

        Note that both the original config and this new config can be trained on 16 GPUs.
        It's up to user whether to enable this feature (by setting ``REFERENCE_WORLD_SIZE``).

        Returns:
            CfgNode: a new config. Same as original if ``cfg.SOLVER.REFERENCE_WORLD_SIZE==0``.
        """
        old_world_size = cfg.SOLVER.REFERENCE_WORLD_SIZE
        if old_world_size == 0 or old_world_size == num_workers:
            return cfg
        cfg = cfg.clone()
        frozen = cfg.is_frozen()
        cfg.defrost()

        assert (
            cfg.SOLVER.IMS_PER_BATCH % old_world_size == 0
        ), "Invalid REFERENCE_WORLD_SIZE in config!"
        scale = num_workers / old_world_size
        bs = cfg.SOLVER.IMS_PER_BATCH = int(round(cfg.SOLVER.IMS_PER_BATCH * scale))
        lr = cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR * scale
        max_iter = cfg.SOLVER.MAX_ITER = int(round(cfg.SOLVER.MAX_ITER / scale))
        warmup_iter = cfg.SOLVER.WARMUP_ITERS = int(
            round(cfg.SOLVER.WARMUP_ITERS / scale)
        )
        cfg.SOLVER.STEPS = tuple(int(round(s / scale)) for s in cfg.SOLVER.STEPS)
        cfg.TEST.EVAL_PERIOD = int(round(cfg.TEST.EVAL_PERIOD / scale))
        cfg.SOLVER.CHECKPOINT_PERIOD = int(round(cfg.SOLVER.CHECKPOINT_PERIOD / scale))
        cfg.SOLVER.REFERENCE_WORLD_SIZE = num_workers  # maintain invariant
        logger = logging.getLogger(__name__)
        logger.info(
            f"Auto-scaling the config to batch_size={bs}, learning_rate={lr}, "
            f"max_iter={max_iter}, warmup={warmup_iter}."
        )

        if frozen:
            cfg.freeze()
        return cfg


def default_writers(output_dir: str, max_iter: Optional[int] = None):
    """
    Build a list of :class:`EventWriter` to be used.
    It now consists of a :class:`CommonMetricPrinter`,
    :class:`TensorboardXWriter` and :class:`JSONWriter`.

    Args:
        output_dir: directory to store JSON metrics and tensorboard events
        max_iter: the total number of iterations

    Returns:
        list[EventWriter]: a list of :class:`EventWriter` objects.
    """
    return [
        # It may not always print what you want to see, since it prints "common" metrics only.
        CommonMetricPrinter(max_iter),
        JSONWriter(os.path.join(output_dir, "metrics.json")),
        TensorboardXWriter(output_dir),
    ]


# Access basic attributes from the underlying trainer
for _attr in [
    "model",
    "data_loader_tr",
    "data_loader_te",
    "data_loader_eval",
    "optimizer",
]:
    setattr(
        DefaultTrainer,
        _attr,
        property(
            # getter
            lambda self, x=_attr: getattr(self._trainer, x),
            # setter
            lambda self, value, x=_attr: setattr(self._trainer, x, value),
        ),
    )
