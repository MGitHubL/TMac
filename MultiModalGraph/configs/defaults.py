from MultiModalGraph.configs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# The version number, to upgrade from old configs to new ones if any
# changes happen. It's recommended to keep a VERSION in your config file.
_C.VERSION = 0

_C.MODEL = CN()
_C.MODEL.LOAD_PROPOSALS = False
_C.MODEL.MASK_ON = False
_C.MODEL.KEYPOINT_ON = False
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"


# Path (a file path, or URL like detectron2://.., https://..) to a checkpoint file
# to be loaded to the model. You can find available models in the model zoo.
_C.MODEL.WEIGHTS = ""

# Values to be used for image normalization (BGR order, since INPUT.FORMAT defaults to BGR).
# To train on images of different number of channels, just set different mean & std.
# Default values are the mean pixel value from ImageNet: [103.53, 116.28, 123.675]
_C.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
# When using pre-trained models in Detectron1 or any MSRA models,
# std has been absorbed into its conv1 weights, so the std needs to be set 1.
# Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
_C.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the smallest side of the image during training
_C.INPUT.MIN_SIZE_TRAIN = (800,)
# Sample size of smallest side by choice or random selection from range give by
# INPUT.MIN_SIZE_TRAIN
_C.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
# Maximum size of the side of the image during training
_C.INPUT.MAX_SIZE_TRAIN = 1333
# Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
_C.INPUT.MIN_SIZE_TEST = 800
# Maximum size of the side of the image during testing
_C.INPUT.MAX_SIZE_TEST = 1333


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training. Must be registered in DatasetCatalog
# Samples from these datasets will be merged and used as one dataset.
_C.DATASETS.TRAIN = ("AudioSet",)
# List of the dataset names for testing. Must be registered in DatasetCatalog
_C.DATASETS.TEST = ("AudioSet",)
# Train ratio
_C.DATASETS.TRAIN_RATIO = 0.7
# Test ratio
_C.DATASETS.TEST_RATIO = 0.2
# Eval ratio
_C.DATASETS.EVAL_RATIO = 0.1
# Path to the train data
_C.DATASETS.TRAIN_PATH = ""
# Path to the eval data
_C.DATASETS.EVAL_PATH = ""
# Path to the test data
_C.DATASETS.TEST_PATH = ""

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# batch size
_C.DATALOADER.BATCH_SIZE = 32
# set data sampler method
_C.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
#
_C.DATALOADER.REPEAT_THRESHOLD = 0.1
# favoured classes, defaut: "All"
_C.DATALOADER.DISERED_CLASSES = ["All"]
# stratified split
_C.DATALOADER.STRATIFIED_SPLIT = False

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.AUDIO_BACKBONE = CN()

_C.MODEL.AUDIO_BACKBONE.NAME = "Vggish"
_C.MODEL.AUDIO_BACKBONE.PRETRAINED_ON = "Vggish"

_C.MODEL.VIDEO_BACKBONE = CN()

_C.MODEL.VIDEO_BACKBONE.NAME = "CoCLR"
_C.MODEL.VIDEO_BACKBONE.PRETRAINED_ON = ""

_C.MODEL.IMAGE_BACKBONE = CN()

_C.MODEL.IMAGE_BACKBONE.NAME = "Vggish"
_C.MODEL.IMAGE_BACKBONE.PRETRAINED_ON = "Vggish"

# set network params
# middle representation dimension
_C.MODEL.HIDDEN_CHANNELS = 128
# output representation dimension (number of classes in classification task)
if _C.DATALOADER.DISERED_CLASSES != ["All"]:
    _C.MODEL.OUT_DIM = len(_C.DATALOADER.DISERED_CLASSES)
else:
    _C.MODEL.OUT_DIM = 527
# number of layers in the network
_C.MODEL.NUM_LAYERS = 2

# -----------------------------------------------------------------------------
# Graph construction options
# -----------------------------------------------------------------------------
_C.GRAPH = CN()

# Number of temporal connections with the neighbours for each node
_C.GRAPH.SPAN_OVER_TIME_AUDIO = 2
_C.GRAPH.AUDIO_DILATION = 1
_C.GRAPH.SPAN_OVER_TIME_VIDEO = 2
_C.GRAPH.VIDEO_DILATION = 1
_C.GRAPH.SPAN_OVER_TIME_BETWEEN = 2
# Dynamic graph construction
_C.GRAPH.DYNAMIC = False
# Normalize feature
_C.GRAPH.NORMALIZE = True
# Add self-loops to the graph
_C.GRAPH.SELF_LOOPS = True
# Add fusion layers
_C.GRAPH.FUSION_LAYERS = []

# -----------------------------------------------------------------------------
# TRAINING
# -----------------------------------------------------------------------------
_C.TRAINING = CN()
# Set the loss function
_C.TRAINING.LOSS = "CrossEntropyLoss"  # "FocalLoss"

# ---------------------------------------------------------------------------- #
# solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

# See detectron2/solver/build.py for LR scheduler options
_C.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"

_C.SOLVER.MAX_ITER = 500  # 40000

_C.SOLVER.BASE_LR = 0.001

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.NESTEROV = False

_C.SOLVER.WEIGHT_DECAY = 0.0001
# The weight decay that's applied to parameters of normalization layers
# (typically the affine transformation)
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0

_C.SOLVER.GAMMA = 0.1
# The iteration number to decrease learning rate by GAMMA.
_C.SOLVER.STEPS = (100,)  # 3000

_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
_C.SOLVER.WARMUP_ITERS = 3000  # 1000
_C.SOLVER.WARMUP_METHOD = "linear"

# Save a checkpoint after every this number of iterations
_C.SOLVER.CHECKPOINT_PERIOD = 4990  # 5000

# The reference number of workers (GPUs) this config is meant to train with.
# It takes no effect when set to 0.
# With a non-zero value, it will be used by DefaultTrainer to compute a desired
# per-worker batch size, and then scale the other related configs (total batch size,
# learning rate, etc) to match the per-worker batch size.
# See documentation of `DefaultTrainer.auto_scale_workers` for details:
_C.SOLVER.REFERENCE_WORLD_SIZE = 0

# Detectron v1 (and previous detection code) used a 2x higher LR and 0 WD for
# biases. This is not useful (at least for recent models). You should avoid
# changing these and they exist only to reproduce Detectron v1 training if
# desired.
_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.WEIGHT_DECAY_BIAS = _C.SOLVER.WEIGHT_DECAY

# Gradient clipping
_C.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": False})
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
# Maximum absolute value used for clipping gradients
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

# Enable automatic mixed precision for training
# Note that this does not change model's inference behavior.
# To use AMP in inference, run inference under autocast()
_C.SOLVER.AMP = CN({"ENABLED": False})

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# For end-to-end tests to verify the expected accuracy.
# Each item is [task, metric, value, tolerance]
# e.g.: [['bbox', 'AP', 38.5, 0.2]]
_C.TEST.EXPECTED_RESULTS = []
# The period (in terms of steps) to evaluate the model during training.
# Set to 0 to disable.
_C.TEST.EVAL_PERIOD = 1000
# The early stopping maximum patience for evaluation. Set to 0 to disable.
_C.TEST.MAX_PATIENCE = 5

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Directory where output files are written
_C.OUTPUT_DIR = "./checkpoints"
# Set seed to negative to fully randomize everything.
# Set seed to positive to use a fixed seed. Note that a fixed seed increases
# reproducibility but does not guarantee fully deterministic behavior.
# Disabling all parallelism further increases reproducibility.
_C.SEED = 1234  # -1
# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same or similar sizes, benchmark is often helpful.
_C.CUDNN_BENCHMARK = False
# The period (in terms of steps) for minibatch visualization at train time.
# Set to 0 to disable.
_C.VIS_PERIOD = 0

# Set device
_C.DEVICE = "cpu"

# global config is for quick hack purposes.
# You can set them in command line or config files,
# and access it with:
#
# from detectron2.config import global_cfg
# print(global_cfg.HACK)
#
# Do not commit any configs into it.
_C.GLOBAL = CN()
_C.GLOBAL.HACK = 1.0
