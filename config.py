import os
import yaml
from yacs.config import CfgNode as CN


# Monkey patching CNs so config params can be added that are not included in this file
initial_init = CN.__init__
def new_allowed_init(self, init_dict=None, key_list=None, new_allowed=True):
    initial_init(self, init_dict, key_list, new_allowed)
CN.__init__ = new_allowed_init

_C = CN()
# -----------------------------------------------------------------------------
# General settings
# -----------------------------------------------------------------------------
_C.GENERAL = CN()
_C.GENERAL.LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
_C.GENERAL.CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
_C.GENERAL.SAVED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
_C.GENERAL.EXP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")
_C.GENERAL.FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
_C.GENERAL.SEED = 88
_C.GENERAL.MODE = "debug"
_C.GENERAL.LEARN_METHOD = "supervised"
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type, in CamelCase, e.g.OpenL3FCHead
_C.MODEL.TYPE = ""  # This select the correct model/architecture to use
# Model name, in Snake_Case, e.g. OpenL3_FC_Head_test
_C.MODEL.NAME = ""  # This specifies the exact model config
# Model-specific parameters to fill in through the configs
_C.MODEL.PARAMS = CN()
# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.TRAIN_VAL_SPLIT = 0.8  # 0.8 means 80% training, 20% eval
# # Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = None
# Ensure each class appears exactly once per batch, will ignore batch size
_C.DATA.ONE_CLASS_PER_BATCH = False
# # Path to all datasets, could be overwritten by command line argument
_C.DATA.DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
# Specific dataset name
_C.DATA.DATASET = None  # ["littleowl", "pipit", "chiffchaff"]
# Specfic dataset relative path
_C.DATA.DATASET_PATH = None
# Dataset evaluation scenario
_C.DATA.SCENARIO = None  # "acrossyear"  # ["acrossyear", "withinyear"]
# Classes to drop from dataset
_C.DATA.CLASSES_TO_DROP = []
# Class to separate by (used e.g. if multiple features to classify by)
_C.DATA.CLASS = None
# Number of classes determined on the fly, included here just for completeness, don't touch
_C.DATA.NUM_CLASSES: None
# Dataset channel count  <-- is this needed?
_C.DATA.CHANNEL_COUNT = 1
# Dataset sampling rate in Hz
_C.DATA.SAMPLING_RATE = 44100  # littleowl: 44100
# The threshold (in decibels) below reference to consider as silence
_C.DATA.TOP_DB = 60  # not used
# Dataset maximum audio length in s (dataset-dependent), use None to allow variable input size
_C.DATA.MAX_AUDIO_LENGTH = 2.0  # littleowl: 2s @ 44100 Hz
# Pad audio samples to maximum audio length with silence
_C.DATA.USE_SILENCE_PADDING = None  # True if _C.DATA.MAX_AUDIO_LENGTH else None
# Signal-to-noise ratio between foreground and background vocalizations in dB
_C.DATA.SIGNAL_TO_NOISE_RATIO = 100 # not used
# If True, ensures perfectly balanced dataset
_C.DATA.USE_BALANCED_DATASET = False
# if !=0 and use_balanced_dataset is False: samples up to max_count_per_cls for each cls
_C.DATA.MAX_COUNT_PER_CLS = 0

# -----------------------------------------------------------------------------
# Input Transformation settings
# -----------------------------------------------------------------------------
_C.INPUT_TRANSFORM = CN()
_C.INPUT_TRANSFORM.METHOD = None
# -----------------------------------------------------------------------------
# Input Transformation settings - Mel Spectrogram
# -----------------------------------------------------------------------------
# _C.INPUT_TRANSFORM.ARGS = CN()
# _C.INPUT_TRANSFORM.ARGS.N_FFT = 1024
# _C.INPUT_TRANSFORM.ARGS.HOP_LENGTH = 512
# _C.INPUT_TRANSFORM.ARGS.N_MELS = 40
# _C.INPUT_TRANSFORM.ARGS.FMIN = 500
# _C.INPUT_TRANSFORM.ARGS.FMAX = 15000
# _C.INPUT_TRANSFORM.ARGS.POWER = 1.0
# # -----------------------------------------------------------------------------
# # Input Transformation settings - Pre-trained Embedding
# # -----------------------------------------------------------------------------
# _C.INPUT_TRANSFORM.OPENL3 = CN()
# _C.INPUT_TRANSFORM.OPENL3.CONTENT_TYPE = "music"  # ("env", "music")
# _C.INPUT_TRANSFORM.OPENL3.INPUT_REPR = "mel128"  # ("linear", "mel128", "mel256")
# _C.INPUT_TRANSFORM.OPENL3.EMBEDDING_SIZE = 6144  # (512, 6144)
# torchopenl3 does not seem to work for this?
# _C.INPUT_TRANSFORM.OPENL3.CENTER = True
# _C.INPUT_TRANSFORM.OPENL3.HOP_SIZE = 2.5

# -----------------------------------------------------------------------------
# Target Transformation settings
# -----------------------------------------------------------------------------
_C.TARGET_TRANSFORM = CN()
_C.TARGET_TRANSFORM.METHOD = None


# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.CROSS_VALIDATE = True
_C.TRAIN.KFOLDS = 5

_C.TRAIN.ARGS = CN()
_C.TRAIN.ARGS.BASE_LR = 0.0001  # 5e-4
_C.TRAIN.ARGS.EPS = 1e-8
_C.TRAIN.ARGS.BETAS = (0.9, 0.999)
_C.TRAIN.ARGS.MOMENTUM = 0.9
_C.TRAIN.ARGS.WEIGHT_DECAY = 0.01
_C.TRAIN.ARGS.USE_X_PERC_DATA = 1.0

_C.TRAIN.AUGS = CN()
_C.TRAIN.AUGS.METHODS = []


# Optimizer
_C.TRAIN.OPTIMIZER = "adamw"

_C.TRAIN.EARLY_STOPPER = CN()
_C.TRAIN.EARLY_STOPPER.PATIENCE = 0  # 0 means it's turned off
_C.TRAIN.EARLY_STOPPER.MIN_DELTA = 0.0

# Criterion
_C.TRAIN.CRITERION = None

# logger
_C.LOGGER = CN()
_C.LOGGER.SAVE_TO_FILE = True


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    # print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    # if args.opts:
    #     config.merge_from_list(args.opts)
    #
    # # merge from specific arguments
    # if args.batch_size:
    #     config.DATA.BATCH_SIZE = args.batch_size
    # if args.data_path:
    #     config.DATA.DATA_PATH = args.data_path
    # if args.zip:
    #     config.DATA.ZIP_MODE = True
    # if args.cache_mode:
    #     config.DATA.CACHE_MODE = args.cache_mode
    # if args.pretrained:
    #     config.MODEL.PRETRAINED = args.pretrained
    # if args.resume:
    #     config.MODEL.RESUME = args.resume
    # if args.accumulation_steps:
    #     config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    # if args.use_checkpoint:
    #     config.TRAIN.USE_CHECKPOINT = True
    # if args.amp_opt_level:
    #     config.AMP_OPT_LEVEL = args.amp_opt_level
    # if args.output:
    #     config.OUTPUT = args.output
    # if args.tag:
    #     config.TAG = args.tag
    # if args.eval:
    #     config.EVAL_MODE = True
    # if args.throughput:
    #     config.THROUGHPUT_MODE = True
    #
    # # set local rank for distributed training
    # config.LOCAL_RANK = args.local_rank
    #
    # # output folder
    # config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config