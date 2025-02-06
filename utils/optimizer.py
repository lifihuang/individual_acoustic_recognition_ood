import warnings

from simclr.modules import LARS
from torch import optim as optim


def build_optimizer(config, model):
    parameters = model.parameters()

    opt_lower = config.TRAIN.OPTIMIZER.lower()
    optimizer = None
    if opt_lower == "sgd":
        optimizer = optim.SGD(
            parameters,
            momentum=config.TRAIN.ARGS.MOMENTUM,
            nesterov=True,
            lr=config.ARGS.BASE_LR,
            weight_decay=config.ARGS.WEIGHT_DECAY,
        )
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(
            parameters,
            eps=config.TRAIN.ARGS.EPS,
            betas=config.TRAIN.ARGS.BETAS,
            lr=config.TRAIN.ARGS.BASE_LR,
            weight_decay=config.TRAIN.ARGS.WEIGHT_DECAY,
        )
    elif opt_lower == "lars":
        # taken from https://github.com/Spijkervet/CLMR/blob/master/clmr/modules/contrastive_learning.py
        # optimized using LARS with linear learning rate scaling
        # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
        learning_rate = 0.3 * config.DATA.BATCH_SIZE / 256
        optimizer = LARS(
            model.parameters(),
            lr=learning_rate,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
            exclude_from_weight_decay=["batch_normalization", "bias"],
        )

    return optimizer


def build_scheduler(config, optimizer):
    sched_lower = config.TRAIN.SCHEDULER.lower()
    scheduler = None
    if sched_lower == "cosineannealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.TRAIN.EPOCHS)
    return scheduler
