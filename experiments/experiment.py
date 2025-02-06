import os
from abc import ABC, abstractmethod

from data.build import build_loader
from logger import create_logger
import numpy as np
import torch
import torch.nn as nn

from config import get_config
from models.build import build_model
from torch.utils.tensorboard import SummaryWriter
from utils.general import EntropicOpensetLoss
from utils.optimizer import build_optimizer, build_scheduler


class Experiment(ABC):
    def __init__(self, config, device):
        self.parallel = isinstance(device, list)
        self.config = config
        self.logger = create_logger(
            config.GENERAL.LOGS_DIR,
            dataset=config.DATA.DATASET,
            save_to_file=config.LOGGER.SAVE_TO_FILE,
            model_type=config.MODEL.TYPE,
            model_name=config.MODEL.NAME,
        )
        if config.DATA.DATASET_PATH:
            run_dir = f"runs/{config.DATA.DATASET}/{config.DATA.DATASET_PATH}/{config.MODEL.TYPE}/{config.MODEL.NAME}"
        else:
            run_dir = (
                f"runs/{config.DATA.DATASET}/{config.MODEL.TYPE}/{config.MODEL.NAME}"
            )
        self.writer = SummaryWriter(run_dir)

        if self.parallel:
            self.device = torch.device(
                f"cuda:{device[0]}" if torch.cuda.is_available() else "cpu"
            )
            self.all_devices = device
            self.logger.info("Running experiment on multiple gpus!")
        else:
            self.device = device
            self.all_devices = [device]
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                [str(dev.index if self.parallel else dev) for dev in self.all_devices]
            )

        ########## seed setting ##########
        torch.manual_seed(self.config.GENERAL.SEED)
        torch.cuda.manual_seed(self.config.GENERAL.SEED)
        np.random.seed(self.config.GENERAL.SEED)
        # random.seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(self.config.GENERAL.SEED)
        rng = np.random.RandomState(self.config.GENERAL.SEED)

        if self.config.GENERAL.MODE == "train":
            self.logger.info(
                "################### Config parameters ################### \n"
                + self.config.dump()
            )

    def _load_ckpt(self):
        # TODO
        pass

    def _initialize_train(self, fold):
        data_loader_train, data_loader_val = build_loader(
            self.config, fold, self.device, self.logger
        )

        self.logger.info(
            f"Class to index: {data_loader_train.dataset.dataset.class_to_idx}"
        )
        self.logger.info(
            f"################### Start training: Fold {fold} ###################\n"
        )
        self.logger.info(
            f"Creating model: {self.config.MODEL.TYPE} with {self.config.MODEL.NAME} specs"
        )
        model = build_model(self.config)
        if "with-negs" in self.config.MODEL.NAME:
            assert self.config.TRAIN.CRITERION != "CrossEntropyLoss"
        if fold == 1:
            self.logger.info(str(model))
        self.logger.debug(f"Moving model to {self.device}")
        model.to(self.device)

        optimizer = build_optimizer(self.config, model)

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Number of params: {n_parameters}")

        lr_scheduler = build_scheduler(self.config, optimizer)

        if self.config.TRAIN.CRITERION == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss()
        elif self.config.TRAIN.CRITERION == "EntropicOpensetLoss":
            criterion = EntropicOpensetLoss(self.config.DATA.NUM_CLASSES, self.device)

        else:
            raise NotImplementedError("No other criterions currently implemented")
        return (
            data_loader_train,
            data_loader_val,
            model,
            optimizer,
            lr_scheduler,
            criterion,
        )

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass
