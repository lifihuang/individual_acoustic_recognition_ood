import copy
import os
from time import time

import numpy as np
import torch
import torch.nn as nn

from experiments.experiment import Experiment
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score
from utils.general import checkpoint, EarlyStopper, EntropicOpensetLoss
from utils.metrics import AverageMeter


class BasicClassification(Experiment):
    def __init__(self, config, device):
        super().__init__(config, device)

    def train(self):
        folds = self.config.TRAIN.KFOLDS if self.config.TRAIN.CROSS_VALIDATE else 1
        fold_accs = {}
        for f in range(1, folds + 1):
            (
                data_loader_train,
                data_loader_val,
                model,
                optimizer,
                lr_scheduler,
                criterion,
            ) = self._initialize_train(f)

            start_time = time()
            max_accuracy = 0
            early_stopper = EarlyStopper(
                patience=self.config.TRAIN.EARLY_STOPPER.PATIENCE,
                min_delta=self.config.TRAIN.EARLY_STOPPER.MIN_DELTA,
            )
            for epoch in range(self.config.TRAIN.START_EPOCH, self.config.TRAIN.EPOCHS):
                if self.config.TRAIN.START_EPOCH > 0:
                    # TODO resume from checkpoint
                    pass
                loss_optimizer = None
                self.train_one_epoch(
                    model,
                    criterion,
                    data_loader_train,
                    optimizer,
                    loss_optimizer,
                    epoch,
                )
                lr_scheduler.step()
                if epoch % 10 == 0:
                    target_outputs, outputs, val_loss = self.validate(
                        data_loader_val, model
                    )
                    acc, prec, rec, f1, roc_auc = self.calc_metrics(
                        target_outputs, outputs
                    )
                    if acc > max_accuracy:
                        # save model state
                        dataset_path = (
                            ""
                            if self.config.DATA.DATASET_PATH is None
                            else self.config.DATA.DATASET_PATH
                        )
                        best_model_name = [
                            self.config.MODEL.TYPE,
                            self.config.MODEL.NAME,
                            self.config.DATA.DATASET,
                            dataset_path,
                            f"best_model_fold{f}.pt",
                        ]
                        if self.config.DATA.CLASS is not None:
                            best_model_name.insert(-1, self.config.DATA.CLASS)
                        best_model_name = "--".join(best_model_name)
                        path = os.path.join(
                            self.config.GENERAL.CHECKPOINTS_DIR, best_model_name
                        )
                        self.logger.info(
                            f"Saving best model at epoch {epoch} as {best_model_name}"
                        )
                    max_accuracy = max(max_accuracy, acc)
                    self.writer.add_scalar("Loss/validation", round(val_loss, 4), epoch)
                    self.writer.add_scalar("Accuracy/validation", round(acc, 3), epoch)
                    self.writer.add_scalar(
                        "Precision/validation", round(prec, 3), epoch
                    )
                    self.writer.add_scalar("Recall/validation", round(rec, 3), epoch)
                    self.writer.add_scalar("F1/validation", round(f1, 3), epoch)
                    self.writer.add_scalar(
                        "ROC_AUC/validation", round(roc_auc, 3), epoch
                    )
                    self.logger.info(
                        f"Epoch [{epoch}/{self.config.TRAIN.EPOCHS}]\t"
                        f"Validation on {len(data_loader_val.dataset)} test recordings:\t"
                        f"* Acc@1 ({acc:.3f})\t"
                        f"* Prec ({prec:.3f})\t"
                        f"* Rec ({rec:.3f})\t"
                        f"* F1 ({f1:.3f})\t"
                        f"* ROC_AUC ({roc_auc:.3f})\t"
                        f"* Loss ({val_loss:.4f})\t"
                        f"* MaxAcc@1 ({max_accuracy*100:.3f}%)"
                    )
                    if early_stopper is not None and early_stopper.early_stop(val_loss):
                        self.logger.info(
                            f"Training stopped after {epoch} epochs due to early stopping condition"
                        )
                        break
            fold_accs[str(f)] = max_accuracy * 100
            fold_time = time() - start_time
            checkpoint(
                model,
                optimizer,
                data_loader_train.dataset.dataset.class_to_idx,
                path,
                self.device,
            )
            self.logger.info(f"Training time for fold {f}: {fold_time:.2f}s")
            self.logger.info(f"MaxAcc@1 by fold: {fold_accs}")

        self.logger.info(fold_accs)
        self.logger.info(
            f"######## Average MaxAcc@1 across {folds} folds: {np.mean(list(fold_accs.values()))} ########"
        )
        self.logger.info(
            f"######## Average MaxAcc@1-Std across {folds} folds: {np.std(list(fold_accs.values()))} ########"
        )
        self.logger.info("################## Stop training ###################\n")
        self.writer.close()

    def train_one_epoch(
        self, model, criterion, data_loader, optimizer, loss_optimizer, epoch
    ):
        model.train()
        loss_meter = AverageMeter()

        if epoch == 0:
            self.logger.debug(f"Moving features and targets to {self.device}")

        outputs = []
        target_outputs = []
        for features, targets in data_loader:
            if isinstance(features, list):
                features = torch.nested.nested_tensor(features)
            features, targets = features.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            output = model(features)
            criterion_output = output
            if output.dim() == 1:
                output = output.unsqueeze(0)
            loss = criterion(criterion_output, targets)
            loss.backward()
            optimizer.step()
            if loss_optimizer is not None:
                loss_optimizer.step()
            loss_meter.update(loss.item(), targets.size(0))
            outputs.append(output.detach().cpu())
            target_outputs.append(targets.detach().cpu())

        outputs = torch.cat(outputs, dim=0)
        target_outputs = torch.cat(target_outputs, dim=0)
        acc, prec, rec, f1, roc_auc = self.calc_metrics(target_outputs, outputs)

        self.writer.add_scalar("Loss/train", round(loss_meter.avg, 4), epoch)
        self.writer.add_scalar("Accuracy/train", round(acc, 3), epoch)
        self.writer.add_scalar("Precision/train", round(prec, 3), epoch)
        self.writer.add_scalar("Recall/train", round(rec, 3), epoch)
        self.writer.add_scalar("F1/train", round(f1, 3), epoch)
        self.writer.add_scalar("ROC_AUC/train", round(roc_auc, 3), epoch)

        self.logger.info(
            f"Epoch [{epoch}/{self.config.TRAIN.EPOCHS}]\t"
            f"* Acc@1 ({acc:.3f})\t"
            f"* Prec ({prec:.3f})\t"
            f"* Rec ({rec:.3f})\t"
            f"* F1 ({f1:.3f})\t"
            f"* ROC_AUC ({roc_auc:.3f})\t"
            f"* Loss ({loss_meter.avg:.4f})"
        )

    @torch.no_grad()
    def validate(self, data_loader, model):
        if self.config.TRAIN.CRITERION == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss()
        elif self.config.TRAIN.CRITERION == "EntropicOpensetLoss":
            criterion = EntropicOpensetLoss(self.config.DATA.NUM_CLASSES, self.device)
        else:
            raise NotImplementedError("No other criterions currently implemented")
        model.eval()

        loss_meter = AverageMeter()

        outputs = []
        target_outputs = []
        with torch.no_grad():
            for features, targets in data_loader:
                if isinstance(features, list):
                    features = torch.nested.nested_tensor(features)
                features, targets = features.to(self.device), targets.to(self.device)
                output = model(features)
                criterion_output = output
                if output.dim() == 1:
                    output = output.unsqueeze(0)
                loss = criterion(criterion_output, targets)
                loss_meter.update(loss.item(), targets.size(0))
                outputs.append(output.detach().cpu())
                target_outputs.append(targets.detach().cpu())

        outputs = torch.cat(outputs, dim=0)
        target_outputs = torch.cat(target_outputs, dim=0)
        return target_outputs, outputs, loss_meter.avg

    def calc_metrics(self, target_outputs, outputs):
        _, predicted = torch.max(outputs, 1)
        acc = accuracy_score(target_outputs, predicted)
        prec = precision_score(
            target_outputs, predicted, average="weighted", zero_division=0
        )
        rec = precision_score(
            target_outputs, predicted, average="weighted", zero_division=0
        )
        f1 = f1_score(target_outputs, predicted, average="weighted", zero_division=0)
        if -1 in target_outputs:
            roc_auc = 0
        else:
            roc_auc = roc_auc_score(
                target_outputs,
                nn.Softmax(dim=1)(outputs),
                average="weighted",
                multi_class="ovr",
            )

        return acc, prec, rec, f1, roc_auc

    def test(self):
        pass

    def embed(self, model, model_path, data_loader):
        embed_model = copy.deepcopy(model)
        checkpoint = torch.load(model_path, map_location="cpu")["model"]
        embed_model.load_state_dict(checkpoint)
        embed_model.use_clf = False
        data = []
        for features, targets in data_loader:
            features = features.to(self.device)
            e = embed_model(features).detach().to("cpu")
            try:
                combined = torch.cat((e, targets.unsqueeze(1)), dim=1)
            except:
                print("")
            data.append(combined)
        data = torch.cat(data, dim=0)
        return data
