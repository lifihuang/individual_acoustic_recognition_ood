import numpy as np
import torch


def checkpoint(model, optimizer, class_to_idx, path, device):
    torch.save(
        {
            "model": model.to(device).state_dict(),
            "optimizer": optimizer.state_dict(),
            "class_to_idx": class_to_idx,
        },
        path,
    )


class EarlyStopper:
    """
    https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    """

    def __init__(self, patience=0, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def __bool__(self):
        return self.patience == 0

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class EntropicOpensetLoss:
    """Taken from https://github.com/AIML-IfI/openset-imagenet/"""

    def __init__(self, num_of_classes, device, unk_weight=1):
        self.device = device
        self.class_count = num_of_classes
        self.eye = torch.eye(self.class_count).to(device)
        self.unknowns_multiplier = unk_weight / self.class_count
        self.ones = (torch.ones(self.class_count) * self.unknowns_multiplier).to(device)
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def __call__(self, logits, target):
        categorical_targets = torch.zeros(logits.shape).to(self.device)
        unk_idx = target < 0
        kn_idx = ~unk_idx
        # check if there is known samples in the batch
        if torch.any(kn_idx):
            categorical_targets[kn_idx, :] = self.eye[target[kn_idx]]

        categorical_targets[unk_idx, :] = self.ones.expand(
            torch.sum(unk_idx).item(), self.class_count
        )
        return self.cross_entropy(logits, categorical_targets)
