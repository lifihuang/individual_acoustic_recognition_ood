import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjFCHead(nn.Module):
    def __init__(
        self,
        latent_dim,
        num_classes,
        num_projection_elements=None,
        use_clf=True,
        return_penult_feat_and_pred=False,
    ):
        super(ProjFCHead, self).__init__()
        self.use_clf = use_clf
        self.return_penult_feat_and_pred = return_penult_feat_and_pred
        if num_projection_elements:
            self.projection = nn.Sequential(
                nn.Linear(latent_dim, num_projection_elements),
                nn.ReLU(),
            )
            self.fc = nn.Linear(num_projection_elements, num_classes)
        else:
            self.projection = None
            self.fc = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        if self.projection:
            x = self.projection(x)
        if self.return_penult_feat_and_pred:
            return x, self.fc(
                x
            )  # returns pretrained embeddings + last layer (# classes)
        if self.use_clf:
            x = self.fc(x)  # returns embeddings if use_clf = False OR last layer
        return x


class AemNet(nn.Module):
    """
    Takes raw audio as input
    """

    def __init__(
        self,
        num_classes,
        num_projection_elements,
        width_mul=0.5,
        dropout=0.2,
        raw_audio_input=True,
        use_clf=True,
        return_penult_feat_and_pred=False,
    ):
        super(AemNet, self).__init__()
        self.num_classes = num_classes
        self.width_mul = width_mul
        # self.dropout = dropout
        self.raw_audio_input = raw_audio_input
        self.num_projection_elements = num_projection_elements
        self.dropout = dropout
        self.use_clf = use_clf
        self.llf, self.hlf, self.proj, self.clf = self._low_and_high_level_features()
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout_layer = nn.Dropout2d(p=dropout)
        self.return_penult_feat_and_pred = return_penult_feat_and_pred

    def sequential(self, x):
        output = x
        if self.raw_audio_input:
            output = self.llf(output)
            output = output.unsqueeze(dim=1)
        output = self.hlf(output)
        output = self.adaptive_pool(output)
        return output

    def _low_and_high_level_features(self):
        llf = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=9, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(80, padding=4),
        )
        width_1 = int(32 * self.width_mul)
        width_2 = int(64 * self.width_mul)
        width_3 = int(128 * self.width_mul)
        width_4 = int(256 * self.width_mul)
        width_5 = int(512 * self.width_mul)
        hlf = nn.Sequential(
            nn.Conv2d(1, width_1, kernel_size=(3, 3), stride=1, padding="same"),
            nn.BatchNorm2d(width_1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(width_1, width_2, kernel_size=(3, 3), stride=1, padding="same"),
            nn.BatchNorm2d(width_2),
            nn.ReLU(),
            nn.Conv2d(width_2, width_2, kernel_size=(3, 3), stride=1, padding="same"),
            nn.BatchNorm2d(width_2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(width_2, width_3, kernel_size=(3, 3), stride=1, padding="same"),
            nn.BatchNorm2d(width_3),
            nn.ReLU(),
            nn.Conv2d(width_3, width_3, kernel_size=(3, 3), stride=1, padding="same"),
            nn.BatchNorm2d(width_3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(width_3, width_4, kernel_size=(3, 3), stride=1, padding="same"),
            nn.BatchNorm2d(width_4),
            nn.ReLU(),
            nn.Conv2d(width_4, width_4, kernel_size=(3, 3), stride=1, padding="same"),
            nn.BatchNorm2d(width_4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(width_4, width_5, kernel_size=(3, 3), stride=1, padding="same"),
            nn.BatchNorm2d(width_5),
            nn.ReLU(),
            nn.Conv2d(width_5, width_5, kernel_size=(3, 3), stride=1, padding="same"),
            nn.BatchNorm2d(width_5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        proj = nn.Sequential(
            nn.Conv2d(
                width_5, self.num_projection_elements, kernel_size=(1, 1), stride=1
            ),
            nn.ReLU(),
        )
        clf = nn.Conv2d(
            self.num_projection_elements, self.num_classes, kernel_size=(1, 1), stride=1
        )

        return llf, hlf, proj, clf

    def forward(self, x):
        output = self.sequential(x)
        output = self.dropout_layer(output)
        output = self.proj(output)
        if self.return_penult_feat_and_pred:
            return output.squeeze(), self.clf(output).squeeze()
        if self.use_clf:
            output = self.clf(output)
        output = output.squeeze()
        return output
