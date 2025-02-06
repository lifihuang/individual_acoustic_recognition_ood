from functools import partial

import audiomentations
import librosa
import numpy as np
import pywt
import torch
import torch.nn.functional as F
import torchaudio
from torchopenl3 import get_audio_embedding
from torchopenl3.models import load_audio_embedding_model
from torchvision import transforms


class RandomApply(torch.nn.Module):
    """
    Source: https://github.com/Spijkervet/torchaudio-augmentations/blob/master/torchaudio_augmentations/apply.py
    """

    def __init__(self, transform, p=0.5):
        super().__init__()
        self.transform = transform
        self.p = p

    def forward(self, img):
        if self.p < torch.rand(1):
            return img
        img = self.transform(img)
        return img


class ComposeViews:
    """
    Inspired by: https://github.com/Spijkervet/torchaudio-augmentations/blob/master/torchaudio_augmentations/compose.py
    """

    def __init__(self, transforms, num_views=2, include_raw_audio=True):
        self.transforms = transforms
        self.num_views = num_views
        self.include_raw_audio = include_raw_audio
        if include_raw_audio:
            assert len(self.transforms) == self.num_views - 1
        else:
            assert len(self.transforms) == self.num_views

    def __call__(self, x):
        # this is so hacky
        samples = [np.expand_dims(x, 0)] if self.include_raw_audio else []
        while len(samples) < self.num_views:
            samples.append(np.expand_dims(self.transform(x).flatten(), 0))
        samples = np.concatenate(samples, 1)
        return samples

    def transform(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class RawAudioAugments:
    def __init__(self, config):
        self.raw_audio_aug = self._add_raw_audio_augmentations(config)

    @staticmethod
    def _add_raw_audio_augmentations(config):
        """
        Raw audio augmentations excluding mixup
        :param config:
        :return:
        """
        raw_audio_aug = []
        for aug in config.TRAIN.AUGS.METHODS:
            if aug == "time-masking":
                raw_audio_aug.append(
                    audiomentations.TimeMask(
                        min_band_part=config.TRAIN.AUGS.ARGS.TM_MIN_BAND_PART,
                        max_band_part=config.TRAIN.AUGS.ARGS.TM_MAX_BAND_PART,
                        fade=config.TRAIN.AUGS.ARGS.TM_FADE,
                        p=config.TRAIN.AUGS.ARGS.TM_P,
                    )
                )
            if aug == "gain":
                raw_audio_aug.append(
                    audiomentations.Gain(
                        min_gain_db=config.TRAIN.AUGS.ARGS.GAIN_MIN,
                        max_gain_db=config.TRAIN.AUGS.ARGS.GAIN_MAX,
                        p=config.TRAIN.AUGS.ARGS.GAIN_P,
                    )
                )

        return raw_audio_aug


class AudioTransforms:
    """
    Performs transformations on raw audio and converts numpy arrays to tensors
    before performing transformed augmentations.
    """

    def __init__(self, config):
        self.input_transforms = self._add_input_transforms(config)
        self.transform_augments = self._add_transform_augments(config)

    def _add_input_transforms(self, config):
        input_transforms = []
        if isinstance(config.INPUT_TRANSFORM.METHOD, str):
            all_transforms = [config.INPUT_TRANSFORM.METHOD]
        else:
            all_transforms = config.INPUT_TRANSFORM.METHOD
        for transform in all_transforms:
            if transform == "mel_spec":
                input_transforms.append(
                    self.mel_spec(
                        sr=config.DATA.SAMPLING_RATE,
                        hop_length=config.INPUT_TRANSFORM.ARGS.HOP_LENGTH,
                        n_fft=config.INPUT_TRANSFORM.ARGS.N_FFT,
                        n_mels=config.INPUT_TRANSFORM.ARGS.N_MELS,
                        fmin=config.INPUT_TRANSFORM.ARGS.FMIN,
                        fmax=config.INPUT_TRANSFORM.ARGS.FMAX,
                        power=config.INPUT_TRANSFORM.ARGS.POWER,
                    )
                )
            elif transform == "wavelet_scaleogram":
                input_transforms.append(
                    self.wavelet_scaleogram(
                        min_scales=config.INPUT_TRANSFORM.ARGS.MIN_SCALES,
                        max_scales=config.INPUT_TRANSFORM.ARGS.MAX_SCALES,
                        num_scales=config.INPUT_TRANSFORM.ARGS.NUM_SCALES,
                        wavelet=config.INPUT_TRANSFORM.ARGS.WAVELET,
                        include_raw_audio=config.INPUT_TRANSFORM.ARGS.INCLUDE_RAW_AUDIO,
                    ),
                )
            elif transform == "multilevel-dwt":
                input_transforms.append(
                    self.multilevel_dwt(
                        wavelet=config.INPUT_TRANSFORM.ARGS.WAVELET,
                        mode="sym",
                        level=config.INPUT_TRANSFORM.ARGS.LEVELS,
                    )
                )
            elif transform == "contrastive":
                input_transforms.append(
                    ComposeViews(
                        [
                            self.mel_spec(
                                sr=config.DATA.SAMPLING_RATE,
                                hop_length=config.INPUT_TRANSFORM.ARGS.HOP_LENGTH,
                                n_fft=config.INPUT_TRANSFORM.ARGS.N_FFT,
                                n_mels=config.INPUT_TRANSFORM.ARGS.N_MELS,
                                fmin=config.INPUT_TRANSFORM.ARGS.FMIN,
                                fmax=config.INPUT_TRANSFORM.ARGS.FMAX,
                                power=config.INPUT_TRANSFORM.ARGS.POWER,
                                as_tensor=False,
                            )
                        ]
                    )
                )
            elif transform == "open_l3":
                input_transforms.append(
                    partial(
                        self._return_only_xth_element(get_audio_embedding, 0),
                        sr=config.DATA.SAMPLING_RATE,
                        model=AudioTransforms.load_openl3_model(config),
                        # center=config.INPUT_TRANSFORM.OPENL3.CENTER,
                        # hop_size=config.INPUT_TRANSFORM.OPENL3.HOP_SIZE,
                        verbose=1,
                    )
                )
        return input_transforms

    def _add_transform_augments(self, config):
        transform_augments = []
        for aug in config.TRAIN.AUGS.METHODS:
            if aug == "freq-masking":
                transform_augments.append(
                    RandomApply(
                        FrequencyMasking(
                            n_mels=config.INPUT_TRANSFORM.ARGS.N_MELS,
                            min_mask_fraction=config.TRAIN.AUGS.ARGS.FM_MIN_MASK_FRAC,
                            max_mask_fraction=config.TRAIN.AUGS.ARGS.FM_MAX_MASK_FRAC,
                        ),
                        p=config.TRAIN.AUGS.ARGS.FM_P,
                    )
                )
            # TODO add time-warp?
        return transform_augments

    @staticmethod
    def mel_spec(
        sr, hop_length, n_fft, n_mels, fmax, fmin, power, normalize=True, as_tensor=True
    ):
        def melspec_in_db(sig):
            # Librosa mel-spectrum
            melspec = librosa.feature.melspectrogram(
                y=sig,
                sr=sr,
                hop_length=hop_length,
                n_fft=n_fft,
                n_mels=n_mels,
                fmax=fmax,
                fmin=fmin,
                power=power,
            )

            # Convert power spec to dB scale (compute dB relative to peak power)
            melspec = librosa.power_to_db(melspec, ref=np.max, top_db=80)

            # Normalize values between 0 and 1
            if normalize:
                melspec -= melspec.min()
                if not melspec.max() == 0:
                    melspec /= melspec.max()
                else:
                    melspec = np.clip(melspec, 0, 1)
            if as_tensor:
                melspec = torch.from_numpy(melspec.astype("float32")).unsqueeze(0)
            return melspec

        return melspec_in_db

    @staticmethod
    def load_openl3_model(config):
        model = load_audio_embedding_model(
            content_type=config.INPUT_TRANSFORM.OPENL3.CONTENT_TYPE,
            input_repr=config.INPUT_TRANSFORM.OPENL3.INPUT_REPR,
            embedding_size=config.INPUT_TRANSFORM.OPENL3.EMBEDDING_SIZE,
        )
        return model

    @staticmethod
    def wavelet_scaleogram(
        min_scales, max_scales, num_scales, wavelet, include_raw_audio=False
    ):
        def wavelet_scaleogram_as_tensor(sig):
            scales = np.linspace(min_scales, max_scales, num_scales)
            scaleogram, _ = pywt.cwt(sig, scales=scales, wavelet=wavelet)

            ####
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(figsize=(10, 10))
            # ax.imshow(scaleogram[:,:400], cmap="RdGy")
            # # ax.set_xlim(0,3000)
            # plt.show()
            ####
            if include_raw_audio:
                scaleogram = np.vstack([sig, scaleogram])
            return torch.from_numpy(scaleogram).unsqueeze(0)

        return wavelet_scaleogram_as_tensor

    @staticmethod
    def multilevel_dwt(wavelet, mode, level, return_all_coeff=True):
        if not return_all_coeff:
            return partial(pywt.dwt, wavelet=wavelet, mode=mode, level=level)
        else:

            def ml_dwt(data):
                coeffs = []
                for i in range(level):
                    cA, cD = pywt.dwt(data, wavelet=wavelet)
                    coeffs.extend([torch.tensor(cA), torch.tensor(cD)])
                    data = cA
                return coeffs

            return ml_dwt

    @staticmethod
    def _return_only_xth_element(func, x):
        """
        Wrapper to use when a function returns multiple arguments, but only one is needed
        """

        def wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            return res[x]

        return wrapper


class FrequencyMasking(torch.nn.Module):
    def __init__(self, n_mels, min_mask_fraction, max_mask_fraction):
        super(FrequencyMasking, self).__init__()
        self.min_mask = n_mels * min_mask_fraction
        self.max_mask = n_mels * max_mask_fraction

    def forward(self, specgram):
        freq_mask_param = torch.FloatTensor(1).uniform_(self.min_mask, self.max_mask)
        return torchaudio.transforms.FrequencyMasking(freq_mask_param)(specgram)


def build_raw_audio_augment(config):

    R = RawAudioAugments(config)
    return audiomentations.Compose(R.raw_audio_aug)  # audiomentations Compose


def build_input_transform(config, dataset_type):
    # TODO config parameters should be controlling how the data is transformed (both
    #  conversion to Mel Specs and augmentation)
    if config.INPUT_TRANSFORM.METHOD is None:
        return None
    A = AudioTransforms(config)

    # augmentation = None
    if config.INPUT_TRANSFORM.METHOD == "openl3":
        assert (
            config.DATA.MAX_AUDIO_LENGTH and config.DATA.USE_SILENCE_PADDING
        ), "openl3 requires inputs of the same length"
    # if config.INPUT_TRANSFORM.OPENL3.CENTER:
    #     print("torch port of openl3 does not seem to work with center")
    # if config.INPUT_TRANSFORM.OPENL3.HOP_SIZE:
    #     print("torch port of openl3 does not seem to work with hop_size")
    t = A.input_transforms.copy()
    if dataset_type == "train" and config.GENERAL.LEARN_METHOD == "supervised":
        t.extend(A.transform_augments)

    return transforms.Compose(t)


def idx_to_one_hot_float(idx, num_classes):
    one_hot = F.one_hot(torch.tensor(idx), num_classes)
    return one_hot.float()


def build_target_transform(config):
    """
    Currently does not work for batched targets
    :param config:
    :return:
    """
    if config.TARGET_TRANSFORM.METHOD == "one_hot_encoding":
        return partial(idx_to_one_hot_float, num_classes=config.DATA.NUM_CLASSES)
        # return partial(
        #         _return_only_xth_element(get_audio_embedding, 0),
        #         sr=config.DATA.SAMPLING_RATE,
        #     )


def batch_mixup(x, y, alpha=5, beta=2):
    lam = np.random.beta(alpha, beta)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    x = lam * x + (1 - lam) * x[index, :]

    # doesn't allow for multi-target
    if lam < 0.5:
        y = y[index]
    return x, y


def dwt_mixup(x: list, y: torch.Tensor, alpha=5, beta=2):
    lam = np.random.beta(alpha, beta)
    batch_size = y.size(0)
    index = torch.randperm(batch_size)
    for i in range(len(x)):
        x[i] = lam * x[i] + (1 - lam) * x[i][index, :]

    # doesn't allow for multi-target
    if lam < 0.5:
        y = y[index]
    return x, y
