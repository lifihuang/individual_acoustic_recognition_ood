import logging
import os
import random
import warnings
from glob import glob

import librosa
import numpy as np
import pandas as pd
import sklearn
import torch
from torch.utils.data import Dataset

from utils.transforms import (
    build_raw_audio_augment,
    build_input_transform,
    build_target_transform,
)

logging.getLogger("numba").setLevel(logging.WARNING)


class AudioDataset(Dataset):
    def __init__(
        self,
        dataset_type,
        config,
        use_mixing,
        device,
        use_negative_samples,
        fold,
        use_subset=True,
    ):
        self.dataset_type = dataset_type
        self.path_to_dataset = os.path.join(config.DATA.DATA_PATH, config.DATA.DATASET)
        self.use_negative_samples = use_negative_samples
        self.fold = fold
        if config.DATA.DATASET_PATH:
            self.path_to_dataset = os.path.join(
                self.path_to_dataset, config.DATA.DATASET_PATH
            )
        self.samples = self._make_samples(dataset_type, config, use_mixing, use_subset)
        classes, class_to_idx = self._find_classes(config)
        self.classes = classes
        self.class_to_idx = class_to_idx
        if "with-negs" in config.MODEL.NAME and dataset_type == "train":
            assert -1 in self.class_to_idx.values()
        self.data_path = config.DATA.DATA_PATH
        self.device = device
        self.use_silence_padding = config.DATA.USE_SILENCE_PADDING
        self.sampling_rate = config.DATA.SAMPLING_RATE
        self.max_audio_length = config.DATA.MAX_AUDIO_LENGTH
        self.signal_to_noise_ratio = config.DATA.SIGNAL_TO_NOISE_RATIO
        self.top_db = config.DATA.TOP_DB
        self._add_to_config(config)
        self.mixup = True if "mixup" in config.TRAIN.AUGS.METHODS else False
        if self.mixup:
            self.mixup_alpha = config.TRAIN.AUGS.ARGS.MIXUP_ALPHA
            self.mixup_beta = config.TRAIN.AUGS.ARGS.MIXUP_BETA
            self.mixup_tracker = 0
            self.mixup_start = (config.TRAIN.AUGS.ARGS.MIXUP_START_EPOCH - 1) * len(
                self
            )
        self.raw_audio_augment = build_raw_audio_augment(config)
        self.transform = build_input_transform(config, self.dataset_type)
        self.target_transform = build_target_transform(config)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # if idx is a list (e.g. when manually making batches for OneClassPerBatch or using Mixup)
        if isinstance(idx, list):
            audio_sigs = None
            target_classes = None
            for i in idx:
                rel_audio_path, target_class, *_ = self.samples.iloc[i]
                target_class = self.class_to_idx[target_class]
                if target_class != -1:
                    audio_path = os.path.join(self.path_to_dataset, rel_audio_path)
                else:
                    audio_path = rel_audio_path

                audio_sig, audio_sr = self.load_audio_sample(
                    path=audio_path,
                    use_silence_padding=self.use_silence_padding,
                    sampling_rate=self.sampling_rate,
                    top_db=self.top_db,
                    max_audio_length=self.max_audio_length,
                )
                audio_sig = np.expand_dims(audio_sig, 0)
                target_class = np.expand_dims(target_class, 0)
                if audio_sigs is None:
                    audio_sigs = audio_sig
                    target_classes = target_class
                else:
                    audio_sigs = np.concatenate([audio_sigs, audio_sig], axis=0)
                    target_classes = np.concatenate(
                        [target_classes, target_class], axis=0
                    )
            #
            if self.dataset_type == "train":
                if self.mixup:
                    if self.mixup_tracker >= self.mixup_start:
                        audio_sigs, target_classes = self.raw_audio_mixup(
                            audio_sigs,
                            target_classes,
                            self.mixup_alpha,
                            self.mixup_beta,
                            batched=True,
                        )
                    else:
                        # TODO doesnt track properly because if manually batched
                        self.mixup_tracker += audio_sigs.shape[0]
                for i in range(len(audio_sigs)):
                    audio_sigs[i] = self.raw_audio_augment(
                        audio_sigs[i], self.sampling_rate
                    )
            if self.transform is None:
                audio_sigs = torch.from_numpy(audio_sigs).unsqueeze(1)
            elif self.is_dwt:
                audio_sigs = self.transform(audio_sigs)
            else:
                audio_sigs = np.apply_along_axis(self.transform, 1, audio_sigs)
            if self.target_transform is None:
                target_classes = torch.from_numpy(target_classes)
            else:
                target_classes = np.apply_along_axis(
                    self.target_transform, 1, target_classes
                )

            return audio_sigs, target_classes

        else:
            raise NotImplementedError("this shouldn't reach")

    @staticmethod
    def load_audio_sample(
        path,
        use_silence_padding,
        sampling_rate,
        top_db,
        max_audio_length=True,
        as_mono=True,
        mean_subtract=True,
        normalize_std=True,
    ):
        sig, sr = librosa.load(path, sr=sampling_rate, mono=as_mono)
        sig, _ = librosa.effects.trim(sig, top_db=top_db)
        sig_length = sig.size
        if mean_subtract:
            sig -= sig.mean()
        if normalize_std:
            sig /= sig.std()
        # pad and trim
        if max_audio_length:
            max_frames = int(sr * max_audio_length)
            front = random.randint(0, abs(max_frames - sig_length))
            if sig_length > max_frames:
                sig = sig[front : front + max_frames]
            elif use_silence_padding:
                back_padding = max_frames - sig_length - front
                sig = np.pad(sig, (front, back_padding))
        return sig, sr

    def _add_to_config(self, config):
        config.defrost()
        config.DATA.NUM_CLASSES = len(self.classes)
        config.freeze()

    def raw_audio_mixup(self, x, y, alpha=5, beta=2, batched=False):
        lam = np.random.beta(alpha, beta)
        if batched:
            batch_size = x.shape[0]
            index = np.random.permutation(batch_size)

            x = lam * x + (1 - lam) * x[index, :]
            if lam < 0.5:
                y = y[index]
        else:
            random_idx = np.random.randint(len(self.samples))
            rel_audio_path, target_class, *_ = self.samples.iloc[random_idx]
            target_class = self.class_to_idx[target_class]
            audio_path = os.path.join(self.path_to_dataset, rel_audio_path)

            mixup_audio, audio_sr = self.load_audio_sample(
                path=audio_path,
                use_silence_padding=self.use_silence_padding,
                sampling_rate=self.sampling_rate,
                top_db=self.top_db,
                max_audio_length=self.max_audio_length,
            )
            x = lam * x + (1 - lam) * mixup_audio

            # doesn't allow for multi-target
            if lam < 0.5:
                y = target_class
        return x, y

    def _find_classes(self, config):
        classes = sorted(self.samples["class"].unique())
        if self.use_negative_samples:
            classes.remove("NEGATIVE")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        if self.use_negative_samples:
            class_to_idx["NEGATIVE"] = -1
        return classes, class_to_idx

    def _make_samples(self, dataset_type, config, use_mixing):
        """
        Return a pandas dataframe containing information about each individual
        vocalisation (class as idx, path_to_file, and potentially any mixing files)
        """
        raise NotImplementedError

    def _make_cross_validation_splits(self, config, meta_file):
        for f in range(1, config.TRAIN.KFOLDS + 1):
            train_indices, _ = sklearn.model_selection.train_test_split(
                meta_file.index,
                train_size=config.DATA.TRAIN_VAL_SPLIT,
                # random_state=config.GENERAL.SEED,
                random_state=f,
                stratify=meta_file["class"],
            )
            meta_file[f"is_train-fold_{f}"] = False
            meta_file.loc[train_indices, f"is_train-fold_{f}"] = True
        return meta_file

    @staticmethod
    def _add_negatives_to_samples(config, meta_file, embedder=None):
        # load negative samples and append to bottom
        if embedder is None:
            neg_path = os.path.join(
                config.DATA.DATA_PATH, "ESC-50-master/meta/esc50.csv"
            )
            negative_samples = pd.read_csv(neg_path)
            negative_samples = negative_samples[negative_samples["esc10"] == True]
            negative_samples = negative_samples.rename(
                columns={"filename": "file_name"}
            )
            negative_samples["class"] = "NEGATIVE"
            negative_samples["file_name"] = (
                config.DATA.DATA_PATH
                + "/ESC-50-master/audio/"
                + negative_samples["file_name"]
            )
            negative_samples = negative_samples[["file_name", "class"]]
        else:
            neg_path = os.path.join(
                config.DATA.DATA_PATH, "ESC-50-master/ESC-10", f"{embedder}-embeddings"
            )
            neg_path = os.path.join(neg_path, "embeddings.json")
            negative_samples = pd.read_json(neg_path, orient="records", lines=True)
            negative_samples["class"] = "NEGATIVE"
            negative_samples = negative_samples[["embedding", "class"]]

        negative_samples["is_train"] = True
        meta_file = pd.concat([meta_file, negative_samples], axis=0, ignore_index=True)
        return meta_file


class RTBCCallTypes(AudioDataset):
    """
    All call types (begging + perch + flight)
    If individual recognition: use DATA.CLASS = individual
    """

    def _make_samples(self, dataset_type, config, use_mixing, use_subset):
        meta_file = pd.read_csv(self.path_to_dataset + "/metadata.csv")
        if self.fold == 1 and self.dataset_type == "train":
            meta_file = self._make_cross_validation_splits(config, meta_file)
            meta_file.to_csv(self.path_to_dataset + "/metadata.csv", index=False)
        meta_file["is_train"] = meta_file[f"is_train-fold_{self.fold}"]
        meta_file = meta_file[~meta_file["class"].isin(config.DATA.CLASSES_TO_DROP)]
        meta_file = meta_file[["file_name", "class", "is_train"]]
        if self.use_negative_samples:
            meta_file = self._add_negatives_to_samples(config, meta_file, embedder=None)
        meta_file = meta_file.reset_index(drop=True)

        return meta_file


class RTBCEmbedded(AudioDataset):
    """
    Begging/Flight/Perch calls only, individual recognition
    """

    def __getitem__(self, idx):
        audio_sig, target_class, *_ = self.samples.iloc[idx]
        target_class = self.class_to_idx[target_class]
        audio_sig = torch.tensor(audio_sig)
        return audio_sig, target_class

    def _make_samples(self, dataset_type, config, use_mixing, use_subset=True):
        if "google-perch" in config.MODEL.NAME:
            embedder = "google-perch"
        elif "birdnet" in config.MODEL.NAME:
            embedder = "birdnet"
        embeddings_path = os.path.join(
            config.DATA.DATA_PATH,
            "rtbc-call-types",
            f"{config.DATA.DATASET_PATH}-{embedder}-embeddings",
            "embeddings.json",
        )
        embeddings = pd.read_json(embeddings_path, orient="records", lines=True)
        if self.fold == 1 and self.dataset_type == "train":
            embeddings["class"] = embeddings["nest"] + "_" + embeddings["season"]
            embeddings = self._make_cross_validation_splits(config, embeddings)
            embeddings.to_json(embeddings_path, orient="records", lines=True)
        embeddings["is_train"] = embeddings[f"is_train-fold_{self.fold}"]
        embeddings = embeddings[["embedding", "class", "is_train"]]
        embeddings = embeddings[~embeddings["class"].isin(config.DATA.CLASSES_TO_DROP)]

        if self.use_negative_samples:
            embeddings = self._add_negatives_to_samples(config, embeddings, embedder)
        embeddings = embeddings[["embedding", "class", "is_train"]]
        embeddings = embeddings.reset_index(drop=True)

        return embeddings


class LittleOwls(AudioDataset):
    def _make_samples(self, dataset_type, config, use_mixing, use_subset):
        meta_file = pd.read_csv(self.path_to_dataset + "/metadata.csv")
        if self.fold == 1 and self.dataset_type == "train":
            meta_file = self._make_cross_validation_splits(config, meta_file)
            meta_file.to_csv(self.path_to_dataset + "/metadata.csv", index=False)
        meta_file["is_train"] = meta_file[f"is_train-fold_{self.fold}"]
        meta_file = meta_file[~meta_file["class"].isin(config.DATA.CLASSES_TO_DROP)]
        meta_file = meta_file[["file_name", "class", "is_train"]]
        if self.use_negative_samples:
            meta_file = self._add_negatives_to_samples(config, meta_file, embedder=None)
        meta_file = meta_file.reset_index(drop=True)
        return meta_file


class LittleOwlsEmbedded(AudioDataset):
    def __getitem__(self, idx):
        audio_sig, target_class, *_ = self.samples.iloc[idx]
        target_class = self.class_to_idx[target_class]
        audio_sig = torch.tensor(audio_sig)
        return audio_sig, target_class

    def _make_samples(self, dataset_type, config, use_mixing, use_subset=True):
        if "google-perch" in config.MODEL.NAME:
            embedder = "google-perch"
        elif "birdnet" in config.MODEL.NAME:
            embedder = "birdnet"
        embedder_path = os.path.join(
            config.DATA.DATA_PATH, "littleowl-fg", f"{embedder}-embeddings"
        )
        embeddings_path = os.path.join(embedder_path, "embeddings.json")
        embeddings = pd.read_json(embeddings_path, orient="records", lines=True)
        embeddings = embeddings.rename(columns={"name": "class"})
        if self.fold == 1 and self.dataset_type == "train":

            embeddings = self._make_cross_validation_splits(config, embeddings)
            embeddings.to_json(embeddings_path, orient="records", lines=True)
        embeddings["is_train"] = embeddings[f"is_train-fold_{self.fold}"]
        embeddings = embeddings[~embeddings["class"].isin(config.DATA.CLASSES_TO_DROP)]
        if self.use_negative_samples:
            embeddings = self._add_negatives_to_samples(config, embeddings, embedder)
        embeddings = embeddings[["embedding", "class", "is_train"]]
        embeddings = embeddings.reset_index(drop=True)
        return embeddings


class Pipits(AudioDataset):
    def _make_samples(self, dataset_type, config, use_mixing, use_subset=True):
        meta_file = pd.read_csv(self.path_to_dataset + f"/metadata.csv")
        meta_file["class"] = meta_file["class"].astype(str)
        if self.fold == 1 and self.dataset_type == "train":
            meta_file = self._make_cross_validation_splits(config, meta_file)
            meta_file.to_csv(self.path_to_dataset + "/metadata.csv", index=False)
        meta_file["is_train"] = meta_file[f"is_train-fold_{self.fold}"]
        meta_file = meta_file[~meta_file["class"].isin(config.DATA.CLASSES_TO_DROP)]
        meta_file = meta_file[["file_name", "class", "is_train"]]
        if self.use_negative_samples:
            meta_file = self._add_negatives_to_samples(config, meta_file, embedder=None)
        meta_file = meta_file.reset_index(drop=True)
        return meta_file


class PipitsEmbedded(AudioDataset):
    def __getitem__(self, idx):
        audio_sig, target_class, *_ = self.samples.iloc[idx]
        target_class = self.class_to_idx[target_class]
        audio_sig = torch.tensor(audio_sig)
        return audio_sig, target_class

    def _make_samples(self, dataset_type, config, use_mixing, use_subset=True):
        if "google-perch" in config.MODEL.NAME:
            embedder = "google-perch"
        elif "birdnet" in config.MODEL.NAME:
            embedder = "birdnet"
        embedder_path = os.path.join(
            config.DATA.DATA_PATH, "pipit-fg", f"{embedder}-embeddings"
        )
        embeddings_path = os.path.join(embedder_path, "embeddings.json")
        embeddings = pd.read_json(embeddings_path, orient="records", lines=True)
        embeddings = embeddings.rename(columns={"name": "class"})
        if self.fold == 1 and self.dataset_type == "train":
            embeddings = self._make_cross_validation_splits(config, embeddings)
            embeddings.to_json(embeddings_path, orient="records", lines=True)
        embeddings["is_train"] = embeddings[f"is_train-fold_{self.fold}"]
        embeddings["class"] = embeddings["class"].astype(str)
        embeddings = embeddings[~embeddings["class"].isin(config.DATA.CLASSES_TO_DROP)]
        if self.use_negative_samples:
            embeddings = self._add_negatives_to_samples(config, embeddings, embedder)
        embeddings = embeddings[["embedding", "class", "is_train"]]
        embeddings = embeddings.reset_index(drop=True)
        return embeddings


class ChiffChaffs(AudioDataset):
    def _make_samples(self, dataset_type, config, use_mixing, use_subset=True):
        meta_file = pd.read_csv(self.path_to_dataset + "/metadata.csv")
        if self.fold == 1 and self.dataset_type == "train":

            meta_file = self._make_cross_validation_splits(config, meta_file)
            meta_file.to_csv(self.path_to_dataset + "/metadata.csv", index=False)
        meta_file["is_train"] = meta_file[f"is_train-fold_{self.fold}"]
        meta_file = meta_file[~meta_file["class"].isin(config.DATA.CLASSES_TO_DROP)]
        meta_file = meta_file[["file_name", "class", "is_train"]]
        if self.use_negative_samples:
            meta_file = self._add_negatives_to_samples(config, meta_file, embedder=None)
        meta_file = meta_file.reset_index(drop=True)
        return meta_file


class ChiffChaffsEmbedded(AudioDataset):
    def __getitem__(self, idx):
        audio_sig, target_class, *_ = self.samples.iloc[idx]
        target_class = self.class_to_idx[target_class]
        audio_sig = torch.tensor(audio_sig)
        return audio_sig, target_class

    def _make_samples(self, dataset_type, config, use_mixing, use_subset=True):
        if "google-perch" in config.MODEL.NAME:
            embedder = "google-perch"
        elif "birdnet" in config.MODEL.NAME:
            embedder = "birdnet"
        embedder_path = os.path.join(
            config.DATA.DATA_PATH, "chiffchaff-fg", f"{embedder}-embeddings"
        )
        embeddings_path = os.path.join(embedder_path, "within-year-embeddings.json")
        embeddings = pd.read_json(embeddings_path, orient="records", lines=True)
        embeddings = embeddings.rename(columns={"name": "class"})
        if self.fold == 1 and self.dataset_type == "train":
            embeddings = self._make_cross_validation_splits(config, embeddings)
            embeddings.to_json(embeddings_path, orient="records", lines=True)
        embeddings["is_train"] = embeddings[f"is_train-fold_{self.fold}"]
        embeddings["class"] = embeddings["class"].astype(str)
        embeddings = embeddings[~embeddings["class"].isin(config.DATA.CLASSES_TO_DROP)]
        if self.use_negative_samples:
            embeddings = self._add_negatives_to_samples(config, embeddings, embedder)
        embeddings = embeddings[["embedding", "class", "is_train"]]
        embeddings = embeddings.reset_index(drop=True)
        return embeddings


class LittlePenguins(AudioDataset):
    def _make_samples(self, dataset_type, config, use_mixing, use_subset=True):
        meta_file = pd.read_csv(self.path_to_dataset + "/metadata.csv")
        meta_file = meta_file[meta_file["call_type"] == "exhale"]
        if self.fold == 1 and self.dataset_type == "train":

            meta_file = self._make_cross_validation_splits(config, meta_file)
            meta_file.to_csv(self.path_to_dataset + "/metadata.csv", index=False)
        meta_file["is_train"] = meta_file[f"is_train-fold_{self.fold}"]
        warnings.warn(f"only using {config.DATA.CLASS} calls")
        if config.DATA.CLASS == "exhale":
            assert config.DATA.MAX_AUDIO_LENGTH == 3.0
        elif config.DATA.CLASS == "inhale":
            assert config.DATA.MAX_AUDIO_LENGTH == 1.5
        else:
            raise ValueError("Unknown class type")
        meta_file = meta_file[~meta_file["class"].isin(config.DATA.CLASSES_TO_DROP)]
        meta_file = meta_file[meta_file["call_type"] == config.DATA.CLASS]
        meta_file = meta_file[["file_name", "class", "is_train"]]
        if self.use_negative_samples:
            meta_file = self._add_negatives_to_samples(config, meta_file, embedder=None)
        meta_file = meta_file.reset_index(drop=True)
        return meta_file


class LittlePenguinsEmbedded(AudioDataset):
    def __getitem__(self, idx):
        audio_sig, target_class, *_ = self.samples.iloc[idx]
        target_class = self.class_to_idx[target_class]
        audio_sig = torch.tensor(audio_sig)
        return audio_sig, target_class

    def _make_samples(self, dataset_type, config, use_mixing, use_subset=True):
        if "google-perch" in config.MODEL.NAME:
            embedder = "google-perch"
        elif "birdnet" in config.MODEL.NAME:
            embedder = "birdnet"
        embedder_path = os.path.join(
            config.DATA.DATA_PATH, "littlepenguin", f"{embedder}-embeddings"
        )
        embeddings_path = os.path.join(embedder_path, "embeddings.json")
        embeddings = pd.read_json(embeddings_path, orient="records", lines=True)
        if self.fold == 1 and self.dataset_type == "train":
            embeddings["class"] = embeddings["nest"]
            embeddings = self._make_cross_validation_splits(config, embeddings)
            embeddings.to_json(embeddings_path, orient="records", lines=True)
        embeddings["is_train"] = embeddings[f"is_train-fold_{self.fold}"]
        embeddings["call_type"] = embeddings["call_type"].apply(
            lambda x: "exhale" if "exhale" in x else "inhale"
        )
        embeddings = embeddings[embeddings["call_type"] == config.DATA.CLASS]
        embeddings = embeddings[~embeddings["class"].isin(config.DATA.CLASSES_TO_DROP)]
        if self.use_negative_samples:
            embeddings = self._add_negatives_to_samples(config, embeddings, embedder)
        embeddings = embeddings[["embedding", "class", "is_train"]]
        embeddings = embeddings.reset_index(drop=True)
        return embeddings
