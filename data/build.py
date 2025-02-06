import sklearn.model_selection
import torch
from copy import deepcopy
from torch.utils.data import DataLoader, Sampler, Subset, BatchSampler, RandomSampler

from data.datasets import (
    RTBCEmbedded,
    RTBCCallTypes,
    LittleOwls,
    LittleOwlsEmbedded,
    LittlePenguins,
    LittlePenguinsEmbedded,
    Pipits,
    PipitsEmbedded,
    ChiffChaffsEmbedded,
    ChiffChaffs,
)


class OneClassPerBatchSampler(Sampler):
    def __init__(self, data):
        self.num_samples = len(data)
        subset_data = data.dataset.samples.iloc[data.indices]
        self.num_classes = subset_data["class"].nunique()
        # this is used for subsets, where the returned indexes need to be positions of self.indices
        self.cls_indices = [
            v.tolist()
            for v in subset_data.reset_index().groupby("class").groups.values()
        ]
        # this is used when the returned indexes are actually indexes of self.samples
        # self.cls_indices = [v.tolist() for v in subset_data.groupby("class").groups.values()]

    def __iter__(self):
        batch = []
        cls_indices = deepcopy(self.cls_indices)
        for _ in range(len(self)):
            for i, cls_idxs in enumerate(cls_indices):
                try:
                    rand_idx = torch.multinomial(torch.ones(len(cls_idxs)), 1).item()
                    cls_idx = cls_idxs.pop(rand_idx)
                except RuntimeError:  # when cls_idxs is empty
                    rand_idx = torch.multinomial(
                        torch.ones(len(self.cls_indices[i])), 1
                    ).item()
                    cls_idx = self.cls_indices[i][rand_idx]
                batch.append(cls_idx)
            yield batch
            batch = []

    def __len__(self):
        return self.num_samples // self.num_classes


def get_train_and_val_dataset(config, fold, device):
    train_indices = None
    val_indices = None
    use_negative_samples = True if "with-negs" in config.MODEL.NAME else False
    if config.DATA.DATASET == "rtbc_embedded":
        train_data = RTBCEmbedded(
            dataset_type="train",
            config=config,
            use_mixing=None,
            use_negative_samples=use_negative_samples,
            fold=fold,
            device=device,
        )
        val_data = RTBCEmbedded(
            dataset_type="validation",
            config=config,
            use_mixing=None,
            use_negative_samples=False,
            fold=fold,
            device=device,
        )
    elif config.DATA.DATASET == "rtbc-call-types":
        train_data = RTBCCallTypes(
            dataset_type="train",
            config=config,
            use_mixing=None,
            use_negative_samples=use_negative_samples,
            fold=fold,
            device=device,
        )
        val_data = RTBCCallTypes(
            dataset_type="validation",
            config=config,
            use_mixing=None,
            use_negative_samples=False,
            fold=fold,
            device=device,
        )
    elif config.DATA.DATASET == "littleowl-fg":
        if (
            config.DATA.DATASET_PATH is not None
            and "embeddings" in config.DATA.DATASET_PATH
        ):
            train_data = LittleOwlsEmbedded(
                dataset_type="train",
                config=config,
                use_mixing=None,
                use_negative_samples=use_negative_samples,
                fold=fold,
                device=device,
            )
            val_data = LittleOwlsEmbedded(
                dataset_type="validation",
                config=config,
                use_mixing=None,
                use_negative_samples=False,
                fold=fold,
                device=device,
            )
        else:
            train_data = LittleOwls(
                dataset_type="train",
                config=config,
                use_mixing=None,
                use_negative_samples=use_negative_samples,
                fold=fold,
                device=device,
            )
            val_data = LittleOwls(
                dataset_type="validation",
                config=config,
                use_mixing=None,
                use_negative_samples=False,
                fold=fold,
                device=device,
            )
    elif config.DATA.DATASET == "chiffchaff-fg":
        if (
            config.DATA.DATASET_PATH is not None
            and "embeddings" in config.DATA.DATASET_PATH
        ):
            train_data = ChiffChaffsEmbedded(
                dataset_type="train",
                config=config,
                use_mixing=None,
                use_negative_samples=use_negative_samples,
                fold=fold,
                device=device,
            )
            val_data = ChiffChaffsEmbedded(
                dataset_type="validation",
                config=config,
                use_mixing=None,
                use_negative_samples=False,
                fold=fold,
                device=device,
            )
        else:
            train_data = ChiffChaffs(
                dataset_type="train",
                config=config,
                use_mixing=None,
                use_negative_samples=use_negative_samples,
                fold=fold,
                device=device,
            )
            val_data = ChiffChaffs(
                dataset_type="validation",
                config=config,
                use_mixing=None,
                use_negative_samples=False,
                fold=fold,
                device=device,
            )
    elif config.DATA.DATASET == "pipit-fg":
        train_data = Pipits(
            dataset_type="train",
            config=config,
            use_mixing=None,
            use_negative_samples=use_negative_samples,
            fold=fold,
            device=device,
        )
        val_data = Pipits(
            dataset_type="validation",
            config=config,
            use_mixing=None,
            use_negative_samples=False,
            fold=fold,
            device=device,
        )
    elif config.DATA.DATASET == "pipit-fg_embedded":
        train_data = PipitsEmbedded(
            dataset_type="train",
            config=config,
            use_mixing=None,
            use_negative_samples=use_negative_samples,
            fold=fold,
            device=device,
        )
        val_data = PipitsEmbedded(
            dataset_type="validation",
            config=config,
            use_mixing=None,
            use_negative_samples=False,
            fold=fold,
            device=device,
        )
    elif config.DATA.DATASET == "littlepenguin":
        train_data = LittlePenguins(
            dataset_type="train",
            config=config,
            use_mixing=None,
            use_negative_samples=use_negative_samples,
            fold=fold,
            device=device,
        )
        val_data = LittlePenguins(
            dataset_type="validation",
            config=config,
            use_mixing=None,
            use_negative_samples=False,
            fold=fold,
            device=device,
        )
    elif config.DATA.DATASET == "littlepenguin_embedded":
        train_data = LittlePenguinsEmbedded(
            dataset_type="train",
            config=config,
            use_mixing=None,
            use_negative_samples=use_negative_samples,
            fold=fold,
            device=device,
        )
        val_data = LittlePenguinsEmbedded(
            dataset_type="validation",
            config=config,
            use_mixing=None,
            use_negative_samples=False,
            fold=fold,
            device=device,
        )

    else:
        raise NotImplementedError("More datasets to come.")

    return train_data, val_data, train_indices, val_indices


def build_loader(config, fold, device, logger):
    train_data, val_data, train_indices, val_indices = get_train_and_val_dataset(
        config, fold, device
    )
    if train_indices is None and val_indices is None:
        if "is_train" in train_data.samples.columns:
            train_indices = train_data.samples[
                train_data.samples["is_train"] == True
            ].index
            val_indices = train_data.samples[
                train_data.samples["is_train"] == False
            ].index
        else:
            train_indices, val_indices = sklearn.model_selection.train_test_split(
                train_data.samples.index,
                train_size=config.DATA.TRAIN_VAL_SPLIT,
                random_state=config.GENERAL.SEED,
                stratify=train_data.samples["class"],
            )
    if config.TRAIN.ARGS.USE_X_PERC_DATA != 1.0:
        cls_col = "category" if "ESC" in config.DATA.DATASET else "class"
        train_indices, _ = sklearn.model_selection.train_test_split(
            train_indices,
            train_size=config.TRAIN.ARGS.USE_X_PERC_DATA,
            random_state=config.GENERAL.SEED,
            stratify=train_data.samples.loc[train_indices][cls_col],
        )
    if fold == 1:
        logger.info(
            f"Dropping classes in {config.DATA.CLASSES_TO_DROP}"
        )  # happens in get_train_and_val_dataset
        logger.info(
            f"Using {config.TRAIN.ARGS.USE_X_PERC_DATA * 100}% of the available training data."
        )
        logger.info(
            f"{train_data.samples.loc[train_indices]['class'].value_counts(True)}"
        )
    train_subset = Subset(train_data, train_indices)
    if config.DATA.ONE_CLASS_PER_BATCH:
        BS = OneClassPerBatchSampler(train_subset)
        train_loader = DataLoader(
            train_subset, sampler=BS, batch_sampler=None, batch_size=None
        )
        logger.info(f"Setting batch size to number of classes {BS.num_classes}")
        config.defrost()
        config.DATA.BATCH_SIZE = BS.num_classes
        config.freeze()
    elif "mixup" in config.TRAIN.AUGS.METHODS:
        # mixup is best done with manual batching
        train_loader = DataLoader(
            train_subset,
            sampler=BatchSampler(
                sampler=RandomSampler(train_subset),
                batch_size=config.DATA.BATCH_SIZE,
                drop_last=False
                if config.GENERAL.LEARN_METHOD != "contrastive"
                else True,
            ),
            batch_sampler=None,
            batch_size=None,
        )
    else:
        train_loader = DataLoader(
            train_subset,
            shuffle=True,
            batch_size=config.DATA.BATCH_SIZE,
            drop_last=False if config.GENERAL.LEARN_METHOD != "contrastive" else True,
        )

    val_subset = Subset(val_data, val_indices)
    val_loader = DataLoader(
        val_subset, shuffle=False, batch_size=config.DATA.BATCH_SIZE
    )
    return train_loader, val_loader
