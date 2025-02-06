import warnings

from torch.utils.data import DataLoader

from data.datasets import (
    LittleOwlsEmbedded,
    LittlePenguinsEmbedded,
    RTBCEmbedded,
    PipitsEmbedded,
    ChiffChaffsEmbedded,
    RTBCCallTypes,
    LittlePenguins,
    LittleOwls,
    Pipits,
    ChiffChaffs,
)


def get_dataset(dataset):
    datasets = {
        "littleowl-fg": LittleOwls,
        "littleowl-fg_embedded": LittleOwlsEmbedded,
        "littlepenguin": LittlePenguins,
        "littlepenguin_embedded": LittlePenguinsEmbedded,
        "rtbc-call-types": RTBCCallTypes,
        "rtbc_embedded": RTBCEmbedded,
        "pipit-fg": Pipits,
        "pipit-fg_embedded": PipitsEmbedded,
        "chiffchaff-fg": ChiffChaffs,
        "chiffchaff-fg_embedded": ChiffChaffsEmbedded,
    }
    return datasets[dataset]


def get_saved_model_path(
    config_model_type,
    config_model_name,
    config_dataset_name,
    config_data_class,
    config_dataset_path,
    fold,
):
    model_path = [
        config_model_type,
        config_model_name,
        config_dataset_name,
        config_dataset_path,
        f"best_model_fold{fold}.pt",
    ]
    if config_data_class is not None:
        model_path.insert(-1, config_data_class)
    model_path = "--".join(model_path)
    return model_path


def get_val_data_and_loader(
    dataset_name, cls_dropped_from_training, config, device, fold, use_as_test_set=True
):
    dataset = get_dataset(dataset_name)
    warnings.warn("Make sure validation contains ALL available data")
    val_data = dataset(
        dataset_type="validation",
        config=config,
        use_mixing=None,
        use_negative_samples=False,
        fold=fold,
        device=device,
    )

    val_data.samples = val_data.samples[
        (val_data.samples["is_train"] == False)
        | (val_data.samples["class"].isin(cls_dropped_from_training))
    ]

    val_loader = DataLoader(val_data, shuffle=False, batch_size=config.DATA.BATCH_SIZE)
    return val_data, val_loader
