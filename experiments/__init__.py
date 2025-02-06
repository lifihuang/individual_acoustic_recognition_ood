from experiments.basic_classification import BasicClassification

__experiments__ = {
    "AemNet": BasicClassification,
    "ProjFCHead": BasicClassification,
}


def select_experiment(config, device):
    experiment = config.MODEL.TYPE
    model_name = config.MODEL.NAME
    if experiment not in __experiments__:
        raise NotImplementedError(f"{experiment} is not implemented!")
    print(f"Running new experiment \"{model_name}\" of type \"{experiment}\". Device: {device}")
    return __experiments__[experiment](config, device)