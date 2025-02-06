import warnings

from models.classifiers import *


def build_model(config):
    if config.MODEL.TYPE == "AemNet":
        return_penult_feat_and_pred = False
        model = AemNet(
            num_classes=config.DATA.NUM_CLASSES,
            num_projection_elements=config.HYPERPARAMS.NUM_PROJ_ELEMS,
            width_mul=config.HYPERPARAMS.WIDTH_MUL,
            raw_audio_input=True if config.INPUT_TRANSFORM.METHOD is None else False,
            use_clf=True,
            return_penult_feat_and_pred=return_penult_feat_and_pred,
        )
    elif config.MODEL.TYPE == "ProjFCHead":
        return_penult_feat_and_pred = False
        model = ProjFCHead(
            latent_dim=config.HYPERPARAMS.LATENT_DIM,
            num_classes=config.DATA.NUM_CLASSES,
            num_projection_elements=config.HYPERPARAMS.NUM_PROJ_ELEMS,
            return_penult_feat_and_pred=return_penult_feat_and_pred,
        )
    else:
        raise NotImplementedError("Model not available")
    return model
