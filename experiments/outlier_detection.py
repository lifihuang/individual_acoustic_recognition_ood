"""
Bits and pieces from: https://github.com/zjs975584714/SHE_ood_detection/blob/main/test_score_ood_detection.py
"""
import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, roc_auc_score
from torch.utils.data import DataLoader

from config import get_config
from models.build import build_model
from experiments.val_utils import (
    get_dataset,
    get_val_data_and_loader,
    get_saved_model_path,
)


def generate_she_patterns(
    model,
    model_cls_to_idx,
    dataset_name,
    model_name,
    cls_dropped_from_training,
    fold,
    config,
):
    # get all data, make sure not using any augmentations
    dataset = get_dataset(dataset_name)
    warnings.warn("Make sure validation contains ALL available data")
    pattern_data = dataset(
        dataset_type="validation",
        config=config,
        use_mixing=None,
        use_negative_samples=False,
        fold=fold,
        device=device,
    )
    # generating training patterns from only training data classes
    warnings.warn("this does not update the classes and class_to_idx variable")
    pattern_data.samples = pattern_data.samples[
        (pattern_data.samples["is_train"] == True)
        & ~(pattern_data.samples["class"].isin(cls_dropped_from_training))
    ]
    # this is not redundant
    config.defrost()
    config.DATA.NUM_CLASSES -= len(CLS_DROPPED_FROM_TRAINING)
    config.freeze()
    pattern_loader = DataLoader(
        pattern_data, shuffle=False, batch_size=config.DATA.BATCH_SIZE
    )

    # this is needed to combat the issue of mismatching cls_to_idx between the trained model and the pattern dataset
    target_to_model_idx = {
        k: model_cls_to_idx.get(v) for v, k in pattern_data.class_to_idx.items()
    }
    pattern_list = [None for _ in range(config.DATA.NUM_CLASSES)]
    with torch.no_grad():
        for features, targets in pattern_loader:
            targets.apply_(target_to_model_idx.get)
            features, targets = (x.to(device) for x in (features, targets))
            patterns, logits = model(features)
            ##
            warnings.warn("USING LOGITS AS PATTERNS")
            patterns = logits
            ##
            # patterns, logits = (x.detach().to("cpu") for x in (patterns, logits))
            _, preds = torch.max(logits, dim=1)
            correct_idx = preds.eq(targets)
            for i in range(config.DATA.NUM_CLASSES):
                each_label_tensor = torch.tensor(
                    [i for _ in range(targets.size(0))]
                ).to(device)
                target_index = preds.eq(each_label_tensor)
                # idx at which both correct classification and idx of interest
                combine_index = correct_idx * target_index
                each_label_pattern = patterns[combine_index]
                if each_label_pattern.size(0) == 0:
                    continue
                if pattern_list[i] is None:
                    pattern_list[i] = each_label_pattern
                else:
                    pattern_list[i] = torch.cat(
                        (pattern_list[i], each_label_pattern), dim=0
                    )

    # ### visualizing pattern list and mean pattern lists
    # data = torch.cat(pattern_list, dim=0).cpu().numpy()
    # means = [tensor.mean(dim=0) for tensor in pattern_list]
    # means = torch.stack(means).cpu().numpy()
    # labels = torch.cat([torch.full((len(tsr),), idx) for idx, tsr in enumerate(pattern_list)]).cpu().numpy()
    # data_and_means = np.vstack([data, means])

    # from sklearn.manifold import TSNE
    # tsne = TSNE(n_components=2, random_state=42)
    # data_and_means_tsne = tsne.fit_transform(data_and_means)
    # data_tsne, means_tsne = data_and_means_tsne[:len(data), :], data_and_means_tsne[len(data):, :]

    # # Step 3: Plot the results
    # plt.figure(figsize=(8, 6))
    # for class_idx in range(len(pattern_list)):
    #     plt.scatter(data_tsne[labels == class_idx, 0],
    #                 data_tsne[labels == class_idx, 1],
    #                 label=f'Class {class_idx}')
    #     plt.scatter(means_tsne[class_idx, 0],
    #                 means_tsne[class_idx, 1],
    #                 marker='X', s=100, edgecolor='k', label=f'Class {class_idx} Mean')
    # plt.legend()
    # plt.title("t-SNE visualization of 2D tensors")
    # plt.xlabel("t-SNE Dimension 1")
    # plt.ylabel("t-SNE Dimension 2")
    # plt.show()

    for i in range(config.DATA.NUM_CLASSES):
        pattern_list[i] = torch.mean(pattern_list[i], dim=0, keepdim=True)
        torch.save(
            pattern_list[i],
            os.path.join(
                config.GENERAL.SAVED_DIR,
                "she_patterns",
                f"{dataset_name}_{model_name}_stored_avg_class_{i}.pth",
            ),
        )


def get_scores_and_labels(
    model, val_loader, dataset_name, model_name, model_cls_to_idx
):
    OOD_INT = 999  # dummy variable for OOD classes in test set
    # this is needed to combat the issue of mismatching cls_to_idx between the trained model and the pattern dataset
    target_to_model_idx = {
        k: model_cls_to_idx.get(v, OOD_INT)
        for v, k in val_loader.dataset.class_to_idx.items()
    }
    id_scores = []
    ood_scores = []
    id_targets = []
    ood_targets = []
    with torch.no_grad():
        for features, targets in val_loader:
            features = features.to(device)
            targets.apply_(target_to_model_idx.get)
            features = features.to(device)
            patterns, logits = model(features)
            ##
            warnings.warn("USING LOGITS AS PATTERNS")
            patterns = logits
            ##
            she_score = compute_she_score(
                patterns, logits, dataset_name, model_name, config
            )
            id_scores.append(she_score[(targets != OOD_INT).numpy()])
            ood_scores.append(she_score[(targets == OOD_INT).numpy()])
            id_targets.append(targets[targets != OOD_INT])
            ood_targets.append(targets[targets == OOD_INT])
    id_scores, ood_scores = (np.concatenate(x, axis=0) for x in (id_scores, ood_scores))
    # id_targets, ood_targets = (np.concatenate(x, axis=0) for x in (id_targets, ood_targets))  # not used
    id_scores = id_scores.reshape(-1, 1)
    ood_scores = ood_scores.reshape(-1, 1)
    scores = np.squeeze(np.vstack((id_scores, ood_scores)))
    labels = np.zeros(len(scores), dtype=np.int32)
    labels[: len(id_scores)] += 1  # in-distribution is True and ood is false
    return scores, labels


def compute_she_score(patterns, logits, dataset_name, model_name, config):
    numclass = config.DATA.NUM_CLASSES
    # ----------------------------------------Step 1: classifier the test feature-----------------------------------
    if logits.dim() == 1:
        patterns = patterns.unsqueeze(0)
        logits = logits.unsqueeze(0)
    pred = logits.argmax(dim=1, keepdim=False)
    pred = pred.cpu().tolist()

    # ----------------------------------------Step 2: get the stored pattern------------------------------------
    total_stored_feature = None
    for i in range(numclass):
        path = os.path.join(
            config.GENERAL.SAVED_DIR,
            "she_patterns",
            f"{dataset_name}_{model_name}_stored_avg_class_{i}.pth",
        )
        stored_tensor = torch.load(path)
        if total_stored_feature is None:
            total_stored_feature = stored_tensor
        else:
            total_stored_feature = torch.cat(
                (total_stored_feature, stored_tensor), dim=0
            )

    # --------------------------------------------------------------------------------------
    target = total_stored_feature[pred, :]
    res_energy_score = torch.sum(torch.mul(patterns, target), dim=1)  # inner product

    return res_energy_score.data.cpu().numpy()


def calculate_metrics(scores, labels):
    """
    for SHE only
    :param _id_scores:
    :param _ood_scores:
    :return:
    """
    aucroc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)

    return aucroc, fpr, tpr


def calculate_oscr(all_logits, all_targets, OOD_INT=999):
    # Taken from: https://github.com/sgvaze/osr_closed_set_all_you_need/blob/main/test/utils.py#L125
    """
    :param all_logits: (# test samples, )
    :param all_targets: (# test samples, # num_id classes + 1) all OOD classes should have label OOD_INT, 999 by default
    :return:
    """
    if type(all_logits) == torch.Tensor and type(all_targets) == torch.Tensor:
        all_logits, all_targets = all_logits.numpy(), all_targets.numpy()
    # split all logits into known and unknown samples
    id_samples, ood_samples = (
        all_logits[all_targets != OOD_INT],
        all_logits[all_targets == OOD_INT],
    )
    id_samples_targets = all_targets[all_targets != OOD_INT]
    # classify known samples
    preds = id_samples.argmax(axis=1)
    correct = preds == id_samples_targets
    # correctly classified known samples
    correctly_classified_id_samples = id_samples[correct]

    n = len(all_logits)
    CCR = [0 for x in range(n + 1)]
    FPR = [0 for x in range(n + 1)]

    # thetas
    thetas = np.linspace(all_logits.min(), all_logits.max(), n)
    for i, theta in enumerate(thetas):
        CC = correctly_classified_id_samples.max(axis=1) > theta
        FP = np.any(ood_samples > theta, axis=1)
        CCR[i] = CC.sum() / len(id_samples)
        FPR[i] = FP.sum() / len(ood_samples)

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)

    OSCR = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n):
        h = ROC[j][0] - ROC[j + 1][0]
        w = (ROC[j][1] + ROC[j + 1][1]) / 2.0

        OSCR = OSCR + h * w

    return OSCR, ROC


def embed_and_evaluate_data_using_msp(
    val_loader, model, model_cls_to_idx, cls_dropped_from_training, device
):
    # embedding data
    OOD_INT = 999  # dummy variable for OOD classes in test set
    open_set_preds = []
    all_targets = []
    target_to_model_idx = {
        k: model_cls_to_idx.get(v, OOD_INT)
        for v, k in val_loader.dataset.class_to_idx.items()
    }
    with torch.no_grad():
        for features, targets in val_loader:
            features = features.to(device)
            l = model(features).detach().to("cpu")
            if l.dim() == 1:
                l = l.unsqueeze(0)
            open_set_preds.append(l)
            targets.apply_(target_to_model_idx.get)
            all_targets.append(targets)
    open_set_preds = torch.cat(open_set_preds, dim=0)
    all_logits = open_set_preds.clone()
    all_targets = torch.cat(
        all_targets, dim=0
    )  # targets in terms of model indices, different to val_loader indices
    # val_idx_to_cls = {v:k for k,v in val_data.class_to_idx.items()}
    model_idx_to_cls = {v: k for k, v in model_cls_to_idx.items()}

    # in/out targets
    in_out_targets = all_targets.clone()

    def map_idx_to_in_or_out(x):
        cls = model_idx_to_cls.get(x)
        if cls in cls_dropped_from_training or cls is None:
            return 1  # unknown class (different to SHE)
        else:
            return 0  # known class

    in_out_targets.apply_(map_idx_to_in_or_out)
    assert sum(in_out_targets) == sum(
        val_data.samples["class"].isin(cls_dropped_from_training)
    ), "mapping from all classes to known/unknown classes wrong"

    # max softmax/logit probability
    use_softmax = True
    if use_softmax:
        open_set_preds = torch.nn.Softmax(dim=1)(open_set_preds)
    open_set_preds = -open_set_preds.max(dim=1)[0]

    return in_out_targets, open_set_preds, all_logits, all_targets


def plot_roc_curve(fpr, tpr, aucroc):
    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % aucroc
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic - {DATASET_NAME} - {MODEL_NAME}")
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser("Testing arguments", add_help=False)
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        metavar="FILE",
        help="path to config file",
    )
    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    config.defrost()
    CLS_DROPPED_FROM_TRAINING = config.DATA.CLASSES_TO_DROP
    config.DATA.CLASSES_TO_DROP = []
    config.LOGGER.SAVE_TO_FILE = False
    config.GENERAL.MODE = "validation"
    config.freeze()
    dataset_path = "" if config.DATA.DATASET_PATH is None else config.DATA.DATASET_PATH
    DATASET_NAME = config.DATA.DATASET
    if config.DATA.DATASET_PATH is not None and "embedd" in config.DATA.DATASET_PATH:
        DATASET_NAME += "_embedded"
    if "birdnet" in config.MODEL.NAME:
        MODEL_NAME = "birdnet"
    elif "google-perch" in config.MODEL.NAME:
        MODEL_NAME = "google-perch"
    elif "aemnet" in config.MODEL.TYPE.lower():  # this is a stupid mistake
        MODEL_NAME = "aemnet"
    else:
        raise NotImplementedError

    df = {
        "she_aucroc": [],
        "she_fpr": [],
        "she_tpr": [],
        "msp_aucroc": [],
        "msp_fpr": [],
        "msp_tpr": [],
        "msp_oscr": [],
        "fold": [],
    }
    for f in range(1, config.TRAIN.KFOLDS + 1):
        # load dataset and edit to only contain testing and unknown individuals
        val_data, val_loader = get_val_data_and_loader(
            DATASET_NAME, CLS_DROPPED_FROM_TRAINING, config, device, f
        )

        # make and load saved model onto device
        config.defrost()
        config.DATA.NUM_CLASSES -= len(CLS_DROPPED_FROM_TRAINING)
        config.freeze()
        model = build_model(config)
        checkpoint_path = get_saved_model_path(
            config.MODEL.TYPE,
            config.MODEL.NAME,
            config.DATA.DATASET,
            config.DATA.CLASS,
            dataset_path,
            f,
        )
        checkpoint_path = os.path.join(config.GENERAL.CHECKPOINTS_DIR, checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        saved_model, model_cls_to_idx = checkpoint["model"], checkpoint["class_to_idx"]
        model.load_state_dict(saved_model)
        model.to(device)
        model.eval()

        # ############ Simplified Hopfield Energy
        # generate she patterns from correctly classified training data
        model.return_penult_feat_and_pred = True
        generate_she_patterns(
            model,
            model_cls_to_idx,
            DATASET_NAME,
            MODEL_NAME,
            CLS_DROPPED_FROM_TRAINING,
            f,
            config,
        )
        # calculate SHE scores (i.e. ood energy values), all_targets are returned as MODEL_IDX
        scores, labels = get_scores_and_labels(
            model, val_loader, DATASET_NAME, MODEL_NAME, model_cls_to_idx
        )
        she_aucroc, she_fpr, she_tpr = calculate_metrics(scores, labels)

        # ############# Maximum Softmax Probability
        model.return_penult_feat_and_pred = False
        (
            in_out_targets,
            open_set_preds,
            all_logits,
            all_targets,
        ) = embed_and_evaluate_data_using_msp(
            val_loader, model, model_cls_to_idx, CLS_DROPPED_FROM_TRAINING, device
        )
        msp_aucroc, msp_fpr, msp_tpr = calculate_metrics(open_set_preds, in_out_targets)
        msp_oscr, msp_fpr_ccr = calculate_oscr(all_logits, all_targets)

        df["she_aucroc"].append(she_aucroc)
        df["she_fpr"].append(she_fpr)
        df["she_tpr"].append(she_tpr)
        df["msp_aucroc"].append(msp_aucroc)
        df["msp_fpr"].append(msp_fpr)
        df["msp_tpr"].append(msp_tpr)
        df["msp_oscr"].append(msp_oscr)
        df["fold"].append(f)
    df["dataset"] = config.DATA.DATASET
    df["model"] = config.MODEL.NAME
    df = pd.DataFrame(df)
    # try:
    #     existing_df = pd.read_json(f"{config.GENERAL.FIGURES_DIR}/she_msp_outlier_results_5fold.json", orient="records", lines=True)
    # except FileNotFoundError:
    #     existing_df = pd.DataFrame()
    # # print("")
    # df = pd.concat([existing_df, df], axis=0, ignore_index=True)
    # df.to_json(f"{config.GENERAL.FIGURES_DIR}/she_msp_outlier_results_5fold.json", orient="records", lines=True)
