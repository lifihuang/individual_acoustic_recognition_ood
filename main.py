import argparse

import torch

from config import get_config
from experiments import select_experiment

if __name__ == "__main__":
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    experiment = select_experiment(config, device)

    if config.GENERAL.MODE == "train":
        experiment.train()
    elif config.GENERAL.MODE == "test":
        experiment.test()
    else:
        raise ValueError(
            f"'mode' parameter should be either 'test' or 'train', currently: {config.GENERAL.MODE}"
        )
