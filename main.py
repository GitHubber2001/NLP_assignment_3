"""
Kevin Kuipers (s5051150)
Federico Berdugo Morales (s5363268)
Nikolaos Skoufis (s5617804)
"""

from utilities.timer import TimeManager

with TimeManager("Torch imports"):
    import torch

import random

import matplotlib.pyplot as plt
import numpy as np

import error_analysis
import model_training
import models
import preprocessing
from evaluation import display_key_metrics
from utilities.debug import DEBUG_ENABLED
from utilities.plots import save_open_plots

# fixed random seed
RANDOM_SEED = 42


def set_deterministic_behaviour(random_seed):
    """Sets deterministic behaviour of the program"""

    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_accelerator_device() -> str:
    """Returns accelerator device for boosting performance"""

    current_accelerator = torch.accelerator.current_accelerator(True)
    if current_accelerator is not None:
        device = current_accelerator.type
    else:
        device = "cpu"

    print(f"Using device {device}")

    return device


def main() -> None:
    set_deterministic_behaviour(RANDOM_SEED)
    device = get_accelerator_device()
    model_name = "distilbert-base-uncased"
    print(f"Using model {model_name}")

    with TimeManager("Split"):
        train_df, dev_df, test_df = preprocessing.preprocessing(RANDOM_SEED)
        tokenizer = preprocessing.setup_tokenizer(model_name)
        train_df_tokens, dev_df_tokens, test_df_tokens = (
            preprocessing.tokenize_datasets(
                tokenizer, train_df, dev_df, test_df
            )
        )

    with TimeManager("Training_setup"):
        model = models.get_model(model_name, 4, device)
        trainer = model_training.generate_trainer(
            model, train_df_tokens, dev_df_tokens
        )

    with TimeManager("Training"):
        trainer.train()

    with TimeManager("Evaluation"):
        results = trainer.evaluate()
        print(results)

    plt.figure()
    plt.hist([1, 2, 3, 4], "auto")
    plt.title("Title")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show(block=False)

    if DEBUG_ENABLED:
        save_open_plots()


if __name__ == "__main__":
    with TimeManager("Program", True):
        main()

    # to keep plots open
    plt.show()
