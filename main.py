"""
Kevin Kuipers (s5051150)
Federico Berdugo Morales (s5363268)
Nikolaos Skoufis (s5617804)
"""

from utilities.timer import TimeManager

with TimeManager("Imports"):
    import random

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer

    import error_analysis
    import model_training
    import models
    import preprocessing
    from evaluation import (
        display_key_metrics,
        evaluate_keyword_masking,
        evaluate_length_buckets,
    )
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
        max_size_dataframes = 100
        train_df, dev_df, test_df = preprocessing.preprocessing(
            RANDOM_SEED, max_size_dataframes
        )
        tokenizer = preprocessing.setup_tokenizer(model_name)
        train_df_tokens, dev_df_tokens, test_df_tokens = (
            preprocessing.tokenize_datasets(tokenizer, train_df, dev_df, test_df)
        )

    with TimeManager("Training_setup"):
        transformer_model = models.get_model(model_name, 4, device)
        trainer = model_training.generate_trainer(
            transformer_model, train_df_tokens, dev_df_tokens
        )

    with TimeManager("Training"):
        trainer.train()

    with TimeManager("Evaluation"), torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def tokenize_function(dataframe: pd.DataFrame):
            """Returns tokenized data"""
            return tokenizer(dataframe["text"], padding="max_length", truncation=True)

        test_dataset = Dataset.from_pandas(test_df[["text", "label"]])
        tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

        # Get predictions and true labels from the Trainer
        test_prediction_output = trainer.predict(tokenized_test_dataset)  # type: ignore
        test_logits = test_prediction_output.predictions
        y_real = test_prediction_output.label_ids  # These are the true labels

        # Convert logits to class predictions
        test_predictions = [np.argmax(logits).item() for logits in test_logits]

        # 1. Run Evaluation (Accuracy, F1, Confusion Matrix)
        display_key_metrics(y_real, test_predictions, f"Transformer ({model_name})")

        # 2. Run Error Analysis
        texts = test_df["text"].tolist()
        error_analysis.print_misclassified_examples(
            texts=texts,
            true_labels=y_real,
            predictions=test_predictions,
            model_name=f"Transformer ({model_name})",
            num_examples=10,
        )
        # 3. Run Slice Evaluation (Length Buckets)
        evaluate_length_buckets(
            texts=texts, true_labels=y_real, predictions=test_predictions
        )

        # 4. Run Slice Evaluation (Keyword Masking)
        # We need a small helper function to tokenize and predict new texts on the fly
        def predict_texts(new_texts):
            new_dataset = Dataset.from_dict({"text": new_texts})
            new_tokenized = new_dataset.map(tokenize_function, batched=True)
            new_output = trainer.predict(new_tokenized)
            return [np.argmax(logits).item() for logits in new_output.predictions]

        evaluate_keyword_masking(
            texts=texts, true_labels=y_real, model_pipeline_fn=predict_texts
        )

    if DEBUG_ENABLED:
        save_open_plots()


if __name__ == "__main__":
    with TimeManager("Program", True):
        main()

    # to keep plots open
    plt.show()
