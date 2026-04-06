import re

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def display_key_metrics(y_real, y_prediction, model_name: str) -> None:
    """Displays key evaluation metrics"""

    report = classification_report(y_real, y_prediction)

    cm = confusion_matrix(y_real, y_prediction)
    confusion_matrix_display_regression = ConfusionMatrixDisplay(confusion_matrix=cm)
    confusion_matrix_display_regression.plot()
    accuracy_classification_score = accuracy_score(y_real, y_prediction, normalize=True)

    print(f"{model_name}:\n {report}")

    print(
        f"{model_name}: Accuracy classificaion score: {accuracy_classification_score}"
    )

    plt.title(f"Confusion Matrix | {model_name}")
    plt.show(block=False)


def evaluate_length_buckets(texts: list, true_labels: list, predictions: list) -> None:
    """
    Evaluates model performance across different text length buckets.
    Splits data into Short (< 30 words), Medium (30-60 words), and Long (> 60 words).
    """
    print(f"\n{'=' * 60}")
    print(" SLICE EVALUATION 1: Length Buckets")
    print(f"{'=' * 60}")

    # Define our buckets
    buckets = {
        "Short (< 30 words)": {"y_true": [], "y_pred": []},
        "Medium (30 - 60 words)": {"y_true": [], "y_pred": []},
        "Long (> 60 words)": {"y_true": [], "y_pred": []},
    }

    # Sort each prediction into the correct bucket based on word count
    for text, true_lbl, pred_lbl in zip(texts, true_labels, predictions):
        word_count = len(text.split())

        if word_count < 30:
            bucket_name = "Short (< 30 words)"
        elif word_count <= 60:
            bucket_name = "Medium (30 - 60 words)"
        else:
            bucket_name = "Long (> 60 words)"

        buckets[bucket_name]["y_true"].append(true_lbl)
        buckets[bucket_name]["y_pred"].append(pred_lbl)

    # Calculate and print metrics per bucket
    print(f"{'Bucket':<25} | {'Count':<6} | {'Accuracy':<8} | {'Macro-F1':<8}")
    print("-" * 55)

    for name, data in buckets.items():
        y_t = data["y_true"]
        y_p = data["y_pred"]

        if len(y_t) > 0:
            acc = accuracy_score(y_t, y_p)
            f1 = f1_score(y_t, y_p, average="macro")
            print(f"{name:<25} | {len(y_t):<6} | {acc:.4f}   | {f1:.4f}")
        else:
            print(f"{name:<25} | {0:<6} | N/A        | N/A")

    print(f"{'=' * 60}\n")


def evaluate_keyword_masking(texts: list, true_labels: list, model_pipeline_fn) -> None:
    """
    Masks highly predictive keywords and re-evaluates to test model robustness.
    model_pipeline_fn is a function that takes a list of strings and returns predicted labels.
    """
    print(f"\n{'=' * 60}")
    print(" SLICE EVALUATION 2: Keyword Masking Probe")
    print(f"{'=' * 60}")

    # Words that strongly indicate Business, Sports, Sci/Tech, or World news
    keywords_to_mask = [
        "stock",
        "market",
        "economy",
        "shares",
        "investors",  # Business
        "baseball",
        "football",
        "game",
        "coach",
        "season",  # Sports
        "software",
        "microsoft",
        "internet",
        "computer",  # Sci/Tech
        "government",
        "iraq",
        "minister",
        "military",
        "un",  # World
    ]

    # Create a regex pattern to match these words (case-insensitive)
    pattern = re.compile(r"\b(" + "|".join(keywords_to_mask) + r")\b", re.IGNORECASE)

    masked_texts = []
    texts_changed = 0

    # Replace keywords with [MASK]
    for text in texts:
        masked_text, count = pattern.subn("[MASK]", text)
        masked_texts.append(masked_text)
        if count > 0:
            texts_changed += 1

    print(f"Masked keywords in {texts_changed} out of {len(texts)} articles.")

    # Get new predictions for the masked texts
    print("Running predictions on masked texts...")
    masked_predictions = model_pipeline_fn(masked_texts)

    # Calculate new metrics
    acc = accuracy_score(true_labels, masked_predictions)
    f1 = f1_score(true_labels, masked_predictions, average="macro")

    print(f"\nResults on Masked Data:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro-F1 : {f1:.4f}")
    print(f"{'=' * 60}\n")
