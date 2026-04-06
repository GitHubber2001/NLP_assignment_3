from utilities.log import Logger

# AG News classes mapping (based on the standard dataset label order)
CLASS_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


def print_misclassified_examples(
    texts: list,
    true_labels,
    predictions: list,
    model_name="Model",
    num_examples=10,
):
    """
    Extracts and logs misclassified examples from the test set.
    """
    errors_found = 0

    Logger.log(f"\n{'=' * 60}")
    Logger.log(f" ERROR ANALYSIS: {model_name} (First {num_examples} Errors)")
    Logger.log(f"{'=' * 60}")

    for i in range(len(true_labels)):
        true_lbl = true_labels[i]
        pred_lbl = predictions[i]

        if true_lbl != pred_lbl:
            original_text = texts[i]

            true_label_str = CLASS_NAMES.get(true_lbl, f"Class {true_lbl}")
            pred_label_str = CLASS_NAMES.get(pred_lbl, f"Class {pred_lbl}")

            Logger.log(f"\n[Mismatch {errors_found + 1}]")
            Logger.log(f"True Label : {true_label_str}")
            Logger.log(f"Predicted  : {pred_label_str}")
            Logger.log(f"Text       : {original_text}")

            errors_found += 1

            # Stop once we hit the required number of examples
            if errors_found >= num_examples:
                Logger.log(f"\n{'=' * 60}\n")
                return

    # In case there are fewer errors than num_examples
    Logger.log(f"\n{'=' * 60}\n")
