from transformers import AutoModelForSequenceClassification


def get_model(model_name: str, class_number: int, device: int | str):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=class_number
    ).to(device)
