from transformers import PreTrainedModel, Trainer, TrainingArguments


def generate_trainer(
    model_name: PreTrainedModel,
    training_set,
    evaluation_set,
    eval_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
):
    training_args = TrainingArguments(
        eval_strategy=eval_strategy,
        logging_strategy=logging_strategy,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
    )

    trainer = Trainer(
        model=model_name,
        args=training_args,
        train_dataset=training_set,
        eval_dataset=evaluation_set,
    )
    return trainer
