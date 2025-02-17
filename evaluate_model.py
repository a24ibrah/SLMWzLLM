from transformers import Trainer

# Load the fine-tuned model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./fine-tuned-sentiment-model")
tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-sentiment-model")

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_datasets["test"],
)

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")