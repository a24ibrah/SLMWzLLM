from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("csv", data_files="sentiment_dataset.txt", delimiter="\t", column_names=["text", "label"])

# Map sentiment labels to integers
label_map = {"positive": 0, "negative": 1, "neutral": 2}
dataset = dataset.map(lambda x: {"label": label_map[x["label"]]})

# Split the dataset into training and validation sets
dataset = dataset["train"].train_test_split(test_size=0.2)

# Load the tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
  
# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine-tuned-sentiment-model")
tokenizer.save_pretrained("./fine-tuned-sentiment-model")
print("Model fine-tuned and saved to fine-tuned-sentiment-model!")