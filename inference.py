from transformers import pipeline

# Load the fine-tuned model for inference
sentiment_analyzer = pipeline("text-classification", model="./fine-tuned-sentiment-model")

# Test the model
text = "This movie was absolutely fantastic!"
result = sentiment_analyzer(text)
print(result)  # Output: [{'label': 'positive', 'score': 0.99}]