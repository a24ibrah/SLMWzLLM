import openai

# Set up OpenAI API
openai.api_key = "your_openai_api_key"

# Define a prompt for generating sentiment analysis data
prompt = """Generate 10 examples of sentences with their corresponding sentiment labels (positive, negative, or neutral). Format each example as: "sentence | sentiment". Do not include any additional text."""

# Call the LLM to generate data
response = openai.Completion.create(
    engine="gpt-4",
    prompt=prompt,
    max_tokens=500,
    temperature=0.7
)

# Extract and parse the generated data
generated_data = response.choices[0].text.strip().split("\n")
dataset = [example.split(" | ") for example in generated_data if " | " in example]

# Save the dataset to a file
with open("sentiment_dataset.txt", "w") as f:
    for sentence, sentiment in dataset:
        f.write(f"{sentence}\t{sentiment}\n")

print("Dataset generated and saved to sentiment_dataset.txt!")