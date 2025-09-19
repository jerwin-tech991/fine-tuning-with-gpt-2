from transformers import GPT2ForSequenceClassification, GPT2Tokenizer
import torch

# Load fine-tuned model
checkpoint_path = "./test_trainer/checkpoint-500"
model_path = "./models/gpt2"  # original GPT-2 tokenizer path

model = GPT2ForSequenceClassification.from_pretrained(checkpoint_path)
model.eval()  # set to evaluation mode

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # GPT2 doesn't have a pad token by default

# Map numeric labels to sentiment
label_map = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

print("Sentiment analysis model ready! Type 'exit' to quit.")

while True:
    text = input("\nEnter text: ")
    if text.lower() in ["exit", "quit"]:
        print("Exiting...")
        break

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Forward pass
    with torch.no_grad(): # not to compute gradient.
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    predicted_label = label_map.get(predicted_class, "unknown")

    print(f"Predicted class number: {predicted_class}")
    print(f"Predicted sentiment: {predicted_label}")
    print(f"Score of each class: {logits.numpy()}")
