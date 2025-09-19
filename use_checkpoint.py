from transformers import GPT2ForSequenceClassification, GPT2Tokenizer
import torch

# Load from the final output directory
checkpoint_path = "./test_trainer/checkpoint-500"
model_path = "./models/gpt2"

model = GPT2ForSequenceClassification.from_pretrained(checkpoint_path)

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

text = "I don't mind."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits # logits = raw scores for each of your 3 labels.
predicted_class = torch.argmax(logits, dim=-1).item() # argmax picks the label with the highest score.

print("Predicted label:", predicted_class)


