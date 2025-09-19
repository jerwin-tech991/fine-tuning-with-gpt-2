from transformers import pipeline
import torch

# Initialize the embedding model on GPU if available
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'cuda' if device == 0 else 'cpu'}")

model_path = "./models/gpt2"

pipe = pipeline(
    task="text-generation,
    model=model_path,
    tokenizer=model_path,
    device=device,
)

input = "I like using JavaScript."

results = pipe(
    input,
    max_length=50,      # total tokens including prompt
    num_return_sequences=1,
    do_sample=True,
    temperature=0.7
)

# Print generated text
for i, result in enumerate(results):
    print(f"=== Generated Text {i+1} ===")
    print(result['generated_text'])

print(result)
