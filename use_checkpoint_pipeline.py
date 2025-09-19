from transformers import pipeline

checkpoint_path = "./test_trainer/checkpoint-500"
model_path = "./models/gpt2"

pipe = pipeline(
    task="text-classification",
    model=checkpoint_path,
    tokenizer=model_path,
)

input = "I like using JavaScript."

result = pipe(input)

print(result)