from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_path = "./models/gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_path)

while True:
    text = input("\nEnter text: ")
    if text.lower() in ["exit", "quit"]:
        print("Exiting...")
        break

    inputs = tokenizer(text, return_tensors="pt")

    # Generate 50 new tokens
    outputs = model.generate(
        **inputs,
        max_length=100,       # how long the generated text will be. total tokens including prompt.
        num_return_sequences=1, # generate multiple outputs.
        do_sample=True,       # use sampling instead of greedy
        temperature=0.7       # randomness in generation. higher = more random.
    )

    # Decode generated tokens
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)
