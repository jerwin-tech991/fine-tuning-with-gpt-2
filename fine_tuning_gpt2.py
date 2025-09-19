from datasets import load_dataset
import pandas as pd
from transformers import GPT2Tokenizer
from transformers import GPT2ForSequenceClassification
import evaluate
import numpy as np
from transformers import TrainingArguments, Trainer

# dataset = load_dataset("./models/tweet_sentiment_extraction")
dataset = load_dataset("mteb/tweet_sentiment_extraction")
df = pd.DataFrame(dataset["train"])

tokenizer = GPT2Tokenizer.from_pretrained("./models/gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

model = GPT2ForSequenceClassification.from_pretrained("./models/gpt2", num_labels=3)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
   output_dir="test_trainer",
   #evaluation_strategy="epoch",
   per_device_train_batch_size=1,  # Reduce batch size here
   per_device_eval_batch_size=1,    # Optionally, reduce for evaluation as well
   gradient_accumulation_steps=4
)


trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=small_train_dataset,
   eval_dataset=small_eval_dataset,
   compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate()

