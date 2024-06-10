import torch
from transformers import BertTokenizer, Trainer, TrainingArguments,BertForSequenceClassification
from datasets import load_dataset
import numpy as np
import pandas as pd
from transformers import TrainerCallback, TrainerState, TrainerControl

load_dataset = load_dataset("imdb",split="train[:1%]")
df = pd.DataFrame(load_dataset)
print(df.head())

#load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

#encode the dataset
tokenized_datasets = load_dataset.map(tokenize_function, batched=True)

def simulated_human_feedback(predictions, labels):
    feedback = []
    for pred, label in zip(predictions, labels):
        if pred == label:
            feedback.append(1)  # Positive feedback for correct predictions
        else:
            feedback.append(-1)  # Negative feedback for incorrect predictions
    return feedback

# Load the model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
)

class HumanFeedbackCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        # Get predictions and labels
        predictions = np.argmax(kwargs['metrics']['eval_preds'], axis=1)
        labels = kwargs['metrics']['eval_labels']
        
        # Get human feedback
        feedback = simulated_human_feedback(predictions, labels)
        
        # Apply feedback to adjust model (simplified version)
        for i, (pred, label) in enumerate(zip(predictions, labels)):
            if feedback[i] == -1:
                # Here you can adjust the model parameters based on feedback
                # For simplicity, we'll just print the feedback
                print(f"Feedback: {feedback[i]}, Prediction: {pred}, Label: {label}")
        return control

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
    callbacks=[HumanFeedbackCallback],
)

# Train the model
trainer.train()