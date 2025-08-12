import pandas as pd
import numpy as np
from hazm import WordTokenizer, stopwords_list
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict

# -------------------------------
# Step 1: Load and preprocess the data
# -------------------------------

# Function to assign labels based on numeric score ranges
def assign_label(score):
    if 0 < score < 20:
        return 'Low-quality-product'
    elif 20 < score < 60:
        return 'Medium-quality-product'
    else:
        return 'High-quality-product'

# Load CSV data
df = pd.read_csv('data.csv')

# Assign quality labels based on 'Score'
df['Label'] = df['Score'].apply(assign_label)

# Initialize Hazm tokenizer for Persian text
tokenizer_hazm = WordTokenizer()
stopwords = stopwords_list()

# Tokenize Persian text and remove stopwords
def preprocess_text(text):
    tokens = tokenizer_hazm.tokenize(text)
    filtered = [w for w in tokens if w not in stopwords]
    return ' '.join(filtered)

# Apply preprocessing to text data
df['clean_text'] = df['Text'].apply(preprocess_text)

# Encode labels into integers for model training
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['Label'])

# Split into train and test datasets (keeping label distribution the same)
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label_encoded'], random_state=42)

# Convert pandas DataFrames to HuggingFace Datasets
train_dataset = Dataset.from_pandas(train_df[['clean_text', 'label_encoded']])
test_dataset = Dataset.from_pandas(test_df[['clean_text', 'label_encoded']])

# Rename label column to 'labels' (HuggingFace convention)
train_dataset = train_dataset.rename_column("label_encoded", "labels")
test_dataset = test_dataset.rename_column("label_encoded", "labels")

# Combine into DatasetDict for convenience
raw_datasets = DatasetDict({'train': train_dataset, 'test': test_dataset})

# -------------------------------
# Step 2: Load ParsBERT tokenizer and model
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained('HooshvareLab/bert-base-parsbert-uncased')
model = AutoModelForSequenceClassification.from_pretrained('HooshvareLab/bert-base-parsbert-uncased', num_labels=3)

# -------------------------------
# Step 3: Tokenize datasets
# -------------------------------
def tokenize_function(examples):
    return tokenizer(examples['clean_text'], padding='max_length', truncation=True, max_length=128)

# Apply tokenization to all datasets
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# -------------------------------
# Step 4: Define training arguments
# -------------------------------
train_args = TrainingArguments(
    output_dir='./results',           # Directory to save model checkpoints
    eval_strategy='epoch',            # Evaluate after each epoch
    save_strategy='epoch',            # Save after each epoch
    learning_rate=2e-5,                # Recommended BERT fine-tuning LR
    per_device_train_batch_size=4,     # Training batch size per GPU/CPU
    per_device_eval_batch_size=4,      # Eval batch size per GPU/CPU
    num_train_epochs=1,                # Number of training epochs
    weight_decay=0.01,                  # Regularization
    logging_dir='./logs',               # Directory for TensorBoard logs
    logging_steps=10,                   # Log every 10 steps
    save_total_limit=2,                 # Keep only last 2 checkpoints
    load_best_model_at_end=True,        # Load best model based on eval metric
    metric_for_best_model='f1',         # Choose best model based on F1 score
    report_to='none'                    # Disable WandB logging
)

# -------------------------------
# Step 5: Define metric computation
# -------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    f1 = f1_score(labels, predictions, average='macro')
    accuracy = accuracy_score(labels, predictions)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# -------------------------------
# Step 6: Initialize Trainer
# -------------------------------
trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    compute_metrics=compute_metrics,
)

# -------------------------------
# Step 7: Fine-tune the model
# -------------------------------
trainer.train()

# -------------------------------
# Step 8: Save fine-tuned model
# -------------------------------
model.save_pretrained('./my_finetuned_model')
tokenizer.save_pretrained('./my_finetuned_tokenizer')

# -------------------------------
# Step 9 (Optional): Evaluate final model
# -------------------------------
metrics = trainer.evaluate()
print(metrics)
