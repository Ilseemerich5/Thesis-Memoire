import pandas as pd


file_path = r"C:\Users\ilsem\Documents\Thesis - memoire\dataset\real_sentiment_2700_900_per_class.xlsx"


# Load the Excel file
df = pd.read_excel(file_path)
print("Dataset loaded successfully")
print(df.head())


import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
import os


# Suppress Hugging Face symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# Load dataset from Excel
file_path = r"C:\Users\ilsem\Documents\Thesis - memoire\dataset\real_sentiment_2700_900_per_class.xlsx"
df = pd.read_excel(file_path)
print("Dataset loaded successfully")
print(df.head())


# Define text and label columns
TEXT_COLUMN = "review_comment_message"
LABEL_COLUMN = "real sentiment"


# Map string labels to integers for BERT
label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {v: k for k, v in label2id.items()}
df["label"] = df[LABEL_COLUMN].map(label2id)


# Ensure text column is string type and handle missing values
df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).fillna("")


# Split dataset into train, validation, and test sets
train_val_df, test_df = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.1111, stratify=train_val_df["label"], random_state=42)


print("Dataset sizes")
print(f"Training: {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Testing: {len(test_df)}")


# Create PyTorch Dataset class for tokenization
class CommentDataset(Dataset):
   def __init__(self, df, tokenizer, max_len=128):
       self.df = df
       self.tokenizer = tokenizer
       self.max_len = max_len


   def __len__(self):
       return len(self.df)


   def __getitem__(self, idx):
       text = str(self.df.iloc[idx][TEXT_COLUMN])
       label = self.df.iloc[idx]["label"]
       encoding = self.tokenizer(
           text,
           add_special_tokens=True,
           truncation=True,
           max_length=self.max_len,
           padding="max_length",
           return_tensors="pt"
       )
       return {
           "input_ids": encoding["input_ids"].squeeze(0),
           "attention_mask": encoding["attention_mask"].squeeze(0),
           "labels": torch.tensor(label, dtype=torch.long)
       }


# Load BERTimbau tokenizer and model for sequence classification
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Using device:", device)


# Create DataLoaders for training, validation, and testing
BATCH_SIZE = 16
train_loader = DataLoader(CommentDataset(train_df, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(CommentDataset(val_df, tokenizer), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(CommentDataset(test_df, tokenizer), batch_size=BATCH_SIZE, shuffle=False)


# Set optimizer
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)


# Set number of epochs for training
EPOCHS = 8


# Training loop with progress bar
for epoch in range(EPOCHS):
   model.train()
   total_loss = 0
   loop = tqdm(train_loader, leave=True, desc=f"Training Epoch {epoch+1}")
   for batch in loop:
       optimizer.zero_grad()
       input_ids = batch["input_ids"].to(device)
       attention_mask = batch["attention_mask"].to(device)
       labels = batch["labels"].to(device)


       outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
       loss = outputs.loss
       total_loss += loss.item()
       loss.backward()
       optimizer.step()
       loop.set_postfix(loss=loss.item())


   avg_loss = total_loss / len(train_loader)
   print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")


# Evaluation on test set
model.eval()
all_preds = []
all_labels = []


with torch.no_grad():
   loop = tqdm(test_loader, leave=True, desc="Evaluating")
   for batch in loop:
       input_ids = batch["input_ids"].to(device)
       attention_mask = batch["attention_mask"].to(device)
       labels = batch["labels"].to(device)


       outputs = model(input_ids=input_ids, attention_mask=attention_mask)
       preds = torch.argmax(F.softmax(outputs.logits, dim=-1), dim=-1)
       all_preds.extend(preds.cpu().numpy())
       all_labels.extend(labels.cpu().numpy())


# Compute metrics
accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0,1,2])
conf_matrix = confusion_matrix(all_labels, all_preds)


# Print metrics
print(f"Accuracy: {accuracy:.4f}")
for i, label_name in id2label.items():
   print(f"Class {label_name}: Precision={precision[i]:.4f} Recall={recall[i]:.4f} F1-Score={f1[i]:.4f}")


print("Confusion Matrix:")
print(conf_matrix)


# Save trained model and tokenizer for later use
save_dir = r"C:\Users\ilsem\Documents\Thesis - memoire\dataset\bert_trained_model"
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"Model and tokenizer saved to {save_dir}")
