import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd


# Load dataset
file_path = r"C:\Users\ilsem\Documents\Thesis - memoire\dataset\real_sentiment_2700_900_per_class.xlsx"
df = pd.read_excel(file_path)


TEXT_COLUMN = "review_comment_message"
LABEL_COLUMN = "real sentiment"


# Map string labels to integers
label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {v: k for k, v in label2id.items()}
df["label"] = df[LABEL_COLUMN].map(label2id)
df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).fillna("")


# Split dataset to get the test set (same as training split)
from sklearn.model_selection import train_test_split
train_val_df, test_df = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=42)


# Dataset class for tokenization
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


# Load the saved model and tokenizer
save_dir = r"C:\Users\ilsem\Documents\Thesis - memoire\dataset\bert_trained_model"
tokenizer = AutoTokenizer.from_pretrained(save_dir)
model = AutoModelForSequenceClassification.from_pretrained(save_dir)


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # evaluation mode


# Prepare test DataLoader
test_loader = DataLoader(CommentDataset(test_df, tokenizer), batch_size=16, shuffle=False)


# Evaluate model
all_preds = []
all_labels = []


with torch.no_grad():
   for batch in test_loader:
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
