import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch.nn.functional as F
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import time

# nltk.download('punkt')

# 配置参数
MODEL_NAME = "SEBIS/code_trans_t5_small_commit_generation_transfer_learning_finetune"
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 5e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = ""
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")

class CommitMessageDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_source_len=512, max_target_len=30):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        diff = str(self.data.loc[idx, "diff"])
        commit_message = str(self.data.loc[idx, "commit_message"])

        inputs = self.tokenizer(
            diff,
            max_length=self.max_source_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        labels = self.tokenizer(
            commit_message,
            max_length=self.max_target_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = inputs.input_ids.squeeze()
        attention_mask = inputs.attention_mask.squeeze()
        labels = labels.input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

start_time = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelWithLMHead.from_pretrained(MODEL_NAME)
model.to(DEVICE)

train_dataset = CommitMessageDataset("train.csv", tokenizer)
val_dataset = CommitMessageDataset("val.csv", tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

def compute_bleu_score(references, predictions):
    smoothie = SmoothingFunction().method4
    bleu_scores = [
        sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie)
        for ref, pred in zip(references, predictions)
    ]
    return np.mean(bleu_scores)

def evaluate(model, val_loader, tokenizer):
    model.eval()
    references = []
    predictions = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=30, num_beams=10)
            decoded_preds = [tokenizer.decode(g, skip_special_tokens=True) for g in outputs]
            decoded_labels = [
                tokenizer.decode(
                    [token_id if token_id != -100 else tokenizer.pad_token_id for token_id in l.tolist()],
                    skip_special_tokens=True
                )
                for l in labels
            ]

            references.extend(decoded_labels)
            predictions.extend(decoded_preds)

    bleu_score = compute_bleu_score(references, predictions)
    return bleu_score

best_bleu = 0

for epoch in tqdm(range(EPOCHS)):
    model.train()
    train_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{EPOCHS}"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Training Loss: {avg_train_loss:.4f}")

    bleu_score = evaluate(model, val_loader, tokenizer)
    print(f"Validation BLEU Score: {bleu_score:.4f}")

    if bleu_score > best_bleu:
        best_bleu = bleu_score
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"New best model saved with BLEU Score: {best_bleu:.4f}")

end_time = time.time()

print(f"Training time: {end_time - start_time:.2f} seconds")
print(f"Training complete. Best BLEU Score: {best_bleu:.4f}")