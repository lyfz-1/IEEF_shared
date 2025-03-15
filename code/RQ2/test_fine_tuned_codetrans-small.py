import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelWithLMHead
from tqdm import tqdm
import time

MODEL_DIR = ""
TEST_FILE = "test.csv"
OUTPUT_FILE = "output.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 30
BATCH_SIZE = 16

start_time = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelWithLMHead.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()


class TestDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=512):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        diff = str(self.data.loc[idx, "diff"])
        inputs = self.tokenizer(
            diff,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.squeeze()
        attention_mask = inputs.attention_mask.squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

test_dataset = TestDataset(TEST_FILE, tokenizer, max_length=MAX_INPUT_LENGTH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

with open(OUTPUT_FILE, "w") as f:
    for batch in tqdm(test_loader, desc="Generating commit messages"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=MAX_OUTPUT_LENGTH,
                num_beams=10,
                early_stopping=True
            )

        generated_messages = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        for message in generated_messages:
            f.write(message)
            f.write("\n")

end_time = time.time()

print(f"Test time cost: {end_time - start_time:.2f}s")
print(f"Test done. The results are saved to: {OUTPUT_FILE}")