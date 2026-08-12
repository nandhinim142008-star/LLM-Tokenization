import pandas as pd
from transformers import AutoTokenizer

# Load clean text dataset
df = pd.read_csv("dataset/clean_text.csv")

# Load pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the text
tokens = tokenizer(
    df["Clean_Text"].astype(str).tolist(),
    padding=True,
    truncation=True
)

print("Tokenization completed!")
print(tokens["input_ids"])