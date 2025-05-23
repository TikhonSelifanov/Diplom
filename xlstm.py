import os
import glob
import json
import pandas as pd
from razdel import sentenize, tokenize
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)
from sklearn.metrics import f1_score
import numpy as np
import pymorphy3
from nltk.corpus import stopwords
import nltk

!pip install git+https://github.com/NX-AI/xlstm.git
!pip install razdel transformers datasets torch scikit-learn pymorphy2 nltk

nltk.download('stopwords')

!git clone https://github.com/nerel-ds/NEREL.git

morph = pymorphy3.MorphAnalyzer()
stop_words = set(stopwords.words('russian'))

tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
vocab_size = tokenizer.vocab_size
max_length = 256

df = pd.read_excel("/kaggle/input/rbc-news-f/rbc_news_firts_week.xlsx")


def preprocess_text(text):
    tokens = [token.text for token in tokenize(text)]
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token not in stop_words and token.isalnum()]
    tokens = [morph.parse(token)[0].normal_form for token in tokens]
    return ' '.join(tokens)

df['preprocessed_text'] = df['text'].apply(preprocess_text)

def parse_ann(ann_file):
    entities = []
    with open(ann_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('T'):
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    _, type_start_end, _ = parts
                    type_start_end = type_start_end.split()
                    if len(type_start_end) == 3:
                        entity_type = type_start_end[0]
                        start = int(type_start_end[1])
                        end = int(type_start_end[2])
                        entities.append({"start": start, "end": end, "type": entity_type})
    return entities

def load_nerel_data(split_dir):
    all_input_ids = []
    all_tags = []
    txt_files = glob.glob(os.path.join(split_dir, "*.txt"))
    for txt_file in txt_files:
        base_name = os.path.basename(txt_file).replace(".txt", "")
        ann_file = os.path.join(split_dir, f"{base_name}.ann")

        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        entities = parse_ann(ann_file)

        sentences = list(sentenize(text))
        for sent in sentences:
            sent_text = sent.text
            sent_start = sent.start
            sent_end = sent.stop

            sent_entities = [ent for ent in entities if ent["start"] < sent_end and ent["end"] > sent_start]
            adjusted_entities = []
            for ent in sent_entities:
                adj_start = max(ent["start"] - sent_start, 0)
                adj_end = min(ent["end"] - sent_start, len(sent_text))
                if adj_start < adj_end:
                    adjusted_entities.append({"start": adj_start, "end": adj_end, "type": ent["type"]})

            encoding = tokenizer.encode_plus(
                sent_text,
                return_offsets_mapping=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].squeeze(0).tolist()
            offset_mapping = encoding['offset_mapping'].squeeze(0).tolist()

            tags = ['O'] * len(input_ids)
            entity_dict = {(e['start'], e['end']): e['type'] for e in adjusted_entities}
            for i, (token_start, token_end) in enumerate(offset_mapping):
                if token_start == token_end:
                    continue
                for (ent_start, ent_end), ent_type in entity_dict.items():
                    if token_start >= ent_start and token_end <= ent_end:
                        if token_start == ent_start:
                            tags[i] = 'B-' + ent_type
                        else:
                            tags[i] = 'I-' + ent_type
                        break

            all_input_ids.append(input_ids)
            all_tags.append(tags)
    return all_input_ids, all_tags

train_dir = '/kaggle/working/NEREL/NEREL-v1.1/train'
dev_dir = '/kaggle/working/NEREL/NEREL-v1.1/dev'
test_dir = '/kaggle/working/NEREL/NEREL-v1.1/test'

train_input_ids, train_tags = load_nerel_data(train_dir)
dev_input_ids, dev_tags = load_nerel_data(dev_dir)
test_input_ids, test_tags = load_nerel_data(test_dir)

all_tags_flat = [tag for sent_tags in train_tags + dev_tags + test_tags for tag in sent_tags if tag != 'O']
unique_tags = list(set(all_tags_flat))
tag2id = {tag: idx + 1 for idx, tag in enumerate(unique_tags)}
tag2id['O'] = 0
tag2id['[PAD]'] = len(tag2id)
num_tags = len(tag2id)

def tags_to_ids(all_tags_list):
    return [[tag2id.get(tag, tag2id['[PAD]']) for tag in sent_tags] for sent_tags in all_tags_list]

train_tags_id = tags_to_ids(train_tags)
dev_tags_id = tags_to_ids(dev_tags)
test_tags_id = tags_to_ids(test_tags)

class NERDataset(Dataset):
    def __init__(self, input_ids_list, tags_list):
        self.input_ids_list = input_ids_list
        self.tags_list = tags_list

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids_list[idx], dtype=torch.long),
            'labels': torch.tensor(self.tags_list[idx], dtype=torch.long)
        }

train_dataset = NERDataset(train_input_ids, train_tags_id)
dev_dataset = NERDataset(dev_input_ids, dev_tags_id)
test_dataset = NERDataset(test_input_ids, test_tags_id)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

class NERxLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_tags, max_length=256):
        super(NERxLSTM, self).__init__()

        mlstm_config = mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=4,
                qkv_proj_blocksize=4,
                num_heads=4
            )
        )

        slstm_config = sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend="cuda" if torch.cuda.is_available() else "cpu",
                num_heads=4,
                conv1d_kernel_size=4,
                bias_init="powerlaw_blockdependent",
            ),
            feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
        )

        config = xLSTMBlockStackConfig(
            mlstm_block=mlstm_config,
            slstm_block=slstm_config,
            context_length=max_length,
            num_blocks=7,
            embedding_dim=embedding_dim,
            slstm_at=[1],
        )

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.xlstm = xLSTMBlockStack(config)
        self.classifier = nn.Linear(embedding_dim, num_tags)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        outputs = self.xlstm(embedded)
        logits = self.classifier(outputs)
        return logits

embedding_dim = 128
max_length = 256

model = NERxLSTM(vocab_size, embedding_dim, num_tags, max_length)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss(ignore_index=tag2id['[PAD]'])

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = loss_fn(outputs.view(-1, num_tags), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids)
            preds = torch.argmax(outputs, dim=2)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    predictions_flat = [item for sublist in predictions for item in sublist]
    true_labels_flat = [item for sublist in true_labels for item in sublist]
    mask = np.array(true_labels_flat) != tag2id['[PAD]']
    predictions_flat = np.array(predictions_flat)[mask]
    true_labels_flat = np.array(true_labels_flat)[mask]
    f1 = f1_score(true_labels_flat, predictions_flat, average='weighted')
    print(f"Dev F1-score: {f1}")

torch.save(model.state_dict(), './ner_xlstm_model.pth')

id2tag = {v: k for k, v in tag2id.items()}

first_text = df['text'].iloc[0]
first_preprocessed = df['preprocessed_text'].iloc[0]

sentences = list(sentenize(first_text))

model.eval()

print("Первый текст:")
print(first_text)
print("\nПредобработанный текст:")
print(first_preprocessed)
print("\nПредсказания NER:")

for sent in sentences:
    sent_text = sent.text
    encoding = tokenizer.encode_plus(
        sent_text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)

    with torch.no_grad():
        logits = model(input_ids)

    predictions = torch.argmax(logits, dim=2).cpu().numpy()[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())

    token_len = len(tokenizer.encode(sent_text, add_special_tokens=True))
    tokens = tokens[:token_len]
    predictions = predictions[:token_len]
    pred_tags = [id2tag[pred] for pred in predictions]

    print(f"\nПредложение: {sent_text}")
    for token, tag in zip(tokens, pred_tags):
        print(f"{token}: {tag}")
