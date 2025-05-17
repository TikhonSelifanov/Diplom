!git clone https://github.com/nerel-ds/NEREL.git
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

def parse_ann_file(ann_path):
    entities = []
    relations = []
    with open(ann_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('T'):
                parts = line.split('\t')
                entity_id = parts[0]
                type_pos = parts[1].split(' ', 1)
                entity_type = type_pos[0]
                pos_str = type_pos[1]

                spans = []
                for span in pos_str.split(';'):
                    span_parts = span.strip().split()
                    if len(span_parts) >= 2:
                        try:
                            start = int(span_parts[0])
                            end = int(span_parts[1])
                            spans.append((start, end))
                        except ValueError as e:
                            print(f"Ошибка в спане '{span}' в файле {ann_path}: {e}")
                            continue
                if spans:
                    start, end = spans[0]
                    text = parts[2]
                    entities.append({
                        'id': entity_id,
                        'type': entity_type,
                        'start': start,
                        'end': end,
                        'text': text
                    })
            elif line.startswith('R'):
                parts = line.split('\t')
                rel_parts = parts[1].split()
                rel_type = rel_parts[0]
                head = rel_parts[1].split(':')[1]
                tail = rel_parts[2].split(':')[1]
                relations.append({
                    'type': rel_type,
                    'head': head,
                    'tail': tail
                })
    return entities, relations

def load_nerel(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            base_name = filename[:-4]
            txt_path = os.path.join(directory, filename)
            ann_path = os.path.join(directory, f"{base_name}.ann")
            if os.path.exists(ann_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                entities, relations = parse_ann_file(ann_path)
                data.append({
                    'text': text,
                    'entities': entities,
                    'relations': relations
                })
    return data

def create_target(example):
    entities = example['entities']
    relations = example['relations']

    entities.sort(key=lambda x: x['start'])
    entity_str = "; ".join([f"{e['start']}:{e['end']}:{e['type']}" for e in entities])

    id_to_index = {e['id']: i for i, e in enumerate(entities)}
    relation_str = "; ".join([f"{id_to_index[r['head']]} {r['type']} {id_to_index[r['tail']]}" for r in relations if r['head'] in id_to_index and r['tail'] in id_to_index])
    target = f"Entities: [{entity_str}]; Relations: [{relation_str}]"
    return target

def prepare_data(data):
    prepared = []
    for example in data:
        input_text = f"Extract entities and relations from the following text: {example['text']}"
        target_text = create_target(example)
        prepared.append({"input_text": input_text, "target_text": target_text})
    return Dataset.from_list(prepared)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_data = load_nerel('/kaggle/working/NEREL/NEREL-v1.1/train')
train_dataset = prepare_data(train_data)

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

model.to(device)

def process_example(example):
    input_text = example['input_text']
    target_text = example['target_text']
    input_tokens = tokenizer(input_text, max_length=512, truncation=True, add_special_tokens=False)['input_ids']
    target_tokens = tokenizer(target_text, max_length=512, truncation=True, add_special_tokens=False)['input_ids']
    eos_token_id = tokenizer.eos_token_id
    input_ids = input_tokens + [eos_token_id] + target_tokens + [eos_token_id]
    labels = [-100] * (len(input_tokens) + 1) + target_tokens + [eos_token_id]
    return {"input_ids": input_ids, "labels": labels}

train_dataset = train_dataset.map(process_example)

training_args = TrainingArguments(
    output_dir="./qwen3_ner_re",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=1000,
    learning_rate=5e-5,
    fp16=True,
    dataloader_num_workers=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

trainer.save_model("./qwen3_ner_re_model")
