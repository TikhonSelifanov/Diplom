import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

combined_prompt_template = """
Для данного текста выполни следующие шаги:
1. Найди все упоминания персон (PERSON), организаций (ORGANIZATION) и локаций (LOCATION). Другие типы сущностей не указывай
2. Перечисли сущности в формате:
Сущности:
- PERSON: [имя]
- ORGANIZATION: [название]
- LOCATION: [место]
3. Определи отношения между этими сущностями и запиши их в виде троек: [сущность1] [отношение] [сущность2]
4. Перечисли тройки в разделе:
Тройки отношений:
- [сущность1] [отношение] [сущность2]
Ответ должен содержать только одну секцию "Сущности" и одну секцию "Тройки отношений". Не добавляй лишний текст или повторы

Пример 1
Текст: Владимир Путин встретился с представителями Газпрома в Москве
Сущности:
- PERSON: Владимир Путин
- ORGANIZATION: Газпром
- LOCATION: Москва
Тройки отношений:
- Владимир Путин встретился с Газпром
- Владимир Путин находился в Москве

Теперь для этого текста: {text}
"""

def query_model(prompt, max_length=1024):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    response = response[len(prompt):].strip()
    return response

def clean_response(response):
    lines = response.split('\n')
    entities_section = []
    relations_section = []
    current_section = None

    for line in lines:
        line = line.strip()
        if line.startswith('Сущности:'):
            current_section = 'entities'
            continue
        elif line.startswith('Тройки отношений:'):
            current_section = 'relations'
            continue
        elif line and current_section == 'entities' and line not in entities_section:
            entities_section.append(line)
        elif line and current_section == 'relations' and line not in relations_section:
            relations_section.append(line)

    cleaned_response = "Сущности:\n" + '\n'.join(entities_section) + "\nТройки отношений:\n" + '\n'.join(relations_section)
    return cleaned_response.strip()

def extract_ner_and_relations(text):
    prompt = combined_prompt_template.format(text=text)
    response = query_model(prompt)
    cleaned_response = clean_response(response)
    return cleaned_response

t = 'Взрыв газа произошел в одной из квартир пятиэтажки на улице Луначарского в Петрозаводске, по предварительным данным, пострадали три человека, сообщили «РИА Новости» в республиканском управлении МЧС.'
first_text = t

df = pd.DataFrame({'text': [first_text]})

df['ner_and_re_results'] = pd.Series(dtype='object')
df.at[0, 'ner_and_re_results'] = extract_ner_and_relations(first_text)

print("NER и Результаты извлечения отношений для текста:")
print(f"Текст: {first_text}")
print(f"Результаты:\n{df['ner_and_re_results'].iloc[0]}")
