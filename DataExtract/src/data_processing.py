import re

def find_career_status(text):
    pattern = r'경력\s*:\s*(유|무|없|있)'
    match = re.search(pattern, text)
    if match:
        raw_career = match.group(1)
        return '무' if raw_career in ['없'] else '유'
    return "경력 유무를 찾을 수 없습니다."

def find_phone_number(text):
    pattern = r'010-?\d{4}-?\d{4}'
    match = re.search(pattern, text)
    return match.group() if match else "전화번호를 찾을 수 없습니다."

def extract_and_combine_entities(predicted_entities):
    entity_info = {"name": "", "local": "", "age": ""}
    for token, tag in predicted_entities:
        if tag == 'B-PS':
            entity_info["name"] += token.strip()
        elif tag == 'B-LC':
            entity_info["local"] += token.strip() if token != " " else token
        elif tag in ['B-QT', 'B-DT']:
            entity_info["age"] = token.strip()
    return entity_info
