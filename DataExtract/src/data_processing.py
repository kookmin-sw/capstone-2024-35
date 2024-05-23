import re

def find_career_status(text):
    pattern = r'(경력\s*(:|은|는|이|가| )?\s*[^,\.]*)'
    match = re.search(pattern, text)
    if match:
        full_text = match.group(1).strip()
        # '유' 또는 '있'으로 시작하는 경우 '유' 반환
        if full_text.startswith('유') or full_text.startswith('있'):
            return '유'
        # '무' 또는 '없'으로 시작하는 경우 '무' 반환
        elif full_text.startswith('무') or full_text.startswith('없'):
            return '무'
        else:
            # 조건에 맞지 않는 경우 전체 텍스트 반환
            return full_text
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
