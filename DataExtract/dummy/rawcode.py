from transformers import BertTokenizerFast, BertForTokenClassification, AdamW
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from kobert_tokenizer import KoBERTTokenizer
from kobert_transformers import get_kobert_model, get_tokenizer
from transformers import BertForTokenClassification

# KLUE NER 데이터셋 로드
dataset = load_dataset("klue", "ner")

# 태그 리스트 확인
tag_list = dataset['train'].features['ner_tags'].feature.names
print(tag_list)

# tag2id 및 id2tag 사전 생성
tag2id = {tag: id for id, tag in enumerate(tag_list)}
id2tag = {id: tag for tag, id in tag2id.items()}

model_name = "mmoonssun/klue_ner_kobert"
model = BertForTokenClassification.from_pretrained(model_name, num_labels=13)  # num_labels는 데이터셋의 라벨 수에 맞춰 조정
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

import re

def predict_entities(text, model, tokenizer, id2tag):
    # GPU 사용 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 평가 모드로 설정
    model.eval()

    # 입력 문장 토크나이징 및 텐서로 변환
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # 예측 수행
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # 예측 결과에서 가장 높은 확률을 가진 태그 ID를 추출
    predictions = torch.argmax(logits, dim=2)

    # ID를 태그로 변환
    predicted_tags = [id2tag[id.item()] for id in predictions[0]]

    # 토큰화된 텍스트와 예측된 태그 결합
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    token_tag_pairs = [(token, tag) for token, tag in zip(tokens, predicted_tags) if token not in ["[CLS]", "[SEP]", "[PAD]", "<pad>"]]

    # '▁' 문자를 공백으로 대체하여 보다 자연스러운 출력을 생성
    token_tag_pairs = [(token.replace('▁', ' '), tag) for token, tag in token_tag_pairs]

    return token_tag_pairs

def find_career_status(text):
    # '경력' 다음에 오는 '유', '무', '없', '있' 찾기
    pattern = r'경력\s*:\s*(유|무|없|있)'

    # 문자열에서 패턴에 해당하는 부분 찾기
    match = re.search(pattern, text)

    # 찾은 값을 변수에 저장하고 처리
    if match:
        raw_career = match.group(1)  # 첫 번째 그룹(유|무|없|있)을 추출
        # '없'이나 '있'을 각각 '무', '유'로 변환
        if raw_career == '없':
            career = '무'
        elif raw_career == '있':
            career = '유'
        else:
            career = raw_career

        return career
    else:
        return "경력 유무를 찾을 수 없습니다."

def find_phone_number(text):
    # 정규 표현식으로 전화번호 패턴 찾기
    # 패턴 설명: '010'으로 시작하며, '-'가 있을 수도 있고 없을 수도 있으며, 숫자가 연속으로 나타남
    pattern = r'010-?\d{4}-?\d{4}'

    # 문자열에서 패턴에 해당하는 부분 찾기
    match = re.search(pattern, text)

    # 찾은 전화번호를 변수에 저장하고 출력
    if match:
        phone_number = match.group()
        return phone_number
    else:
        return "전화번호를 찾을 수 없습니다."

def extract_and_combine_entities(predicted_entities):
    name = ""
    location = ""
    age = ""

    for token, tag in predicted_entities:
        if tag == 'B-PS':  # 이름 추출
            name += token.strip()
        elif tag == 'B-LC':  # 위치 추출
            if token == " ":
                location += token
            else:
                location += token.strip()
        elif tag == 'B-QT' or tag == 'B-DT':  # 나이(수량) 추출
            age = token.strip()

    # 결과 반환
    return {"name": name, "location": location, "age": age}

# 예시 문장
text = "25/ 김준호 /서초구 거주/경력:유/전화번호:010-0000-0000"

# 모델을 사용하여 문장에서 개체 추출
predicted_entities = predict_entities(text, model, tokenizer, id2tag)
career = find_career_status(text)
phone_number = find_phone_number(text)

# 함수를 사용하여 변수에 저장
entities_combined = extract_and_combine_entities(predicted_entities)

# 결과 출력
entities_combined["career"] = career
entities_combined["phone_number"] = phone_number

print(entities_combined)
