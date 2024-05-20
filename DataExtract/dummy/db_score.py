# db_store.py

from model import load_model_and_tokenizer, predict_entities
from data_processing import find_career_status, find_phone_number, extract_and_combine_entities
from datasets import load_dataset
from config.db import connect_db, get_collection

# MongoDB 데이터베이스 연결
db = connect_db()
collection = db['ExtractedEntities']  # 원하는 컬렉션 이름을 지정

# KLUE NER 데이터셋 로드
dataset = load_dataset("klue", "ner")
tag_list = dataset['train'].features['ner_tags'].feature.names
tag2id = {tag: id for id, tag in enumerate(tag_list)}
id2tag = {id: tag for tag, id in tag2id.items()}

# 모델 및 토크나이저 로드
model, tokenizer = load_model_and_tokenizer()

# 예시 텍스트
text = "25/ 김준호 /서초구 거주/경력:유/전화번호:010-0000-0000"

# 엔티티 추출 및 결합
predicted_entities = predict_entities(text, model, tokenizer, id2tag)
entities_combined = extract_and_combine_entities(predicted_entities)
entities_combined["career"] = find_career_status(text)
entities_combined["phonenumber"] = find_phone_number(text)
entities_combined["sex"] = "남"
entities_combined["RRN"] = "000000-0000000"

# 데이터 MongoDB에 저장
insert_result = collection.insert_one(entities_combined)
print(f"삽입된 문서 ID: {insert_result.inserted_id}")
