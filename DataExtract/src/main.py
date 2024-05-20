# db_store.py

from model import load_model_and_tokenizer, predict_entities
from data_processing import find_career_status, find_phone_number, extract_and_combine_entities
from datasets import load_dataset
from config.db import connect_db, get_collection
from employee import Employee, EmployeeRepository

# MongoDB 데이터베이스 연결
db = connect_db()
collection = get_collection('ExtractedEntities')  # 원하는 컬렉션 이름을 지정

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

user_id = '609b8b8f8e4f5b88f8e8e8e8'

new_employee = Employee(
    user=user_id,
    name=entities_combined["name"],
    sex=entities_combined["sex"],
    local=entities_combined["local"],
    rrn=entities_combined["RRN"],
    phonenumber=entities_combined["phonenumber"],
    age = entities_combined["age"]
    )
# 데이터 MongoDB에 저장
employee_repo = EmployeeRepository()
employee_repo.insert(new_employee)

for employee in employee_repo.find_all():
    print(employee)
