# db_store.py

from model import load_model_and_tokenizer, predict_entities
from data_processing import find_career_status, find_phone_number, extract_and_combine_entities
from datasets import load_dataset
from config.db import connect_db, get_collection
from employee import Employee, EmployeeRepository
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)

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
#text = "송문선 / 24/서초구 거주/경력은 사무실 철거 해봤습니다."

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        if text:
            combine(text)
            return redirect(url_for('index'))
    return render_template('index.html')



def combine(text):
    predicted_entities = predict_entities(text, model, tokenizer, id2tag)
    entities_combined = extract_and_combine_entities(predicted_entities)
    entities_combined["career"] = find_career_status(text)
    entities_combined["phonenumber"] = "010-0000-0000"
    entities_combined["sex"] = "남"
    entities_combined["RRN"] = "000000-0000000"
    entities_combined["name"] = entities_combined["name"].replace(' ', '')
    entities_combined["name"] = entities_combined["name"].replace(',', '')
    entities_combined["age"] = entities_combined["age"].replace(' ', '')
    entities_combined["age"] = entities_combined["age"].replace(',', '')
    entities_combined["local"] = entities_combined["local"].replace(' ', '')
    entities_combined["local"] = entities_combined["local"].replace(',', '')
    user_id = '609b8b8f8e4f5b88f8e8e8e8'
    new_employee = Employee(
        user=user_id,
        name=entities_combined["name"],
        sex=entities_combined["sex"],
        local=entities_combined["local"],
        rrn=entities_combined["RRN"],
        career = entities_combined["career"],
        age = entities_combined["age"]
        )
    #데이터 MongoDB에 저장
    employee_repo = EmployeeRepository()
    employee_repo.insert(new_employee)
    print(entities_combined)

@app.route('/employees', methods=['GET'])
def list_employees():
    employee_repo = EmployeeRepository()
    employees = employee_repo.find_all()
    return render_template('employees.html', employees=employees)

if __name__ == '__main__':
    app.run(debug=True)
'''
while(1):
    text = input('메시지 내용 : ')
    if text == 'quit':
        break
    combine(text)

employee_repo = EmployeeRepository()
for employee in employee_repo.find_all():
        print(employee)

'''
