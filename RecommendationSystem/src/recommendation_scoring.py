#rec_sys
from geopy.distance import geodesic
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd


from config.db import connect_db, get_collection
from employee import Employee
from career import Career
from worksite import Worksite
from pymongo import MongoClient
from distance import label_to_value, calculate_distance, get_coordinates_employee,get_coordinates_worksites
from texteval import logits_to_probs, predict_with_ensemble_modified
from flask import Flask, request, render_template, redirect, url_for

# GPU 사용 가능 확인 및 device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 토크나이저 로딩
tokenizer_roberta = AutoTokenizer.from_pretrained("klue/roberta-large")
tokenizer_electra = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")


# 저장된 모델 로드
model_roberta = AutoModelForSequenceClassification.from_pretrained("Chamsol/klue-roberta-sentiment-classification")
model_electra = AutoModelForSequenceClassification.from_pretrained("Chamsol/koelctra-sentiment-classification")

app = Flask(__name__)

# MongoDB 데이터베이스 연결
db = connect_db()
collection = get_collection('Scoring')  # 원하는 컬렉션 이름을 지정



# 쿼리로 actual, applied 계산하기

# db_to_dict employee id당 dict으로 만들어서 people_info dict에 저장
career_collection = db['career']  # 컬렉션 이름으로 변경
employees_collection = db['employees']
worksite_collection = db['worksite']

query = {}  # 필요하다면 특정 조건을 여기에 추가
projection = {'sex': 1, 'user': 1, 'local': 1}
employee_docs = employees_collection.find(query, projection)

# 조회된 데이터를 기반으로 새로운 딕셔너리 생성
employees_info = []
for doc in employee_docs:
    employee_info = {
        'employee_id': str(doc['user']),  # ObjectId를 문자열로 변환
        'sex': doc['sex'],
        'employee_local': doc['local']
    }
    employees_info.append(employee_info)

worksites = worksite_collection.find({}, {'local': 1})

worksites_info = []
for worksite in worksites:
    worksite_info = {
        'worksite_local': worksite.get('local')
    }
    worksites_info.append(worksite_info)

def get_reviews():
    # 커리어 컬렉션의 모든 문서를 검색
    careers = career_collection.find()

    # reviews 딕셔너리 생성
    reviews = {}

    # 각 문서에서 review 필드를 추출하여 딕셔너리에 저장
    for career in careers:
        employee_id = str(career['employee'])
        reviews[employee_id] = career['review']

    return reviews


# 리뷰 데이터를 가져와서 딕셔너리로 저장
reviews_dict = get_reviews()

def count_work_days(employee_id):
    # employee_id에 해당하는 모든 Career 문서의 수를 카운팅하여 applied_work_days로 저장
    applied_work_days = career_collection.count_documents({'employee': ObjectId(employee_id)})

    # employee_id에 해당하면서 done: true인 문서의 수를 카운팅하여 actual_work_days로 저장
    actual_work_days = career_collection.count_documents({'employee': ObjectId(employee_id), 'done': True})

    # 결과를 딕셔너리 형태로 저장
    result = {
        'applied_work_days': applied_work_days,
        'actual_work_days': actual_work_days
    }

    return result


def db_to_dict(employee_id):
    # employee_id에 해당하는 직원 정보 검색
    employee_doc = employees_collection.find_one({'user': ObjectId(employee_id)}, {'sex': 1, 'local': 1})
    if not employee_doc:
        pass

    # 해당 직원의 근무 일수와 리뷰 정보 계산
    work_days_info = count_work_days(employee_id)
    review = reviews_dict.get(employee_id)

    # 직원이 근무하는 사이트 정보 검색
    career_doc = career_collection.find_one({'employee': ObjectId(employee_id)}, {'worksite': 1})
    if not career_doc:
        return 0
    worksite_doc = worksite_collection.find_one({'_id': career_doc['worksite']}, {'local': 1})

    # 최종 딕셔너리 생성
    people_dict = {
        'employee_id': employee_id,
        'employee_local': employee_doc['local'],
        'worksites_local': worksite_doc['local'],
        'sex': employee_doc['sex'],
        'actual_work_days': work_days_info['actual_work_days'],
        'applied_work_days': work_days_info['applied_work_days'],
        'review': review
    }

    return people_dict

# people_info 리스트 초기화
people_info = []

# 모든 직원 문서 검색
employee_docs = employees_collection.find({}, {'user': 1})

# 각 직원에 대해 db_to_dict 함수를 호출하여 people_info에 추가
for doc in employee_docs:
    employee_id = str(doc['user'])  # ObjectId를 문자열로 변환
    person_dict = db_to_dict(employee_id)
    if isinstance(person_dict, dict):  # 유효한 딕셔너리인지 확인
        people_info.append(person_dict)

#score


# calculate_score_for_person 함수 내에서의 수정
def calculate_score_for_person(person_info, model_roberta, model_electra, tokenizer_roberta, tokenizer_electra, device):
    # 각 변수의 가중치
    weight_distance = 0.2
    weight_attendance = 0.3
    weight_work_frequency = 0.2
    weight_label_value = 0.3
    # 성별에 따라 가중치 조정
    gender_weight = 1 if person_info['sex'] == '남자' or person_info['sex'] == '여자' else 0

    work_location = get_coordinates_worksites(person_info['worksites_local'])  # 주소가 없으면 None 반환)
    subway_station = get_coordinates_employee(person_info['employee_local'])


    # 현장과 지하철역의 거리 계산 (km 단위로 환산)
    distance_to_station = calculate_distance(work_location, subway_station)

    # 출석률 계산
    actual_work_days = person_info['actual_work_days']
    applied_work_days = person_info['applied_work_days']
    attendance_rate = actual_work_days / applied_work_days * 100

    # 출석률에 따라 점수 할당
    if attendance_rate >= 100:
        attendance_score = 50
    elif attendance_rate >= 90:
        attendance_score = 20
    else:
        attendance_score = 0

    # 근무횟수에 따라 점수 할당
    if person_info['actual_work_days'] >= 15:
        work_frequency_score = 50
    elif person_info['actual_work_days'] >= 10:
        work_frequency_score = 30
    elif person_info['actual_work_days'] >= 5:
        work_frequency_score = 20
    else:
        work_frequency_score = 0


    # 거리에 따라 점수 할당
    if distance_to_station < 10:
        distance_score = 50
    elif distance_to_station < 30:
        distance_score = 40
    elif distance_to_station < 50:
        distance_score = 30
    elif distance_to_station < 100:
        distance_score = 20
    elif distance_to_station < 200:
        distance_score = 10
    else:
        distance_score = 0


    # 텍스트 예측을 위한 코드 추가
    texts_to_predict = [person_info['review']]  # 텍스트를 리스트로 변환
    final_labels = predict_with_ensemble_modified(texts_to_predict, model_roberta, model_electra, tokenizer_roberta, tokenizer_electra, device)

    for label in final_labels:
      sentiment_score = label_to_value(label)




    # 각 변수에 가중치를 곱하여 합산된 점수를 계산
    total_score = (distance_score * weight_distance) + \
                  (attendance_score * weight_attendance) + \
                  (work_frequency_score * weight_work_frequency) + \
                  (sentiment_score * weight_label_value)

    return total_score*gender_weight


# 각 사람들의 점수 계산 및 MongoDB에 저장
people_info = [
    {'employee_id': 'id1234', 'employee_local': "성북구", 'worksites_local': "성북구", 'sex': '남자', 'actual_work_days': 20,
     'applied_work_days': 20, 'review': "앞으로 일을 맡겨도 좋을 사람임."},
    {'employee_id': 'id1235', 'employee_local': '고양시 일산서구', 'worksites_local': '성북구', 'sex': '여자',
     'actual_work_days': 10, 'applied_work_days': 10, 'review': "불성실하고 매우 필요없음 그냥 출근하지 않는게 나음"}
]

for person_info in people_info:
    score = calculate_score_for_person(person_info, model_roberta, model_electra, tokenizer_roberta, tokenizer_electra,device)  # 점수 계산 함수 호출
    employee_id = person_info['employee_id']

    # MongoDB에 저장
    update_result = career_collection.update_one(
        {'employee_id': employee_id},  # 찾을 조건
        {'$set': {'score': score}},  # 업데이트할 내용
        upsert=True  # 해당하는 문서가 없으면 새로 생성
    )

# 클라이언트 종료
client.close()
