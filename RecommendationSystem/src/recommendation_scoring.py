#rec_sys
from geopy.distance import geodesic
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd

from datasets import load_dataset
from datetime import datetime
from bson import ObjectId
from config.db import connect_db, get_collection
from employee import Employee
from career import Career
from worksites import Worksites
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

db = connect_db()

career_collection = get_collection('careers')
employee_collection = get_collection('employees')
worksite_collection = get_collection('worksites')

# career 컬렉션에서 모든 문서 읽기
careers1 = career_collection.find()

people_info = []

for careers in careers1:
    career_id = str(careers['_id'])
    employee_id = careers['employee']
    worksite_id = careers['worksite']

    employee = employee_collection.find_one({'_id': ObjectId(employee_id)})
    if employee is None:
        print(f"Employee with ID {employee_id} not found. Skipping this entry.")
        continue
    sex = employee['sex']
    employee_local = employee['local']

    worksite = worksite_collection.find_one({'_id': ObjectId(worksite_id)})
    if worksite is None:
        print(f"Worksite with ID {worksite_id} not found. Skipping this entry.")
        continue
    worksite_local = worksite['local']

    applied_work_days = career_collection.count_documents({'employee': employee_id})
    actual_work_days = career_collection.count_documents({'employee': employee_id, 'done': True})

    review = careers.get('review', "")

    result_dict = {
        'career_id': career_id,
        'employee_local': employee_local,
        'worksites_local': worksite_local,
        'sex': sex,
        'actual_work_days': actual_work_days,
        'applied_work_days': applied_work_days,
        'review': review
    }

    people_info.append(result_dict)

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

    if person_info['review'] == "":
        final_labels = [10]
    else:
        texts_to_predict = [person_info['review']]  # 텍스트를 리스트로 변환
        final_labels = predict_with_ensemble_modified(texts_to_predict, model_roberta, model_electra, tokenizer_roberta,
                                                      tokenizer_electra, device)

    for label in final_labels:
      sentiment_score = label_to_value(label)




    # 각 변수에 가중치를 곱하여 합산된 점수를 계산
    total_score = (distance_score * weight_distance) + \
                  (attendance_score * weight_attendance) + \
                  (work_frequency_score * weight_work_frequency) + \
                  (sentiment_score * weight_label_value)



    return total_score*gender_weight


for person_info in people_info:
    score = calculate_score_for_person(person_info, model_roberta, model_electra, tokenizer_roberta, tokenizer_electra, device)
    career_id = person_info['career_id']

    career_collection.update_one(
            {'_id': ObjectId(career_id)},
            {'$set': {'score': score}}
        )

updated_career_docs = career_collection.find()
for doc in updated_career_docs:
        print(doc)
else:
    print("Skipping data processing and score calculation due to DB connection issues.")