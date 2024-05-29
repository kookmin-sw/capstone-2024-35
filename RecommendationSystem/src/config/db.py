# rec_sys/config/db.py
import os
from dotenv import load_dotenv
from pymongo import MongoClient
import certifi

# 환경 변수 로드
load_dotenv()
mongodb_uri = os.getenv('MONGODB_URI')
ca = certifi.where()
# MongoDB 연결 설정
def connect_db():
    client = MongoClient(mongodb_uri, tlsCAFile=ca)
    db = client['Authusers']  # 데이터베이스 이름을 여기에서 변경 가능
    return db
def get_collection(collection_name):
    db = connect_db()
    return db[collection_name]