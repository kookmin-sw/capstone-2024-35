# employee.py

from datetime import datetime
from bson import ObjectId
from config.db import get_collection

class Employee:
    def __init__(self, user, name, sex, local, rrn,career,age, created_at=None, updated_at=None):
        self.user = ObjectId(user)
        self.name = name
        self.sex = sex
        self.local = local
        self.rrn = rrn  # 주민등록번호
        self.age = age
        self.career = career
        self.created_at = created_at if created_at else datetime.utcnow()
        self.updated_at = updated_at if updated_at else datetime.utcnow()

    def to_dict(self):
        return {
            'user': self.user,
            'name': self.name,
            'sex': self.sex,
            'local': self.local,
            'RRN': self.rrn,
            'age': self.age,
            'career': self.career,
            #'phonenumber': self.phonenumber,
            'createdAt': self.created_at,
            'updatedAt': self.updated_at
        }

class EmployeeRepository:
    """
    Employee 데이터를 관리하는 저장소 클래스.
    """
    def __init__(self):
        self.collection = get_collection('employees')

    def insert(self, employee: Employee):
        """
        새로운 Employee를 삽입합니다.
        """
        self.collection.insert_one(employee.to_dict())

    def find_all(self):
        """
        모든 Employee 데이터를 반환합니다.
        """
        return list(self.collection.find())

    def find_by_name(self, name):
        """
        이름으로 Employee 데이터를 찾습니다.
        """
        return list(self.collection.find({'name': name}))

    def update(self, employee_id, updated_fields):
        """
        주어진 Employee ID의 데이터를 업데이트합니다.
        """
        self.collection.update_one({'_id': ObjectId(employee_id)}, {'$set': updated_fields})

    def delete(self, employee_id):
        """
        주어진 Employee ID의 데이터를 삭제합니다.
        """
        self.collection.delete_one({'_id': ObjectId(employee_id)})