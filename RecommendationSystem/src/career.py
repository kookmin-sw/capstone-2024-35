#career

from datetime import datetime
from bson import ObjectId
from config.db import get_collection

class Career:
    def __init__(self, employee, worksite, done, pay, review, score):
        self.employee = ObjectId(employee)
        self.worksite = ObjectId(worksite)
        self.done = done
        self.pay = pay
        self.review = review
        self.score = score

    def to_dict(self):
        return {
            'employee': self.employee,
            'worksite': self.worksite,
            'done': self.done,
            'pay': self.pay,
            'review': self.review,
            'score': self.score
        }

# career에서는 review 데이터 가져오고 score값 저장할 것임.

class Career_repository:
    """
    Career 데이터를 관리하는 저장소 클래스.
    """
    def __init__(self):
        self.collection = get_collection('career')  # 실제 컬렉션 이름에 맞게 수정해야 합니다.

    def insert(self, career: Career):
        """
        새로운 Career를 삽입합니다.
        """
        self.collection.insert_one(career.to_dict())

    def find_all(self):
        """
        모든 Career 데이터를 반환합니다.
        """
        return list(self.collection.find())

    def find_by_employee(self, employee_id):
        """
        직원 ID로 Career 데이터를 찾습니다.
        """
        return list(self.collection.find({'employee': ObjectId(employee_id)}))

    def update(self, career_id, updated_fields):
        """
        주어진 Career ID의 데이터를 업데이트합니다.
        """
        self.collection.update_one({'_id': ObjectId(career_id)}, {'$set': updated_fields})

    def delete(self, career_id):
        """
        주어진 Career ID의 데이터를 삭제합니다.
        """
        self.collection.delete_one({'_id': ObjectId(career_id)})





