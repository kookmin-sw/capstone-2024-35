#employee

from datetime import datetime
from bson import ObjectId
from config.db import get_collection

class Employee:
    def __init__(self, _id,user, name, sex, local, rrn,career,age, created_at=None, updated_at=None):
        self._id=ObjectId(_id)
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

# sex, employee id, local