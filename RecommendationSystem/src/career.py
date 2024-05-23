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




