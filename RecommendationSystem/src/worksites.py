#worksite

from datetime import datetime
from bson import ObjectId
from config.db import get_collection

class Worksites:
    def __init__(self, user, name, address, local, salary, worktype, date, end, nopr, wanted, sent, sendmessage, recieved, denied, hired, worksitenote, createdAt, updateAt):
        self.user = ObjectId(user)
        self.name = name
        self.address = address
        self.local = local
        self.salary = salary
        self.worktype = worktype
        self.date = date
        self.end = end
        self.nopr = nopr
        self.wanted = wanted(employee)
        self.sent = ObjectId(employee)
        self.sendmessage = sendmessage
        self.recieved = recieved
        self.denied = ObjectId(employee)
        self.hired = ObjectId(employee)
        self.worksitenote = worksitenote
        self.createdAt = createdAt
        self.updateAt = updateAt

    def to_dict(self):
        return {
            'user': self.user,
            'name': self.name,
            'address': self.address,
            'local': self.local,
            'salary': self.salary,
            'worktype': self.worktype,
            'date': self.date,
            'end': self.end,
            'nopr': self.nopr,
            'wanted': self.wanted,
            'sent': self.sent,
            'sendmessage': self.sendmessage,
            'recieved': self.recieved,
            'denied': self.denied,
            'hired': self.hired,
            'worksitenote': self.worksitenote,
            'createdAt': self.createdAt,
            'updateAt': self.updateAt
        }

# local 나머지는 디비 쿼리수를 계산하여 연산