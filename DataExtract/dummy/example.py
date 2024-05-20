from config.db import connect_db
import pprint

db = connect_db()
collection = db['ExtractedEntities']
db.list_collection_names()

for document in collection.find():
    print(document)