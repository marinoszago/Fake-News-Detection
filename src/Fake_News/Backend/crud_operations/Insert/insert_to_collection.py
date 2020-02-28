from src.Fake_News.Backend.database.connection import openConnection, closeConnection
from src.Fake_News.Backend.config.config import *

class Insert():
    def insert_to_collection(db, collection_name, data):
        try:

            collection = db[collection_name]
            collection.insert(data)
            print("Connected successfully!!!")
            print("Insert was Successfull! -> Data: "+data['Annotation'])



        except Exception:
            print("Connection Failed!!")
            exit()