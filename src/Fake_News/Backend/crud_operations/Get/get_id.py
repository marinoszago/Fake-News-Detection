from src.Fake_News.Backend.database.connection import openConnection
from src.Fake_News.Backend.config.config import *


class StatementsIDs():
    def getIDS(collection):
        statementsIDS = []
        for statement in collection.find():
            statementsIDS.append({"id": statement['_id']})
        return statementsIDS


