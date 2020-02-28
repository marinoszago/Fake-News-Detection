from src.Fake_News.Backend.database.connection import openConnection
from src.Fake_News.Backend.config.config import *


class AllStatements():
    def getAllStatements(self, collection):
        print(self)
        counter = 0
        statements = []
        print("Statements Deployed... ETA: unknown")
        for statement in collection.find():
            counter += 1
            statements.append(statement)
            print("Statement was successfully appended with id#: "+str(counter))
        return statements


