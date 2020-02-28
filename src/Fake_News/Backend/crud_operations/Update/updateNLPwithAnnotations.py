from src.Fake_News.Backend.database.connection import openConnection,closeConnection
from src.Fake_News.Backend.config.config import *

class UpdateNLPWithAnnotations:
    def UpdateNLPWithAnnotation(self, collection1, collection2):
        query = {}
        projection = {}
        projection["annotation"] = u"$annotation"

        annotations = collection1.find(query, projection=projection)

        for annotation in annotations:
            collection2.update({"_id": annotation['_id']}, {"$set": {"annotation": annotation['annotation']}}, False, True)


if __name__ == "__main__":

    db = openConnection(db_hostname, db_name, db_port, db_username, db_password, db_authMechanism)
    collection_copyStatements = db['Copy_of_statements']
    collection_copyNLP = db['Copy_of_NLP']
    print("OK...Connection Granted")

    update = UpdateNLPWithAnnotations()
    update.UpdateNLPWithAnnotation(collection_copyStatements, collection_copyNLP)

