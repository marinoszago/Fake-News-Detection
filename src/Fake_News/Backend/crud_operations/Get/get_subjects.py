from bson.int64 import Int64


class Subjects():
    # Define the count field so that it returns those specific subjects
    def getSubjectsByCount(self, collection, count):
        subjects_array = []
        query = {}
        query["count"] = {
            u"$gte": Int64(count)
        }

        sort = [(u"count", -1)]

        cursor = collection.find(query, sort=sort)
        try:
            for doc in cursor:
                subjects_array.append(doc)
        finally:
            print("Subjects retrieved")

        return subjects_array

    def getStatementSubjectCount(self, collection, subject_name):
        counter = 0
        pipeline = [
            {
                u"$match": {
                    u"subject.subject": subject_name
                }
            },
            {
                u"$group": {
                    u"_id": {},
                    u"COUNT(_id)": {
                        u"$sum": 1
                    }
                }
            },
            {
                u"$project": {
                    u"_id": 0,
                    u"COUNT(_id)": u"$COUNT(_id)"
                }
            }
        ]

        cursor = collection.aggregate(
            pipeline,
            allowDiskUse=True
        )
        try:
            for doc in cursor:
                counter = doc['COUNT(_id)']
        finally:
            print('Count retrieved!!!')
        return counter