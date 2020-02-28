from src.Fake_News.Backend.database import connection
import time
import sys


def mongo_insert(collection, item):
    if isinstance(item, dict):
        try:
            collection.insert(item)
        except Exception as e:
            print(e)
    else:
        print("Error. Please provide a dictionary to insert")


def mongo_update(collection, item):
    if isinstance(item, dict):
        try:
            collection.update({'_id': item['_id']}, {'$set': item}, upsert=False)
        except Exception as e:
            print(e)
    else:
        print("Error. Please provide a dictionary.")


if __name__ == '__main__':
    conn = connection.openConnection(hostname='155.207.200.153',
                                     port=27027,
                                     username='ggravanis',
                                     password='mongo123gravanis',
                                     dbname='gravanis',
                                     authMechanism='SCRAM-SHA-1')

    # cursor_to = conn.create_collection(name="NLP")

    print(conn)
    db = conn['Copy_of_statements'].find({})

    for index, record in enumerate(db):
        sys.stdout.write('\r')
        sys.stdout.write("Statement %i out of %i" % (index + 1, db.count()))
        sys.stdout.flush()
        time.sleep(0.2)

        item = {"_id": record['_id']}

        if record['ruling']['ruling_slug'] == 'true':
            item['annotation'] = 0
        elif record['ruling']['ruling_slug'] == 'mostly-true':
            item['annotation'] = 1
        elif record['ruling']['ruling_slug'] == 'half-true':
            item['annotation'] = 2
        elif record['ruling']['ruling_slug'] == 'barely-true':
            item['annotation'] = 3
        elif record['ruling']['ruling_slug'] == 'false':
            item['annotation'] = 4
        elif record['ruling']['ruling_slug'] == 'pants-fire':
            item['annotation'] = 5
        else:
            item['annotation'] = 100

        mongo_update(conn['Copy_of_statements'], item=item)

    connection.closeConnection(db)
