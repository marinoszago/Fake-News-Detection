import requests
from src.Fake_News.Backend.database.connection import openConnection
from src.Fake_News.Backend.config.config import *


# Initializing the GET request

def getStatementURL():
    url = "https://www.politifact.com/api/subjects/all/json/"
    return url


# Get the json request

def getSubjectsFromApi():
        try:
                url_to_call = getStatementURL()

                querystring = {"format": "json"}

                headers = {
                    'Content-Type': "application/json",
                    'Cache-Control': "no-cache",
                    'Postman-Token': "6d36965a-88c2-4081-bdc4-b3c007a816ae"
                }

                response = requests.get(url_to_call, headers=headers, params=querystring)
                json_data = response.json()

                print(json_data)
                try:
                    db = openConnection(db_hostname, db_name, db_port, db_username, db_password, db_authMechanism)
                    collection_statements = db[subjects]
                    collection_statements.insert(json_data)
                    print("Connected successfully!!!")

                finally:
                    print("Insert was successful:")
        except ValueError:
            print("Some error with the values")

        except BlockingIOError:
            print("Some error with the IO")

        except ConnectionError:
            print("Some error with the connection")


def main():
    getSubjectsFromApi()


if __name__ == '__main__':
    print("You execute get_subjects.py as single file")