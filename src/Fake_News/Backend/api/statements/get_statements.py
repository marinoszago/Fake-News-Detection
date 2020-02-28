import requests
import time
from src.Fake_News.Backend.database.connection import openConnection, closeConnection
from src.Fake_News.Backend.config.config import *
import ast

#Initializing the GET request

def getStatementURL():
    return "https://www.politifact.com/api/v/2/statement/"


#Get the json request

def getStatements():
    iter_counter = 1
    response_counter = 1

    while True:
        try:
            print("--------", "start=>", response_counter, "-------")
            if iter_counter <= 5:
                url_to_call = getStatementURL()+str(response_counter)+"/"

                querystring = {"format": "json"}

                headers = {
                    'Content-Type': "application/json",
                    'Cache-Control': "no-cache",
                    'Postman-Token': "6d36965a-88c2-4081-bdc4-b3c007a816ae"
                }

                response = requests.get(url_to_call, headers=headers, params=querystring)
                #print(response.text)
                json_data = response.json()

                print(json_data)
                try:
                    db = openConnection(db_hostname, db_name, db_port, db_username, db_password, db_authMechanism)
                    collection_statements = db["statements"]
                    collection_statements.insert(json_data)
                    print("Connected successfully!!!")
                    closeConnection(db)


                except BaseException:
                    pass

                print("Insert was successful:")
                # print(response.text)
                print("--------/", "end=>", response_counter, "-------")
                response_counter += 1
                iter_counter += 1

            else:
                print("Sleeping for 10 seconds to not overflow the requests")
                time.sleep(10)
                iter_counter = 1
                continue
        except ValueError:
            print("Some error with the values")
            f = open("logfile.txt", "a")
            f.write("Could not write statement with id: "+str(response_counter)+"\n")
            response_counter += 1
            iter_counter += 1
            f.close()

            continue
        except BlockingIOError:
            print("Some error with the IO")
            continue
        except ConnectionError:
            print("Some error with the connection")
            continue



def main():
    getStatements()


if __name__ == '__main__':
    print("You execute get_statements.py as single file")