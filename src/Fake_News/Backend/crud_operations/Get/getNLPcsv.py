from src.Fake_News.Backend.database.connection import openConnection,closeConnection
from src.Fake_News.Backend.config.config import *
import pandas as pd
import csv
import os
from collections import namedtuple
import seaborn as sns
import numpy as np
from bson import CodecOptions
from bson.raw_bson import RawBSONDocument



db = openConnection(db_hostname, db_name, db_port, db_username, db_password, db_authMechanism)
collection_copyNLP = db['Copy_of_NLP']
print("OK...Connection Granted")

query = {}


NLP_data = collection_copyNLP.find(query)
NLP_data_list = []
for data in NLP_data:
    NLP_data_list.append(data)
NLP_dataframe = pd.DataFrame(NLP_data_list)

NLP_dataframe.to_csv('nlp_data.csv', sep=',')