from src.neural_networks.preprocess.embeddings import preprocess_articles as pp
import pandas as pd
import numpy as np
from src.Fake_News.Backend.database import connection
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sys

from src.Fake_News.Backend.database import connection
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from time import time, sleep

path = '../../data/raw_text/'
save_path = '../../data/word_embeddings/'
model_path = 'E:\GoogleNews-vectors-negative300.bin'
glove_file = '../../../../data/embeddings/glove.6B.300d.txt'
tmp_file = '../../../../data/embeddings/temp_w2vec.txt'

glove2word2vec(glove_file,tmp_file)
chunksize = 10000

conn = connection.openConnection(hostname='155.207.200.153',
                                 port=27027,
                                 username='ggravanis',
                                 password='mongo123gravanis',
                                 dbname='gravanis',
                                 authMechanism='SCRAM-SHA-1')
# conn.create_collection("FakeNews_w2v")

counter = 0
# load model
model = KeyedVectors.load_word2vec_format(tmp_file)

stop_words = set(stopwords.words('english'))

for chunk in pd.read_csv('../../../../data/embeddings/news_cleaned_2018_02_13.csv', chunksize=chunksize,lineterminator='\n'):
    print("I am in chunk #: {}".format(counter + 1))
    if(counter <= 15):
        counter += 1
        continue
    else:

        df = pd.DataFrame(chunk)
        temp_data = pp.preprocess_articles_fake_news(dataset=df, model=model, stop_words=stop_words, bin=True)

        collection = conn["fakeNewsGlove"]

        for key, value in temp_data.items():
            temp_dict = {}
            for i, v in enumerate(value):
                temp_dict[str(i)] = v

            # print(temp_dict)
            try:
                collection.insert(temp_dict)
                print("Connected successfully!!!")
                print("Insert was Successfull! -> Data:")

            except Exception:
                print("Connection Failed!!")
                exit()
        counter += 1

