from src.neural_networks.preprocess.embeddings import preprocess_articles as pp
import pandas as pd
import numpy as np
from src.Fake_News.Backend.database import connection
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sys
import random
import  numpy
from random import randint

conn = connection.openConnection(hostname='155.207.200.153',
                                 port=27027,
                                 username='ggravanis',
                                 password='mongo123gravanis',
                                 dbname='gravanis',
                                 authMechanism='SCRAM-SHA-1')

collection = conn["FakeNews_w2v_test"]


fake = []
counter = 0
for data in collection.find({'301': 'fake'}).limit(150000):
    counter += 1
    print(counter)
    fake.append(data)

df_fake = pd.DataFrame(fake)
df_fake.to_csv('../../../../data/embeddings/fake_data.csv',sep=',')

satire = []
counter = 0
for data in collection.find({'301': 'satire'}).limit(140000):
    counter += 1
    print(counter)
    satire.append(data)

df_satire = pd.DataFrame(satire)
df_satire.to_csv('../../../../data/embeddings/satire_data.csv',sep=',')

bias = []
counter = 0
for data in collection.find({'301': 'bias'}).limit(100000):
    counter += 1
    print(counter)
    bias.append(data)

df_bias = pd.DataFrame(bias)
df_bias.to_csv('bias_data.csv',sep=',')

conspiracy = []
counter = 0
for data in collection.find({'301': 'conspiracy'}).limit(150000):
    counter += 1
    print(counter)
    conspiracy.append(data)

df_conspiracy = pd.DataFrame(conspiracy)
df_conspiracy.to_csv('conspiracy_data.csv',sep=',')

junksci = []
counter = 0
for data in collection.find({'301': 'junksci'}).limit(1400000):
    counter += 1
    print(counter)
    junksci.append(data)

df_junksci = pd.DataFrame(junksci)
df_junksci.to_csv('junksci_data.csv',sep=',')

numpy.set_printoptions(threshold=sys.maxsize)
hate = []
counter = 0
for data in collection.find({'301': 'hate'}).skip(5).limit(100000):

    counter += 1

    try:
        print(counter)
        hate.append(data)
    except:
        continue

df_hate = pd.DataFrame(hate)
df_hate.to_csv('hate_data.csv',sep=',')

clickbait = []
counter = 0
for data in collection.find({'301': 'clickbait'}).limit(150000):
    counter += 1
    print(counter)
    clickbait.append(data)

df_clickbait = pd.DataFrame(clickbait)
df_clickbait.to_csv('clickbait_data.csv',sep=',')


unreliable = []
counter = 0
for data in collection.find({'301': 'unreliable'}).limit(150000):
    counter += 1
    print(counter)
    unreliable.append(data)

df_unreliable = pd.DataFrame(unreliable)
df_unreliable.to_csv('unreliable_data.csv',sep=',')

#GET ALL UNRELIABLE
unreliable = []
counter = 0
for data in collection.find({'301': 'unreliable'}):
    counter += 1
    print(counter)
    unreliable.append(data)

df_unreliable = pd.DataFrame(unreliable)
df_unreliable.to_csv('unreliable_all_data.csv',sep=',')

political = []
counter = 0
for data in collection.find({'301': 'political'}).limit(150000):
    counter += 1
    print(counter)
    political.append(data)

df_political = pd.DataFrame(political)
df_political.to_csv('political_data.csv',sep=',')

reliable = []
counter = 0
for data in collection.find({'301': 'reliable'}).limit(150000):
    counter += 1
    print(counter)
    reliable.append(data)

df_reliable = pd.DataFrame(reliable)
df_reliable.to_csv('reliable_data.csv',sep=',')