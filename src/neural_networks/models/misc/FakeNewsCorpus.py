"""


https://github.com/several27/FakeNewsCorpus
"""

import pandas as pd

dataset = pd.read_csv('../../../../data/news_sample.csv')

df = pd.DataFrame(dataset)

print(df.count())

# df = df.groupby('type')['type'].nunique()
# print(df)

# df = df['type'].value_counts()
#
# print (df)

for i, v in df.iterrows():
    print(i, v)
    break
