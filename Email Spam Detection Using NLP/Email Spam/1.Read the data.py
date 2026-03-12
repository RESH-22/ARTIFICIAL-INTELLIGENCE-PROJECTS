import pandas as pd
import nltk

##rawdata=open("SMSSpamCollection.tsv").read()
##rawdata[0:250]

data=pd.read_csv('SMSSpamCollection.tsv', sep='\t', names=['label','body_text'], header=None)
print(data.head())
