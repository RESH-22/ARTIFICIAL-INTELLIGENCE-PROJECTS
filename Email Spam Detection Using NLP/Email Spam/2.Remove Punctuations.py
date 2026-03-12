import pandas as pd
import nltk

##rawdata=open("SMSSpamCollection.tsv").read()
##rawdata[0:250]
data=pd.read_csv('SMSSpamCollection.tsv', sep='\t', names=['label','body_text'], header=None)
print("Read data",data.head)


import string
string.punctuation
def remove_punct(text):
    text_nopunct="".join([char for char in text if char not in string.punctuation])
    return text_nopunct
data['body_text_clean']=data['body_text'].apply(lambda x:remove_punct(x))
print("Remove_punctuation",data.head)
