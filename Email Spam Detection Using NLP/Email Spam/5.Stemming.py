import pandas as pd
import nltk

rawdata=open("SMSSpamCollection.tsv").read()
rawdata[0:250]
data=pd.read_csv('SMSSpamCollection.tsv', sep='\t', names=['label','body_text'], header=None)
print(data.head)


##Remove Punctuations
import string
string.punctuation        
def remove_punct(text):
    text_nopunct="".join([char for char in text if char not in string.punctuation])
    return text_nopunct
data['body_text_clean']=data['body_text'].apply(lambda x:remove_punct(x))
print(data.head)

##Tokenization
import re
def tokenize(text):
    tokens=re.split('\W+',text)
    return tokens
data['body_text_tokenized']=data['body_text_clean'].apply(lambda x: tokenize(x.lower()))
print(data.head)

##Remove Stopwords
stopword=nltk.corpus.stopwords.words('english')
def remove_stopwords(tokenized_list):
    text=[word for word in tokenized_list if word not in stopword]
    return text
data['body_text_nostop']=data['body_text_tokenized'].apply(lambda x:remove_stopwords(x))
print(data.head)

##Stemming

ps=nltk.PorterStemmer()
def stemming(tokenized_text):
    text=[ps.stem(word) for word in tokenized_text]
    return text
data['body_text_stemmed']=data['body_text_nostop'].apply(lambda x:stemming(x))
print(data.head)
