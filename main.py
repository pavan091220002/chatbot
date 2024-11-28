import nltk
import re
import string
import pandas as pd
import nlp as np
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI()
templates = Jinja2Templates(directory="templates")

df=pd.read_csv('dialogs.txt',names=('Query','Response'),sep=('\t'))
Text=df['Query']
analyzer = SentimentIntensityAnalyzer()
df['rating'] = Text.apply(analyzer.polarity_scores)
df=pd.concat([df.drop(['rating'], axis=1), df['rating'].apply(pd.Series)], axis=1)

punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())
remove_n = lambda x: re.sub("\n", " ", x)
remove_non_ascii = lambda x: re.sub(r'[^\x00-\x7f]',r' ', x)
alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', x)

df['Query'] = df['Query'].map(alphanumeric).map(punc_lower).map(remove_n).map(remove_non_ascii)

df['Response'] = df['Response'].map(alphanumeric).map(punc_lower).map(remove_n).map(remove_non_ascii)

important_sentence=df.sort_values(by='compound', ascending=False)
postive_sentence=df.sort_values(by='pos', ascending=False)
negative_sentence = df.sort_values(by = 'neg', ascending=False)
neutral_sentence = df.sort_values(by = 'neu', ascending=False)

tfidf = TfidfVectorizer()
factors = tfidf.fit_transform(df['Query']).toarray()
tfidf.get_feature_names_out()

def chatbot(query):
    query = np.lemmatization_sentence(query)
    query_vector = tfidf.transform([query]).toarray()
    similar_score = 1 -cosine_distances(factors,query_vector)
    index = similar_score.argmax() 
    matching_question = df.loc[index]['Query']
    response = df.loc[index]['Response']
    pos_score = df.loc[index]['pos']
    neg_score = df.loc[index]['neg']
    neu_score = df.loc[index]['neu']
    confidence = similar_score[index][0]
    chat_dict = {'match':matching_question,
                'response':response,
                'score':confidence,
                'pos':pos_score,
                'neg':neg_score,
                'neu':neu_score}
    return chat_dict['response']

class Query(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat/")
async def chat(query: Query):
    response = chatbot(query.query)
    return {"response": response}