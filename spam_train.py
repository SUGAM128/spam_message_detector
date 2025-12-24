import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import accuracy_score 


def process_text(text):
    text=text.lower()
    text=re.sub(r'[^a-z\s]','',text)
    return text



if __name__== "__main__":
    df=pd.read_csv("spam_data.csv",usecols=[0,1],encoding="latin-1")  

    # df=df[['v1','v2']]
    df.columns=['Level','Message']

    df=df[df["Level"].isin(["ham","spam"])]

    df["Level"]=df["Level"].astype(str).str.lower()
    df["Level_num"]=df["Level"].map({"ham":1,"spam":0})

    df['Message']=df['Message'].apply(process_text)


    x=df["Message"]
    y=df["Level_num"]

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


    vectorizer=TfidfVectorizer(stop_words='english',ngram_range=(1,2))
    x_train_tfdif=vectorizer.fit_transform(x_train)
    x_test=vectorizer.transform(x_test)

    model=MultinomialNB()
    model.fit(x_train_tfdif,y_train)
    
    y_pred=model.predict(x_test)

    accuracy_NB=accuracy_score(y_test,y_pred)
   

    joblib.dump(model,"spam_model_NB.pkl")
    joblib.dump(vectorizer,"_spam_vectorizer.pkl")

    #logistic_regression model

    model=LogisticRegression()
    model.fit(x_train_tfdif,y_train)

    y_pred_LG=model.predict(x_test)

    accuracy_LG=accuracy_score(y_test,y_pred_LG)
    joblib.dump(model,"_spam_model_LG.pkl")

    print("Accuracy_multinominal: ",int(accuracy_NB*100),"%","\nAccuracy_logistic: ",int(accuracy_LG*100),"%")












