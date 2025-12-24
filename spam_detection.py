from spam_train import process_text
import joblib

# model=joblib.load("spam_detector/spam_model_NB.pkl")
model=joblib.load("spam_detector/_spam_model_LG.pkl")
vectorizer=joblib.load("spam_detector/_spam_vectorizer.pkl")

def spam_check(message):
    message=process_text(message)
    message_trm=vectorizer.transform([message])
    prediction=model.predict(message_trm)[0]
    prediction_proba=model.predict_proba(message_trm)[0]
    # print(prediction)
    print("Detected probability:[Spam/not_spam] ",int(prediction_proba[0]*100),"%",int(prediction_proba[1]*100),"%")
    if prediction==0:
        return "Spam"
    elif prediction==1 and prediction_proba[0]>0.40:
        return "Potential spam"
    else :
        return "Not_spam"
    

while True:
    message_user=input("Enter your message: ")
    if message_user=="":
        break
    print(spam_check(message_user))