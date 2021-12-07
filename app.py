import streamlit as st
import joblib, os

#NLP Packages
import spacy
nlp = spacy.load('en_core_web_sm')

#EDA Packages
import pandas as pd

#Wordcloud
from wordcloud import WordCloud
from PIL import Image

#Visualization
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('Agg')

#Vectorizer
tweet_vectorizer = open("models/vectorizer_1.pkl", "rb")
tweet_cv = joblib.load(tweet_vectorizer)

#Load Models
def load_models(model_file):
    loaded_models = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_models

def get_keys(val,my_dict):
    for key,value in my_dict.items():
        if val == value:
            return key

def main():
    """Tweet Sentiment Classifier with Streamlit """
    st.title("Tweet Sentiment Classifier ML App")
    st.subheader("ML App for Tweet Sentiment Classification")

    activities = ["Sentiment Classifier", "NLP"]

    choice = st.sidebar.selectbox("Choose Activity", activities)

    if choice == 'Sentiment Classifier':
        st.info("Classification with ML")

        tweet = st.text_area("Enter Tweet", "Type/Paste Here")
        ml_models = ["Logistic Regression I","Logistic Regression II","Nearest Neighbors","Multinomial Naive Bayes","Linear SVC","RBF SVC","Linear SVM","Decision Tree","Random Forest","AdaBoost"]
        model_choice = st.selectbox("Choose ML Model", ml_models)
        prediction_labels = {'News':2,'Pro':1,'Neutral':0,'Anti':-1}
        if st.button("Classify"):
            st.text("Original tweet:\n{}".format(tweet))
            vect_tweet = tweet_cv.transform([tweet]).toarray()
            if model_choice == 'Logistic Regression I':
                predictor = load_models("models/logisticreg_model.pkl")
                prediction = predictor.predict(vect_tweet)
                #st.write(prediction)
            elif model_choice == 'Logistic Regression II':
                predictor = load_models("models/logreg_model_1.pkl")
                prediction = predictor.predict(vect_tweet)
                #st.write(prediction)
            elif model_choice == 'Nearest Neighbors':
                predictor = load_models("models/knn_model_1.pkl")
                prediction = predictor.predict(vect_tweet)
                #st.write(prediction)
            elif model_choice == 'Multinomial Naive Bayes':
                predictor = load_models("models/mnb_model_1.pkl")
                prediction = predictor.predict(vect_tweet)
                #st.write(prediction)
            elif model_choice == 'Linear SVC':
                predictor = load_models("models/linear_svc_model_1.pkl")
                prediction = predictor.predict(vect_tweet)
                #st.write(prediction)
            elif model_choice == 'RBF SVC':
                predictor = load_models("models/svc_1_model_1.pkl")
                prediction = predictor.predict(vect_tweet)
                #st.write(prediction)
            elif model_choice == 'Linear SVM':
                predictor = load_models("models/svc_2_model_1.pkl")
                prediction = predictor.predict(vect_tweet)
                #st.write(prediction)
            elif model_choice == 'Decision Tree':
                predictor = load_models("models/dec_tree_model_1.pkl")
                prediction = predictor.predict(vect_tweet)
                #st.write(prediction)
            elif model_choice == 'Random Forest':
                predictor = load_models("models/randfor_model_1.pkl")
                prediction = predictor.predict(vect_tweet)
                #st.write(prediction)
            elif model_choice == 'AdaBoost':
                predictor = load_models("models/adaboost_model_1.pkl")
                prediction = predictor.predict(vect_tweet)
                #st.write(prediction)
            final_result = get_keys(prediction, prediction_labels)
            st.success("Tweet classified as: \n{}".format(final_result))
            
            

    if choice == 'NLP':
        st.info("Natural Language Processing")
        tweet = st.text_area("Enter Tweet", "Type/Paste Here")
        nlp_task = ["Tokenization","NER","Lemmatization","POS Tags"]
        task_choice = st.selectbox("Choose NLP Task", nlp_task)
        if st.button("Analyze"):
            st.info("Origina Tweet {}".format(tweet))

            docx = nlp(tweet)
            if task_choice == 'Tokenization':
                result = [token.text for token in docx]
            elif task_choice == 'NER':
                result = [(entity.text,entity.label_) for entity in docx.ents]
            elif task_choice == 'Lemmatization':
                result = ["'Token':{},'Lemma':{}".format(token.text,token.lemma_) for token in docx]
            elif task_choice == 'POS Tags':
                result = ["'Token':{},'POS':{},'Dependency:{}".format(word.text,word.tag_,word.dep_) for word in docx]
            
            st.json(result)
        
        if st.button("Tabulate"):
            docx = nlp(tweet)
            c_tokens = [token.text for token in docx]
            c_ner = [(entity.text,entity.label_) for entity in docx.ents]
            c_lemmas = [token.lemma_ for token in docx]
            c_pos_tags = [(word.tag_) for word in docx]

            df_initial = pd.DataFrame(zip(c_tokens,c_ner,c_lemmas,c_pos_tags),columns=['Tokens','Entities','Lemmas','POS Tags'])
            st.dataframe(df_initial)

        if st.checkbox("WordCloud"):
            wordcloud = WordCloud().generate(tweet)
            plt.imshow(wordcloud,interpolation='bilinear')
            plt.axis("off")
            st.pyplot()


if __name__ == '__main__':
    main()