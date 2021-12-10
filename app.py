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
import seaborn as sns
import numpy as np
import re

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

#@st.cache
data = pd.read_csv('resources/train.csv')

def main():
    """Tweet Sentiment Classifier with Streamlit """

    st.title("Tweet Sentiment Classifier ML App")

    image = Image.open('resources/imgs/tweeter.jpg')

    st.image(image, caption='Which Tweet are you?', use_column_width=True)

    st.subheader("ML App for Tweet Sentiment Classification")

    activities = ["Information","Visuals","Sentiment Classifier", "NLP","Contact App Developers"]

    choice = st.sidebar.selectbox("Choose Activity", activities)

    if choice == 'Information':
        st.info('General Information')
        st.write('Explorers Explore and Great Things Happen!')
        st.markdown(""" We have deployed Machine Learning models that are able to classify 
        whether or not a person believes in climate change, based on their novel tweet data. 
        Like any data lovers, these are robust solutions to that can provide access to a 
        broad base of consumer sentiment, spanning multiple demographic and geographic categories. 
        So, do you have a Twitter API and ready to scrap? or just have some tweets off the top of your head? 
        Do explore the rest of this app's buttons.
        """)


        raw = st.checkbox('See raw data')
        if raw:
            st.dataframe(data.head(25))

    if choice == 'Visuals':

        st.set_option('deprecation.showPyplotGlobalUse', False)

        st.info('The following are some of the charts that we have created from the raw data. Some of the text is too long and may cut off, feel free to right click on the chart and either save it or open it in a new window to see it properly.')


       # Number of Messages Per Sentiment
        st.write('Distribution of the sentiments')
        # Labeling the target
        data['sentiment'] = [['Negative', 'Neutral', 'Positive', 'News'][x+1] for x in data['sentiment']]
        
        # checking the distribution
        st.write('The numerical proportion of the sentiments')
        values = data['sentiment'].value_counts()/data.shape[0]
        labels = (data['sentiment'].value_counts()/data.shape[0]).index
        colors = ['lightgreen', 'blue', 'purple', 'lightsteelblue']
        plt.pie(x=values, labels=labels, autopct='%1.1f%%', startangle=90, explode= (0.04, 0, 0, 0), colors=colors)
        st.pyplot()
        
        # checking the distribution
        sns.countplot(x='sentiment' ,data = data, palette='PRGn')
        plt.ylabel('Count')
        plt.xlabel('Sentiment')
        plt.title('Number of Messages Per Sentiment')
        st.pyplot()

        # Popular Tags
        st.write('Popular tags found in the tweets')
        data['users'] = [''.join(re.findall(r'@\w{,}', line)) if '@' in line else np.nan for line in data.message]
        sns.countplot(y="users", hue="sentiment", data=data,
                    order=data.users.value_counts().iloc[:20].index, palette='PRGn') 
        plt.ylabel('User')
        plt.xlabel('Number of Tags')
        plt.title('Top 20 Most Popular Tags')
        st.pyplot()

        # Tweet lengths
        st.write('The length of the sentiments')
        st.write('The average Length of Messages in all Sentiments is 100 which is of no surprise as tweets have a limit of 140 characters.')

        # Repeated tags
        
        # Generating Counts of users
        st.write("Analysis of hashtags in the messages")
        counts = data[['message', 'users']].groupby('users', as_index=False).count().sort_values(by='message', ascending=False)
        values = [sum(np.array(counts['message']) == 1)/len(counts['message']), sum(np.array(counts['message']) != 1)/len(counts['message'])]
        labels = ['First Time Tags', 'Repeated Tags']
        colors = ['lightsteelblue', "purple"]
        plt.pie(x=values, labels=labels, autopct='%1.1f%%', startangle=90, explode= (0.04, 0), colors=colors)
        st.pyplot()

        # Popular hashtags
        st.write("The Amount of popular hashtags")
        repeated_tags_rate = round(sum(np.array(counts['message']) > 1)*100/len(counts['message']), 1)
        print(f"{repeated_tags_rate} percent of the data are from repeated tags")
        sns.countplot(y="users", hue="sentiment", data=data, palette='PRGn',
              order=data.users.value_counts().iloc[:20].index) 
        plt.ylabel('User')
        plt.xlabel('Number of Tags')
        plt.title('Top 20 Most Popular Tags')
        st.pyplot()

        st.markdown("Now that we've had a look at the tweets themselves as well as the users, we now analyse the hastags:")

        # Generating graphs for the tags
        st.write('Analysis of most popular tags, sorted by populariy')
        # Analysis of most popular tags, sorted by populariy
        sns.countplot(x="users", data=data[data['sentiment'] == 'Positive'],
                    order=data[data['sentiment'] == 'Positive'].users.value_counts().iloc[:20].index) 

        plt.xlabel('User')
        plt.ylabel('Number of Tags')
        plt.title('Top 20 Positive Tags')
        plt.xticks(rotation=85)
        st.pyplot()

        # Analysis of most popular tags, sorted by populariy
        st.write("Analysis of most popular tags, sorted by populariy")
        sns.countplot(x="users", data=data[data['sentiment'] == 'Negative'],
                    order=data[data['sentiment'] == 'Negative'].users.value_counts().iloc[:20].index) 

        plt.xlabel('User')
        plt.ylabel('Number of Tags')
        plt.title('Top 20 Negative Tags')
        plt.xticks(rotation=85)
        st.pyplot()


        st.write("Analysis of most popular tags, sorted by populariy")
        # Analysis of most popular tags, sorted by populariy
        sns.countplot(x="users", data=data[data['sentiment'] == 'News'],
                    order=data[data['sentiment'] == 'News'].users.value_counts().iloc[:20].index) 

        plt.xlabel('User')
        plt.ylabel('Number of Tags')
        plt.title('Top 20 News Tags')
        plt.xticks(rotation=85)
        st.pyplot()

    if choice == 'Sentiment Classifier':
        st.info("Classification with ML")

        data_source = ['Select option', 'Single text', 'Dataset'] ## differentiating between a single text and a dataset input

        source_selection = st.selectbox('What to classify?', data_source)

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
            #text_input['sentiment'] = prediction
            final_result = get_keys(prediction, prediction_labels)
            st.success("Tweet classified as: \n{}".format(final_result))

    ##contact page
    if choice == 'Contact App Developers':

        st.info('Contact details in case you any query or would like to know more of our designs:')
        st.write('Mulalo: mulalo.malange@yahoo.com')
        st.write('Vuyisile: vuyiedannie@gmail.com')
        st.write('Muhammed: muhammedirfaan1@gmail.com')
        st.write('Jesica: jesicateffo@gmail.com')
        st.write('Floyd: floyd.skakane@gmail.com')

        # Footer 
        image = Image.open('resources/imgs/EDSA_logo.png')

        st.image(image, caption='Team-6-Johannesbrug', use_column_width=True)
            
            

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