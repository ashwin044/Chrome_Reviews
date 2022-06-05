from asyncio import exceptions
import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from nltk.stem import WordNetLemmatizer 
# from textblob import TextBlob
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words.remove('not')

st.set_page_config(layout="wide")
st.title("Positive Reviews with Single Star")

input_file = st.file_uploader(
    label="please upload a csv or excel file.(200MB Max)",
    type=['CSV', 'xlsx'])

global data

if input_file is not None:
    try:
        data = pd.read_csv(input_file)
    except Exception as e:
        print(e)
        data = pd.read_excel(input_file)

try:
    clean_text =[]
    for review in data['Text']:
        review= re.sub(r'[^\w\s]', '', str(review))
        review = re.sub(r'\d','',review)
        review_token = word_tokenize(review.lower().strip()) #convert reviews into lower case and strip leading and tailing spaces followed by spliting sentnece into words
        review_without_stopwords=[]
        for token in review_token:
            if token not in stop_words:
                token= lemmatizer.lemmatize(token)
                review_without_stopwords.append(token)
        cleaned_review = " ".join(review_without_stopwords)
        clean_text.append(cleaned_review)

    data["cleaned_review"] = clean_text
    Single_star_reviews = data[data.Star ==1]


    sia = SentimentIntensityAnalyzer()
    senti_list = []

    for i in Single_star_reviews["cleaned_review"]:
        score = sia.polarity_scores(i)
        # blob_score = TextBlob(i).sentiment.polarity
        if (score['pos'] >= 0.5):
            senti_list.append('Positive')
        else:
            senti_list.append('Negative/Neutral') 

    Single_star_reviews["sentiment"]= senti_list
    # Single_star_reviews.head(5)
    positive_review_with_1_star = Single_star_reviews[Single_star_reviews.sentiment == 'Positive']
    st.write(positive_review_with_1_star)

except Exception as e:
    print(e)
    st.write("Please upload a csv or excel file")
