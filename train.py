import joblib
import pandas as pd
from src.data_preparation import create_training_data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from nltk.stem import SnowballStemmer
from nltk import TweetTokenizer
import lemmy

# Create training data
#create_training_data()

# Load the newly created training data
df = pd.read_pickle("data/processed/processed.pkl")

# Load stopwords
stopwords = stopwords.words("swedish")
#snowball = SnowballStemmer(language="swedish")
lem = lemmy.load("sv")  # lem.lemmatize("NOUN", token)[0]
twtknizer = TweetTokenizer()

# Create a lemmatized column
import re
def remove_punctuations(input_text):
    chars_to_remove = [".", "!", "?", ",", ":", ";"]
    for special_char in chars_to_remove:
        input_text = input_text.replace(special_char, "")
    return input_text

df["tweets"] = df["tweets"].apply(lambda x: remove_punctuations(x))
df["word_tokens"] = df["tweets"].apply(lambda x: twtknizer.tokenize(x))
df["lemmatized"] = df["word_tokens"].apply(lambda y: [lem.lemmatize("NOUN", word)[0] for word in y])
df["tweets_lemma"] = df["lemmatized"].apply(lambda z: " ".join(z))

# Create the CountVectorizer and TfIdf Transformer
pipe = Pipeline([
    ("count", CountVectorizer(ngram_range=(1, 3), analyzer="word", stop_words=stopwords)), 
    ("tfidf", TfidfTransformer())]).fit(df["tweets_lemma"])

X = pipe.transform(df["tweets_lemma"])
y = df["party"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Initiate and train model
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier # Let's try this! 
model = XGBClassifier()
model.fit(X_train, y_train)

# Test
y_prob_pred = model.predict_proba(X_test)
y_pred = model.predict(X_test)

# Quick eval
print(y_test)
print(y_pred)

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=model.classes_)
print(cm)
print(y_pred)
print(y_test)
# Save everything we have done
joblib.dump(model, "data/output/model.pkl")
joblib.dump(pipe, "data/output/pipe.pkl")
