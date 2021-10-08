import pandas as pd
from src.data_preparation import create_training_data
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
import lemmy

# Create training data
#create_training_data()

# Load the newly created training data
df = pd.read_pickle("data/processed/processed.pkl")

# Load stopwords
stopwords = stopwords.words("swedish")
#snowball = SnowballStemmer(language="swedish")
lem = lemmy.load("sv")  # lem.lemmatize("NOUN", token)[0]

# Create a lemmatized column
df["word_tokens"] = df["tweets"].apply(lambda x: word_tokenize(x))
df["lemmatized"] = df["word_tokens"].apply(lambda y: [lem.lemmatize("NOUN", word)[0] for word in y])
df["tweets_lemma"] = df["lemmatized"].apply(lambda z: " ".join(z))

# Create the CountVectorizer
cv = CountVectorizer(ngram_range=(1, 2), 
analyzer="word",
stop_words=stopwords)

# Start the CV
X = cv.fit_transform(df["tweets_lemma"])
y = df["party"] 

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Initiate and train model
RFC = RandomForestClassifier()
RFC.fit(X_train, y_train)

# Test
y_prob_pred = RFC.predict_proba(X_test)
y_pred = RFC.predict(X_test)

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=RFC.classes_)
print(cm)
# Manually validate response
#classes = RFC.classes_

#for i, sample in enumerate(y_pred):
#    print("True label {0}".format(y_test.iloc[i]))
#    for j, oneclass in enumerate(classes):
#        print("Party: {0}, Probability: {1}".format(oneclass, sample[j]))
#    
#    time.sleep(5)
