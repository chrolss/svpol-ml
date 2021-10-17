# #svpol Machine Learning

## Purpose
The machine learning part of the svpol project is to categorize tweets and political text from political figures and organizations, and then use the knowledge to predict party affiliation in new text.

## Todo:
- [x] Implement CountVectorizer & Tfidf transformer 
- [x] Evaluate basic classifiers (Naive Bayes, Random Forest Regressor)
- [x] Implement XGBoostClassifier
- [ ] Tune XGBoostClassifier with Optuna
- [ ] Implement Catboost
- [ ] Force website data to be included in training data

## Data
Raw data of tweets for training data is located in `~/PycharmProjects/svpol/data/tweets/politicans` with the look-up file for users are in `~/PycharmProjects/svpol/data/politicians.csv`. Raw data from party websites are located in `data/raw/websites`.

## Project structure
```
root
| - data
  | - output # Stores models and processing pipelines
  | - processed # Processed training data
  | - raw
    | - websites # Folder for website texts for training
| - src
  | - data_preparation.py
  | - feature_engineering.py
| - train.py
| - predict.py
```

## Current status
The new `XGBoostClassifier` works better than the simple `MultiNominalNB` or `RandomForestClassifier`, but it still needs tuning. For the training data I will also force the website data to be always included in the training set and use tweet history for evaluation to ensure that all classes have a solid baseline in the training data.