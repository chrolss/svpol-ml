import pandas as pd
import os
import glob

# CONSTANTS
raw_folder = "/home/chrolss/PycharmProjects/svpol/data/tweets/politicians"
lookup_path = "/home/chrolss/PycharmProjects/svpol/data/politicians.csv"
website_folder = "/home/chrolss/PycharmProjects/svpol-ml/data/raw/websites/"
TWEETS_TO_COMBINE = 200

party_dict = {
    'vansterpartiet.txt': "v",
    'sverigedemokraterna.txt': "sd",
    'miljopartiet.txt': "mp",
    'centerpartiet.txt': "c",
    'moderaterna.txt': "m",
    'liberalerna.txt': "l",
    'kristdemokraterna.txt': "kd",
    'socialdemokraterna.txt': "s"
}

def parse_website_data():
    websites = os.listdir(website_folder)
    username = ["website" for i in range(len(websites))]
    tweets = []
    party = []

    for website in websites:
        with open(website_folder + website) as f:
            raw_txt = f.readlines()
        long_txt = " ".join(raw_txt)
        long_txt = long_txt.replace("\n", "").lower()
        tweets.append(long_txt)
        party.append(party_dict[website])
    
    # Create the final dataframe
    df = pd.DataFrame({
        "username": username,
        "tweets": tweets,
        "party": party
    })

    return df


def parse_tweet_data():
    lookup = pd.read_csv(lookup_path)
    lookup["username"] = lookup["username"].apply(lambda x: x.lower())

    list_of_raw_files = os.listdir(raw_folder)

    df = pd.concat([pd.read_json(f, lines=True) for f in glob.glob(raw_folder + "/*.json")])

    # Go through each user, and combine their tweets to one long
    data_dict = dict()
    tweets_list = []
    label_list = []

    for username in df["username"].unique().tolist():
        conc_tweets = ""
        temp_df = df[df["username"] == username]
        for i in range(len(temp_df)):
            conc_tweets += temp_df.iloc[i, 10] + " " # 10 = tweet

        # Create dict with the training data
        #data_dict[username] = conc_tweets
        tweets_list.append(conc_tweets)

    data_dict["username"] = df["username"].unique().tolist()
    data_dict["tweets"] = tweets_list
    prep_data = pd.DataFrame(data=data_dict)

    # Set the labels
    for username in prep_data["username"]:
        print(username)
        temp_val = lookup[lookup["username"] == "@" + username]["party"].iat[0]
        label_list.append(temp_val)

    prep_data["party"] = label_list

    return prep_data

def create_training_data():
    website_text = parse_website_data()
    twitter_text = parse_tweet_data()

    prep_data = pd.concat([website_text, twitter_text])

    # Save the dataframe in the training folder
    prep_data.to_pickle("/home/chrolss/PycharmProjects/svpol-ml/data/processed/processed.pkl")

    return True
