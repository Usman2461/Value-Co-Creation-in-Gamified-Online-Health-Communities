import requests
import pandas as pd
from datetime import datetime

def fetch_posts(subreddit, size=1000):
    url = "https://api.pushshift.io/reddit/search/submission/"
    params = {
        "subreddit": subreddit,
        "size": size,  # Max 1000 per request
        "fields": "title,selftext,author,created_utc,score,num_comments,awards,permalink"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()["data"]
        df = pd.DataFrame(data)
        # Convert Unix timestamp to readable date
        df["created_utc"] = df["created_utc"].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d'))
        return df
    else:
        print("Failed to fetch data.")
        return pd.DataFrame()

# Example: Fetch 1000 posts from r/diabetes
diabetes_posts = fetch_posts("diabetes", size=1000)
diabetes_posts.to_csv("diabetes_posts.csv", index=False)

def fetch_comments(subreddit, size=1000):
    url = "https://api.pushshift.io/reddit/search/comment/"
    params = {
        "subreddit": subreddit,
        "size": size,
        "fields": "body,author,created_utc,score,permalink"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()["data"]
        df = pd.DataFrame(data)
        df["created_utc"] = df["created_utc"].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d'))
        return df
    else:
        print("Failed to fetch comments.")
        return pd.DataFrame()

# Example: Fetch 1000 comments from r/diabetes
diabetes_comments = fetch_comments("diabetes", size=1000)
diabetes_comments.to_csv("diabetes_comments.csv", index=False)

posts = pd.read_csv("diabetes_posts.csv")
comments = pd.read_csv("diabetes_comments.csv")

# Aggregate comments per post
comment_counts = comments["permalink"].value_counts().reset_index()
comment_counts.columns = ["permalink", "comment_count"]

# Merge with posts
merged_data = posts.merge(comment_counts, on="permalink", how="left")
merged_data.to_csv("merged_ohc_data.csv", index=False)

