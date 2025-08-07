import nltk
nltk.download('vader_lexicon')

import praw
import requests
import pandas as pd
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon (only once)
nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

reddit = praw.Reddit(
    client_id="c0vkYpFsdow9UxEW2Rd2KA",
    client_secret="UXfF9ORkQzOfNHaXahDv8HDxpDvEZQ",
    user_agent="ResearchDataBot/1.0 by /u/WildLoad5149",
)

def analyze_sentiment(text):
    if text:
        scores = sia.polarity_scores(text)
        return scores['compound']  # Compound score: -1 (negative) to +1 (positive)
    return 0.0

def fetch_posts_and_comments(subreddits, post_limit=10, comment_limit=10):
    all_posts = []
    all_comments = []

    for subreddit_name in subreddits:
        subreddit = reddit.subreddit(subreddit_name)
        print(f"Fetching posts from r/{subreddit_name}...")

        for post in subreddit.new(limit=post_limit):
            post_data = {
                "subreddit": subreddit_name,
                "post_id": post.id,
                "title": post.title,
                "score": post.score,
                "num_comments": post.num_comments,
                "created_utc": datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                "author": str(post.author) if post.author else "[deleted]",
                "awards": post.total_awards_received,
                "title_sentiment": analyze_sentiment(post.title)
            }
            all_posts.append(post_data)

            post.comments.replace_more(limit=0)  # Fetch all comments without "MoreComments"
            comments = post.comments.list()[:comment_limit]

            for comment in comments:
                comment_data = {
                    "post_id": post.id,
                    "comment_id": comment.id,
                    "author": str(comment.author) if comment.author else "[deleted]",
                    "created_utc": datetime.utcfromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                    "body": comment.body,
                    "score": comment.score,
                    "comment_sentiment": analyze_sentiment(comment.body)
                }
                all_comments.append(comment_data)

    posts_df = pd.DataFrame(all_posts)
    comments_df = pd.DataFrame(all_comments)

    return posts_df, comments_df


# Define subreddits of interest
health_subreddits = ["diabetes", "mentalhealth", "ChronicPain", "Fitness"]

posts_df, comments_df = fetch_posts_and_comments(health_subreddits, post_limit=10, comment_limit=10)

# Save to CSV files
posts_df.to_csv("health_posts.csv", index=False)
comments_df.to_csv("health_comments.csv", index=False)

print("Data collection complete. Posts and comments saved to CSV files.")
