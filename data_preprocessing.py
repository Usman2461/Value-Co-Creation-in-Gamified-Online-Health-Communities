# data_preprocessing.py
import pandas as pd
import re
import matplotlib.pyplot as plt

# ----------------------------
# 1. Data Loading & Cleaning
# ----------------------------
def clean_text(text):
    """Clean text data with proper null handling"""
    if pd.isna(text):
        return ''
    text = str(text)
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
    return text.lower().strip()

# Load datasets
posts = pd.read_csv("health_posts.csv", keep_default_na=False)
comments = pd.read_csv("health_comments.csv", keep_default_na=False)

# Merge posts with comments
merged_df = posts.merge(
    comments,
    on="post_id",
    how="left",
    suffixes=("_post", "_comment")
)

# Clean text columns
merged_df['cleaned_body'] = merged_df['body'].fillna('').apply(clean_text)
merged_df['cleaned_title'] = merged_df['title'].fillna('').apply(clean_text)

# ----------------------------
# 2. Feature Engineering
# ----------------------------
# Create gamification features
merged_df['has_badge'] = merged_df['cleaned_body'].str.contains(
    r'\b(badge|achievement|reward)\b',
    case=False,
    na=False
)

merged_df['has_leaderboard'] = merged_df['cleaned_body'].str.contains(
    r'\b(leaderboard|rank|score)\b',
    case=False,
    na=False
)

merged_df['altruistic_act'] = merged_df['cleaned_body'].str.contains(
    r'\b(help|support|share|advice)\b',
    case=False,
    na=False
)

# Create temporal features
merged_df['created_utc'] = pd.to_datetime(merged_df['created_utc_post'])
merged_df['hour_of_day'] = merged_df['created_utc'].dt.hour
merged_df['day_of_week'] = merged_df['created_utc'].dt.dayofweek

# ----------------------------
# 3. Exploratory Data Analysis
# ----------------------------
# Verify columns exist
print("\nCurrent columns:", merged_df.columns.tolist())

# 1. Sentiment Analysis
plt.figure(figsize=(10, 6))
merged_df.groupby('subreddit')['comment_sentiment'].mean().plot(kind='bar')
plt.title('Average Comment Sentiment by Subreddit')
plt.ylabel('Sentiment Score (-1 to 1)')
plt.tight_layout()
plt.savefig('sentiment_by_subreddit.png')

# 2. Gamification Statistics
gamification_stats = merged_df[['has_badge', 'has_leaderboard', 'altruistic_act']].mean()
print("\nGamification Prevalence:")
print(f"- Badges: {gamification_stats['has_badge']:.1%}")
print(f"- Leaderboards: {gamification_stats['has_leaderboard']:.1%}")
print(f"- Altruistic Acts: {gamification_stats['altruistic_act']:.1%}")

# 3. Temporal Analysis
plt.figure(figsize=(12, 6))
merged_df.groupby('hour_of_day')['num_comments'].mean().plot(marker='o')
plt.title('Average Comments per Post by Hour of Day')
plt.xlabel('Hour of Day (UTC)')
plt.ylabel('Average Comments')
plt.grid(True)
plt.tight_layout()
plt.savefig('engagement_by_hour.png')

print("\nPreprocessing completed successfully!")