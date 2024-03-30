import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# Read the Twitter dataset
tweets_df = pd.read_csv('twitter_dataset.csv', encoding='latin1')

# Perform sentiment analysis using TextBlob
def analyze_sentiment(tweet):
    analysis = TextBlob(tweet)
    # Sentiment polarity ranges from -1 (negative) to 1 (positive)
    return analysis.sentiment.polarity

# Add a new column to the dataframe with sentiment polarity
tweets_df['Sentiment'] = tweets_df['Tweet'].apply(analyze_sentiment)

# Calculate percentage of positive and negative sentiments
positive_percentage = (tweets_df['Sentiment'] > 0).mean() * 100
negative_percentage = (tweets_df['Sentiment'] <= 0).mean() * 100

# Plotting sentiment distribution
plt.figure(figsize=(10, 6))

# Line plot for sentiment polarity
plt.subplot(1, 2, 1)
plt.plot(tweets_df.index, tweets_df['Sentiment'], color='skyblue')
plt.title('Sentiment Polarity Over Tweets')
plt.xlabel('Tweet Index')
plt.ylabel('Sentiment Polarity')

# Bar plot for sentiment categories
plt.subplot(1, 2, 2)
sentiment_counts = tweets_df['Sentiment'].apply(lambda x: 'Positive' if x > 0 else ('Neutral' if x == 0 else 'Negative')).value_counts()
sentiment_counts.plot(kind='bar', color=['green', 'gray', 'red'])
plt.title('Sentiment Category Distribution')
plt.xlabel('Sentiment Category')
plt.ylabel('Frequency')

# Adjust layout
plt.tight_layout()

# Show plots
plt.show()

# Print percentage of positive and negative sentiments
print("Percentage of Positive Sentiments: {:.2f}%".format(positive_percentage))
print("Percentage of Negative Sentiments: {:.2f}%".format(negative_percentage))
