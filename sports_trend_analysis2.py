import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from statsmodels.tsa.arima.model import ARIMA

# **STEP 1: Scraping Sports Gear Industry News (Google News)**
def scrape_sports_trends():
    url = "https://news.google.com/search?q=sports+gear+innovation&hl=en-US&gl=US&ceid=US:en"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    
    articles = soup.find_all("h3")
    
    news_data = []
    for article in articles[:10]:  # Scrape top 10 articles
        title = article.text
        link = "https://news.google.com" + article.a["href"][1:]
        news_data.append({"title": title, "link": link})
    
    return pd.DataFrame(news_data)

# Fetch sports gear news
news_df = scrape_sports_trends()
print("\n📰 Scraped Sports Gear News:\n", news_df)

# **STEP 2: Preprocessing - Clean Text**
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    return text

news_df["cleaned_text"] = news_df["title"].apply(clean_text)

# **STEP 3: NLP Analysis - TF-IDF & Topic Modeling (LDA)**
vectorizer = TfidfVectorizer(stop_words="english", max_features=10)
tfidf_matrix = vectorizer.fit_transform(news_df["cleaned_text"])
keywords = vectorizer.get_feature_names_out()

lda_model = LatentDirichletAllocation(n_components=2, random_state=42)
lda_matrix = lda_model.fit_transform(tfidf_matrix)

# **STEP 4: Sentiment Analysis**
news_df["sentiment"] = news_df["cleaned_text"].apply(lambda text: TextBlob(text).sentiment.polarity)

# **STEP 5: Time-Series Prediction (ARIMA)**
# Simulating historical sentiment data (Ideally, use real-time collected data)
news_df["date"] = pd.date_range(start="2020-01-01", periods=len(news_df), freq="6M")
news_df.set_index("date", inplace=True)

# Train ARIMA model to predict future sentiment trends
model = ARIMA(news_df["sentiment"], order=(2, 1, 2))
model_fit = model.fit()

# Forecast sentiment for the next 6 months
future_dates = pd.date_range(start=news_df.index[-1], periods=7, freq="6M")[1:]
forecast = model_fit.forecast(steps=6)

# Create Forecast DataFrame
forecast_df = pd.DataFrame({"date": future_dates, "predicted_sentiment": forecast})
forecast_df.set_index("date", inplace=True)

# **STEP 6: Visualizations**
plt.figure(figsize=(12, 6))

# **Word Cloud**
plt.subplot(2, 2, 1)
wordcloud = WordCloud(width=400, height=200, background_color="white").generate(" ".join(news_df["cleaned_text"]))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud: Sports Gear Trends")

# **Sentiment Trend Analysis**
plt.subplot(2, 2, 2)
sns.lineplot(data=news_df, x=news_df.index, y="sentiment", marker="o", label="Historical Sentiment")
sns.lineplot(data=forecast_df, x=forecast_df.index, y="predicted_sentiment", marker="s", linestyle="dashed", color="red", label="Predicted Sentiment")
plt.axvline(news_df.index[-1], color="gray", linestyle="--", label="Forecast Start")
plt.xlabel("Year")
plt.ylabel("Sentiment Score")
plt.title("Sentiment Trends in Sports Gear Innovations")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# **Results Display**
print("\n### Extracted Keywords (TF-IDF):", keywords)
print("\n### Forecasted Sentiment Trends:\n", forecast_df)


# 📊 What This Code Does
# ✅ Scrapes Latest Sports Gear News from Google News
# ✅ Cleans Text & Extracts Keywords (TF-IDF & LDA)
# ✅ Performs Sentiment Analysis to track industry perception
# ✅ Predicts Future Trends for the next 6 months (ARIMA model)
# ✅ Visualizes Insights using:

# Word Cloud (top discussed topics)
# Sentiment Trends (historical & predicted values)
# 🚀 Next Steps
# Expand Data Sources: Integrate Statista API, Twitter API, Deloitte Reports.
# Use Deep Learning (Transformers): Apply BERT/GPT-based models for advanced trend analysis.
# Improve Forecasting: Use Facebook Prophet for better time-series predictions.
