import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
from textblob import TextBlob
from statsmodels.tsa.arima.model import ARIMA

# Sample dataset (Replace this with real-world data)
sample_data = {
    "text": [
        "Nike introduced a new AI-powered running shoe with smart sensors.",
        "Adidas' latest wearables integrate biometric tracking for performance optimization.",
        "The sports industry is moving towards sustainable materials like recycled polyester.",
        "Under Armour is exploring AI-driven coaching tools to enhance training.",
        "Smart fitness gear, including AI-enhanced compression wear, is gaining popularity.",
        "Wearable technology in sports continues to evolve, focusing on injury prevention.",
        "Augmented Reality (AR) is being integrated into sports gear for immersive training.",
        "Consumers are increasingly demanding sustainable and customizable sportswear.",
        "AI-driven injury prevention gear is helping athletes recover faster and train smarter.",
        "New smart helmets with real-time impact sensors are improving safety in contact sports.",
    ]
}

df = pd.DataFrame(sample_data)

# Preprocessing: Lowercasing text
df["processed_text"] = df["text"].str.lower()

# TF-IDF Analysis to Identify Key Terms
vectorizer = TfidfVectorizer(stop_words="english", max_features=10)
tfidf_matrix = vectorizer.fit_transform(df["processed_text"])
keywords = vectorizer.get_feature_names_out()

# Topic Modeling using LDA
lda_model = LatentDirichletAllocation(n_components=2, random_state=42)
lda_matrix = lda_model.fit_transform(tfidf_matrix)

# Sentiment Analysis
df["sentiment"] = df["processed_text"].apply(lambda text: TextBlob(text).sentiment.polarity)

# Visualization: Word Cloud
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(df["processed_text"]))

# Plot Word Cloud & Sentiment Trends
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Key Trends")

plt.subplot(1, 2, 2)
sns.lineplot(data=df, x=df.index, y="sentiment", marker="o")
plt.xticks(df.index, labels=df["text"], rotation=90)
plt.title("Sentiment Trends in Sports Gear")

plt.tight_layout()
plt.show()

# Time-Series Trend Prediction
df["date"] = pd.date_range(start="2020-01-01", periods=len(df), freq="M")
df.set_index("date", inplace=True)

# ARIMA Forecasting
model = ARIMA(df["sentiment"], order=(2, 1, 2))
model_fit = model.fit()

# Predict future sentiment trends for the next 6 months
future_dates = pd.date_range(start=df.index[-1], periods=7, freq="M")[1:]
forecast = model_fit.forecast(steps=6)

# Create a DataFrame for predictions
forecast_df = pd.DataFrame({"date": future_dates, "predicted_sentiment": forecast})
forecast_df.set_index("date", inplace=True)

# Plot Sentiment Trend with Forecast
plt.figure(figsize=(10, 5))
plt.plot(df.index, df["sentiment"], marker="o", label="Historical Sentiment")
plt.plot(forecast_df.index, forecast_df["predicted_sentiment"], marker="s", linestyle="dashed", color="red", label="Predicted Sentiment")
plt.axvline(df.index[-1], color="gray", linestyle="--", label="Forecast Start")
plt.xlabel("Date")
plt.ylabel("Sentiment Score")
plt.title("Sentiment Trend Prediction for Sports Gear Innovations")
plt.legend()
plt.grid(True)
plt.show()
