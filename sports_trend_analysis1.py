import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from statsmodels.tsa.arima.model import ARIMA

# Sample Dataset (Replace with real-world scraped reports)
data = {
    "date": pd.date_range(start="2020-01-01", periods=10, freq="6M"),
    "text": [
        "Nike introduced AI-powered running shoes with smart sensors.",
        "Adidas wearables now integrate biometric tracking for athletes.",
        "The industry is shifting towards sustainable materials like recycled polyester.",
        "Under Armour is developing AI-driven training and recovery gear.",
        "Smart fitness equipment, including AI-enhanced compression wear, is popular.",
        "Wearable technology is evolving for enhanced injury prevention.",
        "Augmented Reality (AR) is revolutionizing immersive sports training.",
        "Consumers demand eco-friendly and customizable sportswear options.",
        "AI-driven injury prevention gear is making athletes safer and stronger.",
        "Smart helmets with real-time impact sensors are improving safety in contact sports."
    ]
}

df = pd.DataFrame(data)

# **Step 1: Preprocessing - Clean Text**
df["processed_text"] = df["text"].str.lower()

# **Step 2: Sentiment Analysis**
df["sentiment"] = df["processed_text"].apply(lambda text: TextBlob(text).sentiment.polarity)

# **Step 3: TF-IDF Analysis (Extracting Key Terms)**
vectorizer = TfidfVectorizer(stop_words="english", max_features=10)
tfidf_matrix = vectorizer.fit_transform(df["processed_text"])
keywords = vectorizer.get_feature_names_out()

# **Step 4: Topic Modeling (LDA)**
lda_model = LatentDirichletAllocation(n_components=2, random_state=42)
lda_matrix = lda_model.fit_transform(tfidf_matrix)

# **Step 5: Trend Prediction (ARIMA Time-Series)**
df.set_index("date", inplace=True)
model = ARIMA(df["sentiment"], order=(2, 1, 2))
model_fit = model.fit()

# Forecast future trends for the next 6 months
future_dates = pd.date_range(start=df.index[-1], periods=7, freq="6M")[1:]
forecast = model_fit.forecast(steps=6)

# Create Forecast DataFrame
forecast_df = pd.DataFrame({"date": future_dates, "predicted_sentiment": forecast})
forecast_df.set_index("date", inplace=True)

# **Step 6: Visualization**
plt.figure(figsize=(12, 6))

# **Word Cloud**
plt.subplot(2, 2, 1)
wordcloud = WordCloud(width=400, height=200, background_color="white").generate(" ".join(df["processed_text"]))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud: Sports Gear Trends")

# **Sentiment Analysis Trend**
plt.subplot(2, 2, 2)
sns.lineplot(data=df, x=df.index, y="sentiment", marker="o", label="Historical Sentiment")
sns.lineplot(data=forecast_df, x=forecast_df.index, y="predicted_sentiment", marker="s", linestyle="dashed", color="red", label="Predicted Sentiment")
plt.axvline(df.index[-1], color="gray", linestyle="--", label="Forecast Start")
plt.xlabel("Year")
plt.ylabel("Sentiment Score")
plt.title("Sentiment Trends in Sports Gear Innovations")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Display Extracted Trends
print("\n### Extracted Keywords (TF-IDF):", keywords)
print("\n### Forecasted Sentiment Trends:\n", forecast_df)


# Next Steps
# Integrate Real-World Reports: Connect APIs from Statista, Deloitte, Reuters.
# Refine Topic Modeling: Increase LDA topics for better categorization.
# Enhance Prediction Accuracy: Use Prophet or Deep Learning Models.