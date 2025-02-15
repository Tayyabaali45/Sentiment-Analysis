import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
train_path = "/mnt/data/train.csv"
test_path = "/mnt/data/test (1).csv"

train_df = pd.read_csv(train_path, encoding='ISO-8859-1')
test_df = pd.read_csv(test_path, encoding='ISO-8859-1')
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(f"[{string.punctuation}]", "", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"http\S+", "", text)
    return text

train_df['clean_text'] = train_df['text'].apply(clean_text)
test_df['clean_text'] = test_df['text'].apply(clean_text)
train_df.dropna(subset=['clean_text', 'sentiment'], inplace=True)
sns.countplot(data=train_df, x='sentiment', palette='coolwarm')
plt.title("Sentiment Distribution")
plt.show()
pos_text = " ".join(train_df[train_df['sentiment'] == 'positive']['clean_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(pos_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Most Common Words in Positive Sentiment")
plt.show()

X = train_df['clean_text']
y = train_df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('classifier', MultinomialNB())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
