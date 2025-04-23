import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from wordcloud import WordCloud
from collections import Counter
import matplotlib.pyplot as plt
import string
import nltk
import pandas as pd

# 🔄 Safe download of required NLTK packages
nltk_packages = ['punkt', 'vader_lexicon']
for pkg in nltk_packages:
    try:
        nltk.data.find(f'tokenizers/{pkg}' if pkg == 'punkt' else f'sentiment/{pkg}')
    except LookupError:
        nltk.download(pkg)

# 🌟 Streamlit App Setup
st.set_page_config(page_title="Sentiment & Emotion Analyzer", layout="centered")
st.title("🧠 Sentiment and Emotion Analysis App")

# 📤 File Upload or Text Input
uploaded_file = st.file_uploader("📂 Upload a .txt or .csv file", type=['txt', 'csv'])
text = ""

if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        text = ' '.join(df.iloc[:, 0].astype(str))
else:
    text = st.text_area("✍️ Or enter your text here:")

# 🔍 Analyze Button
if st.button("🔍 Analyze"):
    if not text.strip():
        st.warning("⚠️ Please provide text input or upload a file.")
    else:
        # 📊 Sentiment Analysis
        analyzer = SentimentIntensityAnalyzer()
        score = analyzer.polarity_scores(text)

        st.subheader("📊 Sentiment Scores")
        st.json(score)

        if score['compound'] >= 0.05:
            st.success("Overall Sentiment: Positive 😊")
        elif score['compound'] <= -0.05:
            st.error("Overall Sentiment: Negative 😞")
        else:
            st.info("Overall Sentiment: Neutral 😐")

        # ✨ Clean text and tokenize
        text_clean = text.lower().translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text_clean)
        emotions = []

        # 🧠 Emotion Detection
        try:
            with open("emotions.txt", 'r') as file:
                for line in file:
                    word, emotion = line.strip().split(':')
                    if word in tokens:
                        emotions.append(emotion)

            st.subheader("🧠 Detected Emotions")
            if emotions:
                st.write(emotions)

                # 📈 Plot Emotion Counts
                emotion_count = Counter(emotions)
                fig, ax = plt.subplots()
                ax.bar(emotion_count.keys(), emotion_count.values(), color='skyblue')
                ax.set_title("Emotion Distribution")
                ax.set_xlabel("Emotions")
                ax.set_ylabel("Frequency")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.info("No specific emotions detected.")

        except FileNotFoundError:
            st.warning("⚠️ emotions.txt not found. Please make sure it's in the same folder.")

        # 🔍 Subjectivity Score
        blob = TextBlob(text)
        st.subheader("🌀 Subjectivity")
        st.write(f"{blob.sentiment.subjectivity:.2f} (0 = Objective, 1 = Subjective)")

        # ☁️ Word Cloud
        try:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_clean)
            st.subheader("☁️ Word Cloud")
            st.image(wordcloud.to_array(), use_column_width=True)
        except:
            st.warning("⚠️ Word cloud generation failed (text might be too short).")
