import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from wordcloud import WordCloud
from collections import Counter
import matplotlib.pyplot as plt
import string
import nltk
import pandas as pd

# 🔄 Download required NLTK packages safely
nltk_packages = ['punkt', 'vader_lexicon', 'wordnet']
for pkg in nltk_packages:
    try:
        nltk.data.find(f'tokenizers/{pkg}' if pkg == 'punkt' else f'sentiment/{pkg}' if pkg == 'vader_lexicon' else f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg)

# 🧠 Lemmatizer
lemmatizer = WordNetLemmatizer()

# 🌟 Streamlit Setup
st.set_page_config(page_title="Sentiment & Emotion Analyzer", layout="centered")
st.title("🧠 Sentiment and Emotion Analysis App")

# Analyzer
analyzer = SentimentIntensityAnalyzer()

# Tabs for structured UI
tabs = st.tabs(["📄 Single Text Analysis", "📂 Batch CSV Analysis"])

# Utility: Load emotions from emotions.txt
def load_emotions():
    emotion_map = {}
    try:
        with open("emotions.txt", 'r') as file:
            for line in file:
                line = line.strip()
                if ':' in line:
                    parts = line.split(':')
                    if len(parts) == 2:
                        word, emotion = parts
                        emotion_map[word.strip()] = emotion.strip()
    except FileNotFoundError:
        st.warning("⚠️ emotions.txt not found. Please place it in the project folder.")
    return emotion_map

# Utility: Detect emotions smartly
def get_emotions_from_tokens(tokens, emotion_dict):
    detected_emotions = []
    for word in tokens:
        lemma = lemmatizer.lemmatize(word)
        if lemma in emotion_dict:
            detected_emotions.append(emotion_dict[lemma])
    return detected_emotions

# ================================================
# 📄 SINGLE TEXT ANALYSIS
# ================================================
with tabs[0]:
    uploaded_file_single = st.file_uploader("📂 Upload a .txt file", type=['txt'], key="single")
    text = ""

    if uploaded_file_single is not None:
        text = uploaded_file_single.read().decode("utf-8")
    else:
        text = st.text_area("✍️ Or enter your text here:")

    if st.button("🔍 Analyze Text", key="analyze_single"):
        if not text.strip():
            st.warning("⚠️ Please enter some text or upload a file.")
        else:
            score = analyzer.polarity_scores(text)
            st.subheader("📊 Sentiment Scores")
            st.json(score)

            if score['compound'] >= 0.05:
                st.success("Overall Sentiment: Positive 😊")
            elif score['compound'] <= -0.05:
                st.error("Overall Sentiment: Negative 😞")
            else:
                st.info("Overall Sentiment: Neutral 😐")

            # Clean + Tokenize
            text_clean = text.lower().translate(str.maketrans('', '', string.punctuation))
            tokens = word_tokenize(text_clean)
            emotion_dict = load_emotions()
            emotions = get_emotions_from_tokens(tokens, emotion_dict)

            st.subheader("🧠 Detected Emotions")
            if emotions:
                # Show emotions beautifully
                unique_emotions = list(set(emotions))
                for emo in unique_emotions:
                    st.markdown(f"<span style='background-color:#D1E8E4;padding:8px;border-radius:10px;margin:2px;display:inline-block;'>{emo.capitalize()}</span>", unsafe_allow_html=True)

                # Plot emotion counts
                emotion_count = Counter(emotions)
                fig, ax = plt.subplots()
                ax.bar(emotion_count.keys(), emotion_count.values(), color='skyblue')
                ax.set_title("Emotion Distribution")
                ax.set_xlabel("Emotions")
                ax.set_ylabel("Frequency")
                plt.xticks(rotation=45)
                st.pyplot(fig)

                if "sad" in emotions:
                    st.info("💡 You seem down. Here's a positive quote: *'Tough times don't last, tough people do.'*")
            else:
                st.info("No specific emotions detected.")

            # Subjectivity
            blob = TextBlob(text)
            st.subheader("🌀 Subjectivity")
            st.write(f"{blob.sentiment.subjectivity:.2f} (0 = Objective, 1 = Subjective)")

            # Word Cloud
            try:
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_clean)
                st.subheader("☁️ Word Cloud")
                st.image(wordcloud.to_array(), use_column_width=True)
            except:
                st.warning("⚠️ Word cloud generation failed (text might be too short).")

            # Save results
            single_result = pd.DataFrame({
                "Original Text": [text],
                "Compound Score": [score['compound']],
                "Emotions": [', '.join(unique_emotions) if emotions else "None"],
                "Subjectivity": [blob.sentiment.subjectivity]
            })

            csv = single_result.to_csv(index=False).encode('utf-8')
            st.download_button("📅 Download Report as CSV", csv, "sentiment_report.csv", "text/csv")

# ================================================
# 📟 BATCH CSV ANALYSIS
# ================================================
with tabs[1]:
    uploaded_file_batch = st.file_uploader("📂 Upload a CSV file for batch analysis", type=['csv'], key="batch")
    if uploaded_file_batch is not None:
        df = pd.read_csv(uploaded_file_batch)
        if df.empty:
            st.warning("⚠️ CSV file is empty.")
        else:
            results = []
            sad_detected = False
            emotion_dict = load_emotions()

            for row in df.iloc[:, 0]:
                row_text = str(row)
                row_score = analyzer.polarity_scores(row_text)
                row_blob = TextBlob(row_text)

                tokens = word_tokenize(row_text.lower().translate(str.maketrans('', '', string.punctuation)))
                row_emotions = get_emotions_from_tokens(tokens, emotion_dict)

                if "sad" in row_emotions:
                    sad_detected = True

                results.append({
                    "Text": row_text,
                    "Compound Score": row_score['compound'],
                    "Subjectivity": row_blob.sentiment.subjectivity,
                    "Emotions": ', '.join(list(set(row_emotions))) if row_emotions else "None"
                })

            result_df = pd.DataFrame(results)
            st.subheader("📋 Analysis Results")
            st.dataframe(result_df)

            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("📅 Download Full Analysis CSV", csv, "full_analysis.csv", "text/csv")

            emotion_counts = Counter([emo for r in results for emo in r['Emotions'].split(', ') if emo != "None"])
            if emotion_counts:
                st.subheader("📊 Overall Emotion Summary")
                st.write(emotion_counts)
                fig, ax = plt.subplots()
                ax.bar(emotion_counts.keys(), emotion_counts.values(), color='lightgreen')
                ax.set_title("Overall Emotion Frequency")
                ax.set_xlabel("Emotions")
                ax.set_ylabel("Count")
                plt.xticks(rotation=45)
                st.pyplot(fig)

            if sad_detected:
                st.info("💡 Some texts indicate sadness. Here's a positive quote: *'After every storm comes a rainbow.'*")
    else:
        st.info("📅 Upload a CSV file to begin batch sentiment and emotion analysis.")
