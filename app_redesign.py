import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
from textblob import TextBlob
from wordcloud import WordCloud
import torch
import transformers

st.title("✅ Test Environnement DuoScan AI")

# Vérification rapide
st.write("### 🟢 Import des modules réussis !")

# Test VADER
analyzer = SentimentIntensityAnalyzer()
test_sentence = "Duolingo est super motivant !"
score = analyzer.polarity_scores(test_sentence)
st.write(f"**Test VADER:** {test_sentence} → {score}")

# Test traduction
trad = GoogleTranslator(source="fr", target="en").translate("Ceci est un test de traduction automatique.")
st.write(f"**Test Deep Translator:** Traduction → {trad}")

# Test TextBlob
blob = TextBlob("This is a wonderful app!")
st.write(f"**Test TextBlob:** Polarité → {blob.sentiment.polarity}")

# Test WordCloud
textes = "amour amour amour joie motivation progrès progrès progrès"
wordcloud = WordCloud(width=400, height=200).generate(textes)
fig, ax = plt.subplots()
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)
st.write("**Test WordCloud :** OK")

# Test Plotly
df = pd.DataFrame({'sentiment': ['Positif', 'Négatif', 'Neutre'], 'valeur': [10, 5, 3]})
fig2 = px.bar(df, x='sentiment', y='valeur', color='sentiment', title="Test Plotly Bar")
st.plotly_chart(fig2)
st.write("**Test Plotly :** OK")

# Test torch + transformers
st.write(f"**Test torch version :** {torch.__version__}")
st.write(f"**Test transformers version :** {transformers.__version__}")

st.success("🎉 Tout fonctionne ! Si cette page s'affiche sans erreur, tu peux utiliser tous ces modules dans ton projet.")
