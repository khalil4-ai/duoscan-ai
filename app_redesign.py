import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
from textblob import TextBlob
from wordcloud import WordCloud
from collections import Counter
from transformers import pipeline, CamembertTokenizerFast, AutoModelForSequenceClassification

st.set_page_config(page_title="DuoScan+ Analyse d'Avis", page_icon="🦉", layout="wide")

# --- CHARGEMENT CamemBERT ---
model_path = "./mon_model_personnalise"
try:
    tokenizer = CamembertTokenizerFast.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)
    camembert_ok = True
except Exception:
    camembert_ok = False

vader = SentimentIntensityAnalyzer()
lexique_pos = {"excellent","super","génial","utile","rapide","fluide","ludique","intéressant","facile","bien","bon","parfait"}
lexique_neg = {"nul","pénible","lent","bug","ennuyeux","répétitif","trop long","horrible","mauvais","pourri","nulle"}
emoji_map   = {"😍":"Positif","👍":"Positif","😐":"Neutre","😡":"Négatif","😴":"Négatif"}
mapping = {
    "LABEL_0":"Très Négatif","LABEL_1":"Négatif","LABEL_2":"Plutôt Négatif","LABEL_3":"Neutre","LABEL_4":"Positif",
    "1 star": "Très Négatif", "2 stars": "Négatif", "3 stars": "Neutre", "4 stars": "Positif", "5 stars": "Très Positif"
}

def analyse_hf(text):
    if not camembert_ok:
        return "Non dispo"
    try:
        out = classifier(text)[0]["label"]
        return mapping.get(out, out)
    except:
        return "Erreur"

def analyse_vader(text):
    try:
        eng = GoogleTranslator(source="fr", target="en").translate(text)
        score = vader.polarity_scores(eng)["compound"]
        return "Positif" if score>=.05 else ("Négatif" if score<=-.05 else "Neutre")
    except: return "Erreur"

def analyse_lexique(text):
    txt = re.sub(r"[^\w\s]", "", text.lower())
    if any(w in txt for w in lexique_neg): return "Négatif"
    if any(w in txt for w in lexique_pos): return "Positif"
    return "Neutre"

def analyse_textblob(text):
    try:
        eng = GoogleTranslator(source="fr", target="en").translate(text)
        pol = TextBlob(eng).sentiment.polarity
        return "Positif" if pol>0.1 else ("Négatif" if pol<-0.1 else "Neutre")
    except: return "Erreur"

def analyse_emoji(text):
    for ch in text:
        if ch in emoji_map: return emoji_map[ch]
    return "Neutre"

def generer_wordcloud(texts):
    corpus = " ".join(re.sub(r"[^\w\s]", "", t.lower()) for t in texts)
    if not corpus.strip():
        st.info("Pas assez de texte pour nuage de mots.")
        return
    wc = WordCloud(width=800, height=400, background_color="white").generate(corpus)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

st.title("🦉 DuoScan+ Analyse d'Avis")
st.markdown("Analyse de sentiments avec **VADER (graph/nuage)** + tableau multi-méthodes et micro-stats avancées.")

mode = st.sidebar.radio("Mode d’entrée", ["Texte unique","Fichier CSV"], index=0)

if mode=="Texte unique":
    avis = st.text_area("Saisissez votre avis:", height=100)
    if st.button("Analyser"):
        st.markdown("#### Tableau multi-méthodes")
        resultats = {
            "CamemBERT": analyse_hf(avis),
            "VADER": analyse_vader(avis),
            "Lexique": analyse_lexique(avis),
            "TextBlob": analyse_textblob(avis),
            "Emoji": analyse_emoji(avis)
        }
        df = pd.DataFrame([resultats])
        st.dataframe(df, use_container_width=True)

elif mode=="Fichier CSV":
    uploaded = st.file_uploader("Importer un CSV (colonne 'avis')", type="csv")
    if uploaded:
        import chardet, io
        uploaded.seek(0)
        raw = uploaded.read()
        encoding_detected = chardet.detect(raw)['encoding'] or 'utf-8'
        df = pd.read_csv(io.BytesIO(raw), encoding=encoding_detected)
        df.columns = df.columns.str.strip().str.lower()
        if "avis" not in df.columns:
            st.error("Colonne 'avis' introuvable.")
        else:
            st.success(f"{len(df)} avis chargés.")
            if st.button("🚀 Lancer l’analyse"):
                with st.spinner("Analyse en cours..."):
                    texts = df["avis"].dropna().astype(str).tolist()
                    st.session_state['texts'] = texts
                    résultats = []
                    for t in texts:
                        résultats.append({
                            "Avis": t,
                            "CamemBERT": analyse_hf(t),
                            "VADER": analyse_vader(t),
                            "Lexique": analyse_lexique(t),
                            "TextBlob": analyse_textblob(t),
                            "Emoji": analyse_emoji(t)
                        })
                df_res = pd.DataFrame(résultats)
                st.session_state['df_res'] = df_res
                st.dataframe(df_res, use_container_width=True)

                st.subheader("📊 Répartition VADER")
                dist = df_res["VADER"].value_counts().reset_index()
                dist.columns = ["Label","Count"]
                fig = px.bar(dist, x="Label", y="Count", color="Label", text="Count")
                st.plotly_chart(fig, use_container_width=True)

            # Récupère texts du session_state pour le nuage et les stats
            texts = st.session_state.get('texts', [])
            df_res = st.session_state.get('df_res', None)
            if st.checkbox("Afficher nuage de mots global"):
                if not texts or len(texts) == 0:
                    st.warning("Aucun texte à analyser pour le nuage de mots. Lance l'analyse d'abord !")
                else:
                    generer_wordcloud(texts)
                    # Mot le plus utilisé
                    mots = " ".join(texts).lower().split()
                    mots_filtrés = [m for m in mots if len(m)>2 and m not in {"les", "des", "une", "avec", "pour", "dans", "que", "qui", "pas", "est", "le", "la", "et", "en", "je"}]
                    mot_courant = Counter(mots_filtrés).most_common(1)
                    if mot_courant:
                        mot, occ = mot_courant[0]
                        st.info(f"🏅 Mot le plus utilisé : **{mot}** ({occ} fois)")
                    # Pourcentage mot-clé
                    mot_cle = st.text_input("Tape un mot-clé à analyser dans les avis :", "motivant")
                    if mot_cle:
                        nb = sum(mot_cle.lower() in t.lower() for t in texts)
                        pourcent = (nb * 100) // len(texts) if texts else 0
                        st.info(f"🔎 Le mot '**{mot_cle}**' apparaît dans **{pourcent}%** des avis ({nb} sur {len(texts)})")
                    # Taux questions/exclamations
                    nb_q = sum("?" in t for t in texts)
                    nb_ex = sum("!" in t for t in texts)
                    st.info(f"❓ {nb_q} avis ({(nb_q*100)//len(texts) if texts else 0}%) contiennent une question.")
                    st.info(f"❗ {nb_ex} avis ({(nb_ex*100)//len(texts) if texts else 0}%) contiennent une exclamation.")
                    # Top 3 positifs/négatifs
                    if df_res is not None:
                        st.subheader("🌟 Top 3 avis les plus positifs (VADER)")
                        for a in df_res[df_res["VADER"]=="Positif"]["Avis"].head(3):
                            st.success(a)
                        st.subheader("💔 Top 3 avis les plus négatifs (VADER)")
                        for a in df_res[df_res["VADER"]=="Négatif"]["Avis"].head(3):
                            st.error(a)

st.markdown("---")
st.caption("© 2025 • Conçu par Khalil JABER")
