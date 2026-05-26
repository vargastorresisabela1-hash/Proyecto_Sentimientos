import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import requests
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os

import pandas as pd

df = pd.read_excel("Opiniones.xlsx")
print(df.head())

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Análisis de Opiniones — Hábitos Atómicos",
    page_icon="📚",
    layout="wide",
)

# ── Stopwords españolas (integradas, sin NLTK) ───────────────────────────────
STOPWORDS_ES = set("""
a al algo algunas algunos ante antes como con contra cual cuando de del desde
donde durante e el ella ellas ellos en entre era eras eramos eran eres es esa
ese eso esta este esto fue fueron fue fuera han has hay he her him his hizo
hoy io la las le les lo los mas me mi mis mo muy más ni no nos nu nuestro
o os para pero poco por que quien sea sed ser si sin sino sobre su sus también
tan tanto te tengo ti toda todo todos tu tus un una uno unos ya yo él él
había habían hacer hacia hasta hubo igual incluso junto largo le les lo los
mediante mejor mismo mucho nada ni no nos nueva nuevo o os parece pero poco
además aunque bien cada casi como con cómo cuando cuándo donde durante
él ella ellos en entre era estar este eso fue gracias gran grande hay hizo
igual junto la las le les lo los más mismo mucho nada ni no nos nunca o os
para pero poco por porque quien se sea según ser si sin sino sobre su sus
tal también tan tanto te tiene todo tu un una unos ya yo
""".split())

EXTRA_STOPS = {
    "libro", "libros", "autor", "autora", "leer", "lectura", "leer",
    "leído", "página", "páginas", "capítulo", "capítulos", "vez", "veces",
    "hace", "hacer", "puede", "pueden", "ser", "solo", "aún", "así",
    "parte", "forma", "bien", "mal", "muy", "más", "menos", "ahora",
    "aunque", "quizá", "cierto", "buenos", "buenas", "solo", "tras",
    "nuevo", "nuevos", "nueva", "nuevas"
}
STOPWORDS_ES.update(EXTRA_STOPS)

# ── Spanish suffix-based lemmatizer (no external model) ──────────────────────
def simple_lemmatize(word):
    w = word.lower()
    suffixes = [
        ("aciones","ar"),("ación","ar"),("ando","ar"),("ado","ar"),
        ("ando","ar"),("aron","ar"),("ará","ar"),("arán","ar"),
        ("iendo","er"),("ido","er"),("ieron","er"),
        ("mente",""),
        ("ísimo","o"),("ísima","a"),("ísimos","os"),
        ("dades","dad"),("dad","dad"),
        ("emente","e"),
        ("icos","ico"),("icas","ica"),
        ("osos","oso"),("osas","osa"),
        ("entes","ente"),("antes","ante"),
    ]
    for suf, rep in suffixes:
        if w.endswith(suf) and len(w) - len(suf) > 3:
            return w[:-len(suf)] + rep
    return w

def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-záéíóúüñ\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 3 and t not in STOPWORDS_ES]
    tokens = [simple_lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in STOPWORDS_ES and len(t) > 3]
    return tokens

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    raw = pd.read_excel("Opiniones.xlsx", header=None)
    # Row 1 = headers, rows 2+ = data; columns 2=opinion, 3=stars
    df = raw.iloc[2:].copy()
    df = df[[1, 2, 3]].copy()
    df.columns = ["numero", "opinion", "estrellas"]
    df = df.dropna(subset=["opinion"])
    df["estrellas"] = pd.to_numeric(df["estrellas"], errors="coerce")
    df["numero"] = pd.to_numeric(df["numero"], errors="coerce").astype(int)
    # Classify by stars
    def classify(s):
        if s >= 4: return "Positiva (4-5 ⭐)"
        if s == 3: return "Neutra (3 ⭐)"
        return "Negativa (1-2 ⭐)"
    df["clase"] = df["estrellas"].apply(classify)
    df = df.reset_index(drop=True)
    return df

df_all = load_data()

# ── Groq API call ──────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")

def call_claude(prompt, system="Eres un analizador de sentimientos experto en español."):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    body = {
        "model": "llama-3.3-70b-versatile",
        "max_tokens": 4000,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
    }
    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                          headers=headers, json=body, timeout=60)
        data = r.json()
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        else:
            return f"ERROR_GROQ: {data}"
    except Exception as e:
        return f"ERROR: {e}"


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/book.png", width=80)
st.sidebar.title("📊 Filtros")

clases = ["Todas"] + sorted(df_all["clase"].unique().tolist())
filtro_clase = st.sidebar.selectbox("Filtrar por clase de opinión:", clases)

if filtro_clase == "Todas":
    df = df_all.copy()
else:
    df = df_all[df_all["clase"] == filtro_clase].copy()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Opiniones seleccionadas:** {len(df)} de {len(df_all)}")
st.sidebar.markdown("**Distribución de estrellas:**")
for c, cnt in df_all["clase"].value_counts().items():
    st.sidebar.markdown(f"- {c}: {cnt}")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📚 Análisis de Opiniones — *Hábitos Atómicos*")
st.markdown("Análisis de texto y sentimientos sobre las 30 reseñas del libro en Amazon.")
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "☁️ Análisis de Texto",
    "📈 Gráfico Adicional",
    "🤖 Clasificación LLM",
    "💬 Nuevo Comentario"
])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — Word Cloud + Top 10 palabras
# ═══════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader(f"Análisis de palabras — {filtro_clase}")

    all_tokens = []
    for text in df["opinion"].dropna():
        all_tokens.extend(clean_and_tokenize(str(text)))

    if not all_tokens:
        st.warning("No hay suficiente texto para analizar.")
    else:
        freq = Counter(all_tokens)
        top10 = freq.most_common(10)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ☁️ Nube de Palabras")
            wc_text = " ".join(all_tokens)
            wc = WordCloud(
                width=700, height=400,
                background_color="white",
                colormap="viridis",
                max_words=80,
                prefer_horizontal=0.8,
            ).generate(wc_text)
            fig_wc, ax = plt.subplots(figsize=(7, 4))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig_wc)

        with col2:
            st.markdown("### 📊 Top 10 Palabras Más Frecuentes")
            words_df = pd.DataFrame(top10, columns=["Palabra", "Frecuencia"])
            fig_bar = px.bar(
                words_df,
                x="Frecuencia", y="Palabra",
                orientation="h",
                color="Frecuencia",
                color_continuous_scale="Viridis",
                text="Frecuencia",
            )
            fig_bar.update_layout(
                yaxis=dict(categoryorder="total ascending"),
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=10, b=10),
                height=400,
            )
            fig_bar.update_traces(textposition="outside")
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("#### 📋 Tokens procesados (muestra):")
        st.caption(", ".join(all_tokens[:50]) + "...")

# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — Gráfico adicional: distribución de longitud + scatter
# ═══════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📈 Análisis adicional — Longitud vs. Calificación")

    df_plot = df.copy()
    df_plot["num_palabras"] = df_plot["opinion"].apply(
        lambda x: len(str(x).split()) if pd.notna(x) else 0
    )
    df_plot["num_tokens"] = df_plot["opinion"].apply(
        lambda x: len(clean_and_tokenize(str(x))) if pd.notna(x) else 0
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📏 Longitud de reseñas por calificación")
        fig_box = px.box(
            df_plot, x="estrellas", y="num_palabras",
            color="clase",
            labels={"estrellas": "Estrellas", "num_palabras": "Nº de palabras", "clase": "Clase"},
            color_discrete_map={
                "Positiva (4-5 ⭐)": "#2ecc71",
                "Neutra (3 ⭐)": "#f39c12",
                "Negativa (1-2 ⭐)": "#e74c3c"
            },
        )
        fig_box.update_layout(margin=dict(t=20))
        st.plotly_chart(fig_box, use_container_width=True)
        st.caption("Las reseñas positivas suelen ser más largas, lo que sugiere mayor entusiasmo.")

    with col2:
        st.markdown("### 🔵 Dispersión: palabras vs. calificación")
        fig_sc = px.scatter(
            df_plot, x="num_palabras", y="estrellas",
            color="clase",
            size="num_palabras",
            hover_data={"opinion": True, "num_palabras": True},
            labels={"num_palabras": "Nº de palabras", "estrellas": "Estrellas"},
            color_discrete_map={
                "Positiva (4-5 ⭐)": "#2ecc71",
                "Neutra (3 ⭐)": "#f39c12",
                "Negativa (1-2 ⭐)": "#e74c3c"
            },
        )
        fig_sc.update_layout(margin=dict(t=20))
        st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("### 📊 Distribución de estrellas")
    dist_df = df_all["estrellas"].value_counts().reset_index()
    dist_df.columns = ["Estrellas", "Cantidad"]
    dist_df = dist_df.sort_values("Estrellas")
    fig_dist = px.bar(
        dist_df, x="Estrellas", y="Cantidad",
        color="Cantidad", color_continuous_scale="RdYlGn",
        text="Cantidad",
    )
    fig_dist.update_layout(coloraxis_showscale=False, xaxis=dict(tickmode="linear"))
    st.plotly_chart(fig_dist, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# TAB 3 — Clasificación LLM
# ═══════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🤖 Clasificación de Sentimientos con Claude")
    st.markdown(
        "Usa el modelo Claude para clasificar cada opinión como "
        "**Positivo**, **Negativo** o **Neutro** y explicar brevemente el razonamiento."
    )

    if "llm_results" not in st.session_state:
        st.session_state.llm_results = None

    if st.button("🚀 Clasificar las 30 opiniones con IA", type="primary"):
        opiniones_texto = ""
        for _, row in df_all.iterrows():
            opiniones_texto += f"[{int(row['numero'])}] {row['opinion']}\n\n"

        system_prompt = """Eres un analizador de sentimientos experto en español.
Clasifica CADA opinión como Positivo, Negativo o Neutro.
Responde ÚNICAMENTE con JSON válido, sin markdown ni backticks.
Formato exacto:
[
  {"numero": 1, "sentimiento": "Positivo", "razon": "breve razón en español"},
  ...
]
"""
        prompt = f"Clasifica el sentimiento de cada una de estas 30 opiniones sobre el libro 'Hábitos Atómicos':\n\n{opiniones_texto}"

        with st.spinner("Analizando con Claude... puede tardar unos segundos ⏳"):
            raw = call_claude(prompt, system=system_prompt)

        try:
            # Strip potential markdown fences
            clean = re.sub(r"```json|```", "", raw).strip()
            results = json.loads(clean)
            st.session_state.llm_results = results
        except Exception as e:
            st.error(f"Error al parsear respuesta: {e}\n\nRespuesta raw:\n{raw[:500]}")
            st.session_state.llm_results = None

    if st.session_state.llm_results:
        results = st.session_state.llm_results
        res_df = pd.DataFrame(results)
        # Merge with original
        merged = df_all.merge(res_df, left_on="numero", right_on="numero", how="left")
        merged_display = merged[["numero","opinion","estrellas","clase","sentimiento","razon"]].copy()
        merged_display.columns = ["#","Opinión","⭐","Clase (original)","Sentimiento IA","Razón"]

        # Color map
        color_map = {"Positivo": "#d4edda", "Negativo": "#f8d7da", "Neutro": "#fff3cd"}

        def highlight_sent(row):
            color = color_map.get(row["Sentimiento IA"], "white")
            return [f"background-color: {color}" for _ in row]

        st.markdown("#### 📋 Tabla de resultados")
        st.dataframe(
            merged_display.style.apply(highlight_sent, axis=1),
            use_container_width=True,
            height=500,
        )

        # Pie chart
        sent_counts = res_df["sentimiento"].value_counts().reset_index()
        sent_counts.columns = ["Sentimiento", "Cantidad"]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🥧 Distribución de sentimientos (IA)")
            fig_pie = px.pie(
                sent_counts, values="Cantidad", names="Sentimiento",
                color="Sentimiento",
                color_discrete_map={
                    "Positivo": "#2ecc71",
                    "Negativo": "#e74c3c",
                    "Neutro": "#f39c12"
                },
                hole=0.4,
            )
            fig_pie.update_traces(textinfo="percent+label+value")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.markdown("#### 📊 Comparación: Estrellas vs. Sentimiento IA")
            cross = pd.crosstab(merged["estrellas"], merged["sentimiento"])
            fig_cross = px.bar(
                cross.reset_index().melt(id_vars="estrellas"),
                x="estrellas", y="value", color="sentimiento",
                barmode="group",
                labels={"estrellas":"Estrellas","value":"Cantidad","sentimiento":"Sentimiento"},
                color_discrete_map={
                    "Positivo": "#2ecc71",
                    "Negativo": "#e74c3c",
                    "Neutro": "#f39c12"
                },
            )
            st.plotly_chart(fig_cross, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# TAB 4 — Nuevo comentario
# ═══════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("💬 Analiza un Comentario Nuevo")
    st.markdown("Escribe o pega cualquier opinión sobre el libro y la IA clasificará su sentimiento.")

    nuevo = st.text_area(
        "✍️ Escribe tu opinión aquí:",
        height=150,
        placeholder="Ej: Me pareció un libro increíble, lleno de consejos prácticos...",
    )

    if st.button("🔍 Analizar sentimiento", type="primary") and nuevo.strip():
        with st.spinner("Analizando... ⏳"):
            prompt = f"""Analiza el sentimiento del siguiente comentario sobre el libro 'Hábitos Atómicos'.
Responde SOLO con JSON sin markdown:
{{"sentimiento": "Positivo|Negativo|Neutro", "confianza": "Alta|Media|Baja", "resumen": "breve explicación en español", "aspectos": ["aspecto1","aspecto2"]}}

Comentario: {nuevo}"""
            raw = call_claude(prompt)

        try:
            clean = re.sub(r"```json|```", "", raw).strip()
            result = json.loads(clean)

            sent = result.get("sentimiento", "Neutro")
            conf = result.get("confianza", "Media")
            resumen = result.get("resumen", "")
            aspectos = result.get("aspectos", [])

            color_map2 = {"Positivo": "green", "Negativo": "red", "Neutro": "orange"}
            emoji_map = {"Positivo": "😊", "Negativo": "😞", "Neutro": "😐"}
            color = color_map2.get(sent, "gray")
            emoji = emoji_map.get(sent, "🤔")

            st.markdown(f"## Resultado: :{color}[{emoji} **{sent}**]")
            col1, col2, col3 = st.columns(3)
            col1.metric("Sentimiento", f"{emoji} {sent}")
            col2.metric("Confianza", conf)
            col3.metric("Aspectos detectados", len(aspectos))

            st.markdown(f"**📝 Análisis:** {resumen}")

            if aspectos:
                st.markdown("**🏷️ Aspectos identificados:**")
                cols = st.columns(min(len(aspectos), 4))
                for i, asp in enumerate(aspectos):
                    cols[i % len(cols)].info(asp)

        except Exception as e:
            st.error(f"Error procesando respuesta: {e}")
            st.code(raw)

    elif not nuevo.strip() and st.button("🔍 Analizar sentimiento"):
        st.warning("Por favor escribe un comentario antes de analizar.")

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("📚 Análisis de opiniones de *Hábitos Atómicos* · Powered by Claude AI · Taller NLP")