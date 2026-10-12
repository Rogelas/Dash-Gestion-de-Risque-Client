import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as i_go

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Dashboard Exécutif Interactif", layout="wide")

# --- TITRE ET SOUS-TITRE CENTRALISÉS ---
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="font-size: 2.5rem; font-weight: bold; color: #1E293B; margin-bottom: 5px;">
            TABLEAU DE BORD EXÉCUTIF
        </h1>
        <p style="font-size: 1.1rem; color: #64748B; font-style: italic;">
            Analyse de la valeur client, des facteurs d'attrition et suivi des revenus
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================
# GÉNÉRATION DES DONNÉES FICTIVES COHÉRENTES
# ==========================================
@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 400

    # Graphique 1 : K-Means
    anc_bleu = np.random.uniform(1, 35, int(n_samples * 0.4))
    frais_bleu = np.random.uniform(20, 80, int(n_samples * 0.4))
    anc_vert = np.random.uniform(35, 72, int(n_samples * 0.4))
    frais_vert = np.random.uniform(20, 80, int(n_samples * 0.4))
    anc_rouge = np.random.uniform(1, 72, int(n_samples * 0.2))
    frais_rouge = np.random.uniform(80, 120, int(n_samples * 0.2))

    df1 = pd.DataFrame({
        'Anciennete': np.concatenate([anc_bleu, anc_vert, anc_rouge]),
        'FraisMensuels': np.concatenate([frais_bleu, frais_vert, frais_rouge]),
        'Segment': (['Nouveaux / À Risque'] * len(anc_bleu) +
                    ['Professionnels / Data-vores'] * len(anc_vert) +
                    ['Fidèles Économes'] * len(anc_rouge))
    })

    # Graphique 2 : Random Forest
    df2 = pd.DataFrame({
        'Facteur': ['FraisMensuels', 'Anciennete', 'FraisTotaux', 'DataGo'][::-1],
        'Importance': [0.31, 0.28, 0.21, 0.195][::-1]
    })

    # Graphique 3 : Scoring
    score_churn = np.random.beta(1, 8, 300) * 100
    score_non_churn = np.random.beta(8, 1.5, 900) * 100
    df3 = pd.DataFrame({
        'Score': np.concatenate([score_churn, score_non_churn]),
        'Churn': ['1'] * len(score_churn) + ['0'] * len(score_non_churn)
    })

    # Graphique 4 : OLS
    mois = np.arange(1, 25)
    reel = 80000 + 10000 * np.sin(mois) + np.random.normal(0, 5000, 24)
    modele = reel + np.random.normal(0, 2500, 24)
    df4 = pd.DataFrame({'Mois': mois, 'Réel': reel, 'Modélisé (OLS)': modele})

    return df1, df2, df3, df4


df1_raw, df2, df3, df4 = load_data()

# ==========================================
# ZONE DE FILTRES INTERACTIFS (TOP BAR)
# ==========================================
st.markdown("<h4 style='color: #475569; margin-bottom: 5px;'>🔍 Filtres globaux</h4>", unsafe_allow_html=True)
f_col1, f_col2 = st.columns(2)

with f_col1:
    segments_selectionnes = st.multiselect(
        "Sélectionner les segments de clients :",
        options=df1_raw['Segment'].unique(),
        default=df1_raw['Segment'].unique()
    )

with f_col2:
    budget_min, budget_max = st.slider(
        "Intervalle des Frais Mensuels (€) :",
        min_value=int(df1_raw['FraisMensuels'].min()),
        max_value=int(df1_raw['FraisMensuels'].max()),
        value=(int(df1_raw['FraisMensuels'].min()), int(df1_raw['FraisMensuels'].max()))
    )

# Application des filtres
df1_filtre = df1_raw[
    (df1_raw['Segment'].isin(segments_selectionnes)) &
    (df1_raw['FraisMensuels'] >= budget_min) &
    (df1_raw['FraisMensuels'] <= budget_max)
    ]

st.markdown("---")

# ==========================================
# INDICATEURS CLÉS DE PERFORMANCE (KPI CARDS)
# ==========================================
total_revenue_dernier_mois = df4['Réel'].iloc[-1]
taux_churn_estime = (df3['Churn'] == '1').mean() * 100
panier_moyen_filtre = df1_filtre['FraisMensuels'].mean() if not df1_filtre.empty else 0

kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.metric(
        label="💰 Revenu Mensuel Global (Dernier Mois)",
        value=f"{total_revenue_dernier_mois:,.0f} €".replace(",", " "),
        delta="-4.2% vs Mois Précédent"
    )

with kpi2:
    st.metric(
        label="📉 Taux d'Attrition Moyen (Churn)",
        value=f"{taux_churn_estime:.1f} %",
        delta="Algorithme de scoring stable",
        delta_color="off"
    )

with kpi3:
    st.metric(
        label="🛒 Panier Moyen Mensuel (Clients Filtrés)",
        value=f"{panier_moyen_filtre:.2f} €",
        delta=f"{len(df1_filtre)} clients actifs correspondants"
    )

st.markdown("---")

# ==========================================
# CRÉATION DES GRAPHIQUES PLOTLY
# ==========================================

# 1. Segmentation K-Means
fig1 = px.scatter(
    df1_filtre, x='Anciennete', y='FraisMensuels', color='Segment',
    color_discrete_map={
        'Fidèles Économes': '#DC2626',
        'Nouveaux / À Risque': '#2563EB',
        'Professionnels / Data-vores': '#16A34A'
    },
    title=f"1. Segmentation K-Means ({len(df1_filtre)} clients affichés)",
    labels={'Anciennete': 'Ancienneté (Mois)', 'FraisMensuels': 'Frais Mensuels (€)'}
)
# CORRECTION ICI : Remplacement de orient="h" par orientation="h"
fig1.update_layout(
    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
    margin=dict(t=50, b=50)
)

# 2. Facteurs Clés du Churn
fig2 = px.bar(
    df2, x='Importance', y='Facteur', orientation='h',
    title="2. Facteurs Clés du Churn (Random Forest)",
    labels={'Importance': 'Importance Relative', 'Facteur': ''}
)
fig2.update_traces(
    marker_color='#008080',
    text=[f"{val:.1%}" for val in df2['Importance']],
    textposition='inside',
    insidetextanchor='end'
)

# 3. Distribution de l'Algorithme de Scoring
fig3 = px.histogram(
    df3, x='Score', color='Churn', barmode='overlay',
    title="3. Distribution de l'Algorithme de Scoring",
    color_discrete_map={'0': '#2563EB', '1': '#F97316'},
    labels={'Score': 'Score de Fidélité (0 = Risque Max, 100 = Sûr)'}
)

# 4. Modèle Économétrique de Suivi des Revenus
fig4 = i_go.Figure()
fig4.add_trace(i_go.Scatter(x=df4['Mois'], y=df4['Réel'], name='Réel', mode='lines+markers', line=dict(color='black')))
fig4.add_trace(i_go.Scatter(x=df4['Mois'], y=df4['Modélisé (OLS)'], name='Modélisé (OLS)', mode='lines',
                            line=dict(color='red', dash='dash')))
fig4.update_layout(
    title=dict(text="4. Modèle Économétrique de Suivi des Revenus"),
    xaxis=dict(title="Chronologie (Mois)"),
    yaxis=dict(title="Revenu Global (€)")
)

# ==========================================
# DISPOSITION DE LA GRILLE (2x2)
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig4, use_container_width=True)
