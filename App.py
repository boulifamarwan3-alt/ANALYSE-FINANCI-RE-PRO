# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from datetime import timedelta

warnings.filterwarnings('ignore')

# --- Configuration de la Page ---
st.set_page_config(
    page_title="Système Expert d'Analyse Financière",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# --- Design System Light (Professionnel) ---
THEME = {
    'bg_main': '#f8f9fa',       # Gris très clair
    'card_bg': '#ffffff',       # Blanc
    'primary': '#1e3a8a',       # Bleu royal foncé (Titres)
    'secondary': '#334155',     # Gris ardoise
    'accent': '#0284c7',        # Bleu ciel vif
    'success': '#16a34a',       # Vert vibrant
    'warning': '#d97706',       # Orange
    'danger': '#dc2626',        # Rouge
    'text_main': '#0f172a',     # Texte sombre
    'text_sub': '#64748b',      # Texte gris
    'border': '#e2e8f0'         # Bordures
}

# Palette de couleurs vibrantes pour les graphiques
COLORS = [
    '#2563eb', # Bleu vibrant
    '#db2777', # Rose magenta
    '#16a34a', # Vert
    '#9333ea', # Violet
    '#ea580c', # Orange
    '#0891b2', # Cyan
    '#ca8a04'  # Or
]

st.markdown(f"""
<style>
    /* Global Reset & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: {THEME['text_main']};
    }}
    
    /* Background */
    .stApp {{
        background-color: {THEME['bg_main']};
    }}
    
    /* Headings - CLAIRES ET VISIBLES */
    h1, h2, h3 {{
        color: {THEME['primary']} !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
        text-transform: uppercase;
    }}
    
    h1 {{ font-size: 2.5rem !important; }}
    h2 {{ font-size: 1.8rem !important; border-bottom: 2px solid {THEME['border']}; padding-bottom: 10px; }}
    h3 {{ font-size: 1.4rem !important; margin-top: 20px !important; }}
    
    /* Cards */
    .premium-card {{
        background: {THEME['card_bg']};
        border: 1px solid {THEME['border']};
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }}
    
    /* Metrics */
    div[data-testid="stMetric"] {{
        background: {THEME['card_bg']};
        border-radius: 8px;
        padding: 16px;
        border: 1px solid {THEME['border']};
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    div[data-testid="stMetricLabel"] {{
        color: {THEME['text_sub']} !important;
        font-size: 0.9rem;
        font-weight: 600;
    }}
    div[data-testid="stMetricValue"] {{
        color: {THEME['primary']} !important;
        font-weight: 700;
        font-size: 1.5rem !important;
    }}
    
    /* Buttons */
    .stButton>button {{
        background-color: {THEME['primary']};
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        transition: all 0.2s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .stButton>button:hover {{
        background-color: {THEME['accent']};
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }}
    
    /* Explanation Box Professional */
    .explanation-box {{
        background-color: #f8fafc;
        border-left: 4px solid {THEME['accent']};
        padding: 20px;
        border-radius: 0 8px 8px 0;
        margin-top: 20px;
        color: {THEME['secondary']};
        font-size: 0.95rem;
        line-height: 1.6;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }}
    .explanation-title {{
        color: {THEME['primary']};
        font-weight: 700;
        margin-bottom: 8px;
        display: block;
        font-size: 1.05rem;
    }}
    
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {THEME['card_bg']};
        border-right: 1px solid {THEME['border']};
    }}
</style>
""", unsafe_allow_html=True)

# --- Fonctions Utilitaires ---

def safe_numeric(df, cols):
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def detect_columns(df):
    date_col = next((c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()), None)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) or 'price' in c.lower() or 'close' in c.lower()]
    return date_col, num_cols

# --- Fonctions de Traitement ---

def traiter_valeurs_manquantes(df, method):
    """Applique la méthode de traitement des valeurs manquantes choisie."""
    df_clean = df.copy()
    if method == "Remplacement par 0":
        df_clean = df_clean.fillna(0)
    elif method == "Propagation (Forward Fill)":
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
    elif method == "Interpolation Linéaire":
        df_clean = df_clean.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    return df_clean

# --- Fonctions de Visualisation ---

def plot_price_evolution(df, main_col, bench_cols):
    fig = go.Figure()
    
    # Main Asset
    fig.add_trace(go.Scatter(
        x=df.index, y=df[main_col],
        name=main_col,
        line=dict(color=COLORS[0], width=3)
    ))
    
    # Benchmarks
    for i, col in enumerate(bench_cols):
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col],
            name=col,
            line=dict(color=COLORS[(i+1) % len(COLORS)], width=2, dash='solid')
        ))
        
    fig.update_layout(
        title=dict(text="HISTORIQUE DES PRIX", font=dict(color=THEME['primary'], size=20)),
        template="plotly_white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified",
        height=500,
        xaxis=dict(showgrid=True, gridcolor=THEME['border']),
        yaxis=dict(showgrid=True, gridcolor=THEME['border']),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_normalized_returns(df, main_col, bench_cols):
    df_norm = df / df.iloc[0] * 100
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=df_norm.index, y=df_norm[main_col], name=main_col, line=dict(color=COLORS[0], width=3)))
    
    for i, col in enumerate(bench_cols):
        fig.add_trace(go.Scatter(x=df_norm.index, y=df_norm[col], name=col, line=dict(color=COLORS[(i+1) % len(COLORS)], width=2)))
        
    fig.update_layout(
        title=dict(text="RENDEMENTS CUMULÉS (BASE 100)", font=dict(color=THEME['primary'], size=20)),
        template="plotly_white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis_title="Indice de Performance",
        height=500,
        xaxis=dict(showgrid=True, gridcolor=THEME['border']),
        yaxis=dict(showgrid=True, gridcolor=THEME['border']),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_correlation_heatmap(df):
    # Utilisation des rendements logarithmiques
    returns = np.log(df / df.shift(1)).dropna()
    corr = returns.corr()
    
    fig = px.imshow(
        corr,
        text_auto='.2f',
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1
    )
    
    fig.update_layout(
        title=dict(text="MATRICE DE CORRÉLATION (LOG RETURNS)", font=dict(color=THEME['primary'], size=20)),
        template="plotly_white",
        height=500
    )
    return fig

def calculate_ratios_table(df, market_col):
    # Rendements Logarithmiques
    returns = np.log(df / df.shift(1)).dropna()
    
    # Paramètres Marché
    rf = 0.03 # Taux sans risque 3%
    market_ret = returns[market_col]
    market_var = market_ret.var()
    market_ann_ret = market_ret.mean() * 252
    
    metrics_list = []
    
    for col in df.columns:
        r = returns[col]
        
        # 1. Métriques de base
        ann_return = r.mean() * 252
        ann_vol = r.std() * np.sqrt(252)
        sharpe = (ann_return - rf) / ann_vol if ann_vol != 0 else 0
        
        # 2. Max Drawdown
        running_max = df[col].cummax()
        drawdown = (df[col] - running_max) / running_max
        max_dd = drawdown.min()
        
        # 3. Métriques Avancées (Beta, Treynor, Jensen)
        if market_var > 0:
            cov = r.cov(market_ret)
            beta = cov / market_var
            treynor = (ann_return - rf) / beta if beta != 0 else 0
            jensen = ann_return - (rf + beta * (market_ann_ret - rf))
        else:
            beta = np.nan
            treynor = np.nan
            jensen = np.nan
            
        # 4. Métriques de Risque (VaR, CVaR)
        var_95 = np.percentile(r, 5)
        cvar_95 = r[r <= var_95].mean()

        metrics_list.append({
            'Actif': col,
            'Rendement Annuel': ann_return,
            'Volatilité': ann_vol,
            'Ratio de Sharpe': sharpe,
            'Beta': beta,
            'Ratio de Treynor': treynor,
            'Alpha de Jensen': jensen,
            'Max Drawdown': max_dd,
            'VaR (95%)': var_95,
            'CVaR (95%)': cvar_95
        })
    
    metrics = pd.DataFrame(metrics_list).set_index('Actif')
    
    return metrics

# --- Main Application ---

def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/ios-filled/100/1e3a8a/bullish.png", width=60)
        st.title("ANALYSE FINANCIÈRE PRO")
        st.markdown("---")
        
        step = st.radio("NAVIGATION", [
            "1. Import & Configuration",
            "2. Analyse & Benchmarking"
        ])
        
        st.markdown("---")
        st.caption("Plateforme d'analyse quantitative avancée.")

    # State Management
    if 'df_processed' not in st.session_state:
        st.session_state.df_processed = None
    if 'main_col' not in st.session_state:
        st.session_state.main_col = None
    if 'bench_cols' not in st.session_state:
        st.session_state.bench_cols = []

    # --- STEP 1: IMPORT ---
    if step == "1. Import & Configuration":
        st.markdown("# 📥 IMPORTATION ET TRAITEMENT DES DONNÉES")
        
        with st.container():
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Déposez votre fichier CSV ici", type=['csv'])
            st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file:
            df_raw = pd.read_csv(uploaded_file)
            
            # Column Selection
            st.markdown("### 🛠 CONFIGURATION DES VARIABLES")
            col1, col2 = st.columns(2)
            
            with col1:
                date_col_guess, num_cols_guess = detect_columns(df_raw)
                date_col = st.selectbox("Colonne Date", df_raw.columns, index=df_raw.columns.get_loc(date_col_guess) if date_col_guess else 0)
                main_col = st.selectbox("Actif Principal (Cible)", num_cols_guess)
            
            with col2:
                # Multiselect is scrollable by default in Streamlit
                bench_cols = st.multiselect(
                    "Benchmarks (Sélectionnez plusieurs)", 
                    [c for c in num_cols_guess if c != main_col],
                    default=None,
                    help="Utilisez la liste déroulante pour choisir vos benchmarks. Vous pouvez faire défiler si la liste est longue."
                )

            # Missing Value Handling Strategy
            st.markdown("### 🧹 TRAITEMENT DES VALEURS MANQUANTES")
            
            method = st.radio(
                "Méthode de traitement :",
                ["Remplacement par 0", "Propagation (Forward Fill)", "Interpolation Linéaire"],
                horizontal=True
            )

            if st.button("🔄 APPLIQUER ET TRAITER", use_container_width=True):
                # Processing
                try:
                    df_proc = df_raw.copy()
                    df_proc[date_col] = pd.to_datetime(df_proc[date_col], errors='coerce')
                    df_proc = df_proc.set_index(date_col).sort_index()
                    
                    # Select only relevant columns
                    cols_to_keep = [main_col] + bench_cols
                    df_proc = df_proc[cols_to_keep]
                    
                    # Apply numeric conversion
                    df_proc = safe_numeric(df_proc, cols_to_keep)
                    
                    # Apply Missing Value Strategy
                    df_proc = traiter_valeurs_manquantes(df_proc, method)
                    
                    st.session_state.df_processed = df_proc
                    st.session_state.main_col = main_col
                    st.session_state.bench_cols = bench_cols
                    
                    st.success(f"✅ Données traitées avec succès ! {len(df_proc)} observations validées.")
                    # Utilisation de st.table pour un affichage fixe
                    st.table(df_proc.head())
                    
                except Exception as e:
                    st.error(f"Erreur critique lors du traitement : {str(e)}")

    # --- STEP 2: ANALYSIS ---
    elif step == "2. Analyse & Benchmarking":
        if st.session_state.df_processed is None:
            st.warning("⚠️ Veuillez d'abord importer et traiter les données à l'étape 1.")
        else:
            df = st.session_state.df_processed
            main_col = st.session_state.main_col
            bench_cols = st.session_state.bench_cols
            
            st.markdown(f"# 📊 ANALYSE COMPARATIVE : {main_col.upper()}")
            
            # 1. Price Evolution
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.plotly_chart(plot_price_evolution(df, main_col, bench_cols), use_container_width=True)
            st.markdown("""
            <div class="explanation-box">
                <span class="explanation-title">ANALYSE DE LA TENDANCE</span>
                Ce graphique illustre l'évolution brute des prix. Il permet d'identifier les cycles de marché et les tendances séculaires.
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 2. Normalized Returns
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.plotly_chart(plot_normalized_returns(df, main_col, bench_cols), use_container_width=True)
            st.markdown("""
            <div class="explanation-box">
                <span class="explanation-title">PERFORMANCE RELATIVE (ALPHA)</span>
                Comparaison des rendements cumulés en base 100. Permet de visualiser la surperformance (Alpha) ou sous-performance relative.
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 3. Correlation Heatmap
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)
            st.markdown("""
            <div class="explanation-box">
                <span class="explanation-title">CORRÉLATIONS CROISÉES</span>
                Matrice de corrélation des rendements. Une valeur proche de 1 indique une forte corrélation positive (les actifs bougent ensemble), tandis qu'une valeur proche de -1 indique une corrélation négative (diversification).
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 4. Ratios Table
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.markdown("### 📐 RATIOS FINANCIERS ET RISQUES")
            
            # Sélecteur de marché pour Beta
            market_opts = list(df.columns)
            default_idx = 1 if len(market_opts) > 1 else 0 # Essaie de prendre le 1er benchmark par défaut
            market_ref = st.selectbox("Indice de Référence (Marché) pour Beta/Alpha :", market_opts, index=default_idx)
            
            ratios_df = calculate_ratios_table(df, market_ref)
            
            # Configuration du formatage
            format_dict = {
                'Rendement Annuel': '{:.2%}',
                'Volatilité': '{:.2%}',
                'Ratio de Sharpe': '{:.2f}',
                'Beta': '{:.2f}',
                'Ratio de Treynor': '{:.2f}',
                'Alpha de Jensen': '{:.2%}',
                'Max Drawdown': '{:.2%}',
                'VaR (95%)': '{:.2%}',
                'CVaR (95%)': '{:.2%}'
            }
            
            # Utilisation de st.table avec le style appliqué
            st.table(
                ratios_df.style
                .format(format_dict)
                .background_gradient(cmap='Blues', subset=['Ratio de Sharpe', 'Alpha de Jensen'])
            )
            st.markdown("""
            <div class="explanation-box">
                <span class="explanation-title">ANALYSE FONDAMENTALE QUANTITATIVE</span>
                <ul>
                    <li><b>Ratio de Sharpe :</b> Performance ajustée du risque total.</li>
                    <li><b>Ratio de Treynor :</b> Performance ajustée du risque de marché (Beta).</li>
                    <li><b>Alpha de Jensen :</b> Surperformance par rapport au rendement théorique attendu (CAPM).</li>
                    <li><b>VaR (95%) :</b> Perte maximale attendue sur un jour avec 95% de confiance.</li>
                    <li><b>CVaR (95%) :</b> Perte moyenne attendue dans les 5% des pires cas (Expected Shortfall).</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()