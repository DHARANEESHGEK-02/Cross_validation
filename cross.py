import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold

# Dark theme CSS
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    .stMetric {background-color: #1f2937; border: 1px solid #374151;}
    h1, h2, h3 {color: #ffffff !important;}
    .stMarkdown {color: #e8e8e8;}
    .dataframe {background-color: #1a1d24;}
    .dataframe th {background-color: #2a2e3b; color: #ffffff;}
    .dataframe td {background-color: #1a1d24; color: #e8e8e8;}
    .stButton > button {background: linear-gradient(45deg, #1f77b4, #00d4aa); color: white;}
</style>
""", unsafe_allow_html=True)

st.title("🔍 Cross-Validation Dashboard")
st.markdown("Compare model performance using **cross_val_score** on Digits dataset")

# Sidebar controls
st.sidebar.header("⚙️ Cross-Validation Settings")
cv_folds = st.sidebar.slider("CV Folds", 3, 10, 5)
dataset_choice = st.sidebar.radio("Dataset", ["Digits (Default)", "Upload CSV"])

# Load data
if dataset_choice == "Digits (Default)":
    digits = load_digits()
    X, y = digits.data, digits.target
    df_data = pd.DataFrame(X)
    df_data['target'] = y
    st.success(f"✅ Digits dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        df_data = pd.read_csv(uploaded_file)
        feature_cols = [col for col in df_data.columns if col != 'target']
        X = df_data[feature_cols].select_dtypes(include=[np.number]).fillna(0).values
        y = df_data['target'].values
    else:
        X, y = load_digits().data, load_digits().target
        df_data = pd.DataFrame(X)
        df_data['target'] = y

# Dataset metrics
col1, col2, col3 = st.columns(3)
col1.metric("Samples", X.shape[0])
col2.metric("Features", X.shape[1])
col3.metric("Classes", len(np.unique(y)))

st.subheader("📊 Dataset Preview")
st.dataframe(df_data.head(), width="stretch")

# Model selection (FIXED for sklearn 1.8+)
st.subheader("🤖 Select Models to Compare")
models = {
    "Logistic Regression": LogisticRegression(solver='liblinear', max_iter=1000),
    "SVM": SVC(gamma='auto'),
    "Random Forest (n=40)": RandomForestClassifier(n_estimators=40, random_state=42)
}

selected_models = st.multiselect("Choose models", list(models.keys()), default=list(models.keys()))

# Run cross-validation
if st.button("🚀 Run Cross-Validation", type="primary"):
    results = {}
    
    kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    with st.spinner("Running CV..."):
        for name, model in models.items():
            if name in selected_models:
                scores = cross_val_score(model, X, y, cv=kf, n_jobs=-1)
                results[name] = {
                    'scores': scores,
                    'mean': np.mean(scores),
                    'std': np.std(scores)
                }
    
    # Results table
    st.subheader("📈 Cross-Validation Results")
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Mean CV Score': [f"{results[name]['mean']:.3f} ± {results[name]['std']:.3f}" for name in results],
        'Best Score': [f"{np.max(results[name]['scores']):.3f}" for name in results],
        'Worst Score': [f"{np.min(results[name]['scores']):.3f}" for name in results]
    })
    st.dataframe(results_df, width="stretch")
    
    # Best model
    best_model_name = max(results, key=lambda k: results[k]['mean'])
    st.success(f"🏆 **Best Model**: {best_model_name} ({results[best_model_name]['mean']:.3f})")
    
    # Bar chart
    st.subheader("📊 Mean CV Scores")
    fig_bar = px.bar(
        x=list(results.keys()), 
        y=[results[name]['mean'] for name in results],
        error_y=[results[name]['std'] for name in results],
        title="Cross-Validation Mean Scores ± Std",
        template="plotly_dark"
    )
    fig_bar.update_layout(paper_bgcolor="#1a1d24", plot_bgcolor="#1a1d24")
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Fold scores plot
    st.subheader("📈 Individual Fold Scores")
    fig_scores, ax = plt.subplots(figsize=(12, 6), facecolor='#1a1d24')
    for i, (name, res) in enumerate(results.items()):
        ax.plot(range(1, cv_folds+1), res['scores'], 'o-', 
                label=f"{name} (μ={res['mean']:.3f})", linewidth=2, markersize=8)
    ax.set_facecolor('#0e1117')
    ax.set_xlabel("Fold", fontsize=12, color='white')
    ax.set_ylabel("Accuracy", fontsize=12, color='white')
    ax.set_title("Cross-Validation Scores per Fold", fontsize=14, color='white')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, color='gray')
    plt.tight_layout()
    st.pyplot(fig_scores)

# Parameter tuning
st.subheader("🔧 Parameter Tuning (Random Forest)")
n_estimators = st.slider("n_estimators", 5, 100, 40, 5)

if st.button("Test n_estimators", key="tune"):
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    scores = cross_val_score(rf_model, X, y, cv=10, n_jobs=-1)
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean CV", f"{np.mean(scores):.3f}")
    col2.metric("± Std", f"{np.std(scores):.3f}")
    col3.metric("Best Fold", f"{np.max(scores):.3f}")

# Download
if st.session_state.get('results_df', None):
    csv = st.session_state.results_df.to_csv(index=False).encode('utf-8')
    st.download_button("💾 Download Results", csv, "cv_results.csv", "text/csv")
