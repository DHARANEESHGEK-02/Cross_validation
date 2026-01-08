import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

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

# Load digits dataset (10 classes = multiclass)
digits = load_digits()
X, y = digits.data, digits.target
df_data = pd.DataFrame(X, columns=[f"pixel_{i}" for i in range(X.shape[1])])
df_data['target'] = y

# Dataset metrics
col1, col2, col3 = st.columns(3)
col1.metric("Samples", X.shape[0])
col2.metric("Features", X.shape[1])
col3.metric("Classes", len(np.unique(y)))

st.subheader("📊 Dataset Preview")
st.dataframe(df_data.head(), width="stretch")

# Model selection (FIXED for sklearn 1.8 + multiclass)
st.subheader("🤖 Select Models")
models = {
    "Logistic (OvR)": OneVsRestClassifier(LogisticRegression(solver='liblinear', max_iter=1000)),
    "Logistic (lbfgs)": LogisticRegression(solver='lbfgs', max_iter=1000),
    "SVM": SVC(gamma='auto'),
    "Random Forest": RandomForestClassifier(n_estimators=40, random_state=42)
}

selected_models = st.multiselect("Choose models", list(models.keys()), default=["Logistic (lbfgs)", "SVM", "Random Forest"])

# Run CV
if st.button("🚀 Run Cross-Validation", type="primary"):
    results = {}
    kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    with st.spinner(f"Running {cv_folds}-fold CV..."):
        for name, model in models.items():
            if name in selected_models:
                scores = cross_val_score(model, X, y, cv=kf, n_jobs=-1)
                results[name] = {
                    'scores': scores,
                    'mean': np.mean(scores),
                    'std': np.std(scores)
                }
    
    # Results table
    st.subheader("📈 Results")
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Mean ± Std': [f"{results[name]['mean']:.3f}±{results[name]['std']:.3f}" for name in results],
        'Best': [f"{np.max(results[name]['scores']):.3f}" for name in results],
        'Worst': [f"{np.min(results[name]['scores']):.3f}" for name in results]
    })
    st.dataframe(results_df, width="stretch")
    st.session_state.results_df = results_df
    
    # Best model
    best_name = max(results, key=lambda k: results[k]['mean'])
    st.balloons()
    st.success(f"🏆 **Best**: {best_name} ({results[best_name]['mean']:.3f})")
    
    # Charts
    fig_bar = px.bar(x=list(results.keys()), y=[results[n]['mean'] for n in results],
                    error_y=[results[n]['std'] for n in results],
                    title="Mean CV Accuracy ± Std", template="plotly_dark")
    fig_bar.update_layout(paper_bgcolor="#1a1d24", plot_bgcolor="#1a1d24")
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Fold plot
    fig, ax = plt.subplots(figsize=(12,6), facecolor='#1a1d24')
    for name, res in results.items():
        ax.plot(range(1,cv_folds+1), res['scores'], 'o-', label=f"{name}", linewidth=2)
    ax.set_facecolor('#0e1117')
    ax.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    ax.grid(alpha=0.3, color='gray')
    ax.set_title("CV Scores per Fold", color='white', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)

# Tuning section
st.subheader("🔧 RF Parameter Tuning")
n_est = st.slider("n_estimators", 5, 100, 40, 5)
if st.button("Tune RF", key="tune"):
    rf = RandomForestClassifier(n_estimators=n_est, random_state=42)
    scores = cross_val_score(rf, X, y, cv=10, n_jobs=-1)
    col1, col2 = st.columns(2)
    col1.metric("Mean CV", f"{np.mean(scores):.3f}")
    col2.metric("±Std", f"{np.std(scores):.3f}")

# Download
if 'results_df' in st.session_state:
    st.download_button("💾 Download CSV", st.session_state.results_df.to_csv().encode(),
                      "cv_results.csv", "text/csv")
