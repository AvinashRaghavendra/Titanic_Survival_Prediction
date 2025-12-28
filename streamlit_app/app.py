import sys
import os

# --------------------------------------------------
# Add project root to Python path
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# --------------------------------------------------
# Imports
# --------------------------------------------------
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from joblib import load
from src.data_preprocessing import preprocess

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Titanic Survival Analyzer",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Initialize theme in session state
# --------------------------------------------------
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# --------------------------------------------------
# Theme Configuration
# --------------------------------------------------
def get_theme_config(theme):
    if theme == 'dark':
        return {
            'bg_primary': '#0f172a',
            'bg_secondary': '#1e293b',
            'bg_tertiary': '#334155',
            'text_primary': '#f1f5f9',
            'text_secondary': '#cbd5e1',
            'text_muted': '#94a3b8',
            'border': '#334155',
            'accent_cyan': '#06b6d4',
            'accent_blue': '#3b82f6',
            'accent_purple': '#8b5cf6',
            'accent_pink': '#ec4899',
            'accent_green': '#34d399',
            'gradient_start': '#06b6d4',
            'gradient_mid': '#3b82f6',
            'gradient_end': '#8b5cf6',
            'kpi_bg': 'linear-gradient(135deg, #1e293b 0%, #334155 100%)',
            'kpi_border': '#475569',
            'insight_bg': 'linear-gradient(135deg, #1e40af 0%, #7c3aed 100%)',
            'insight_text': '#ffffff',
            'chart_colors': ['#3b82f6', '#8b5cf6', '#06b6d4', '#ec4899', '#34d399'],
            'chart_bg': '#0f172a',
            'chart_text': '#e2e8f0',
            'grid_color': '#1e293b',
            'male_color': '#06b6d4',
            'female_color': '#ec4899'
        }
    else:  # light theme
        return {
            'bg_primary': '#ffffff',
            'bg_secondary': '#f8fafc',
            'bg_tertiary': '#e2e8f0',
            'text_primary': '#0f172a',
            'text_secondary': '#1e293b',
            'text_muted': '#475569',
            'border': '#cbd5e1',
            'accent_cyan': '#0284c7',
            'accent_blue': '#2563eb',
            'accent_purple': '#7c3aed',
            'accent_pink': '#db2777',
            'accent_green': '#059669',
            'gradient_start': '#0284c7',
            'gradient_mid': '#2563eb',
            'gradient_end': '#7c3aed',
            'kpi_bg': 'linear-gradient(135deg, #dbeafe 0%, #e0e7ff 100%)',
            'kpi_border': '#93c5fd',
            'insight_bg': 'linear-gradient(135deg, #bfdbfe 0%, #c7d2fe 100%)',
            'insight_text': '#1e3a8a',
            'chart_colors': ['#2563eb', '#7c3aed', '#0284c7', '#db2777', '#059669'],
            'chart_bg': '#ffffff',
            'chart_text': '#0f172a',
            'grid_color': '#e5e7eb',
            'male_color': '#0284c7',
            'female_color': '#db2777'
        }

theme = get_theme_config(st.session_state.theme)

# --------------------------------------------------
# Dynamic CSS based on theme
# --------------------------------------------------
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * {{
        font-family: 'Inter', sans-serif;
    }}
    
    /* Main container background */
    .stApp {{
        background-color: {theme['bg_primary']};
    }}
    
    /* Main header */
    .main-header {{
        background: linear-gradient(135deg, {theme['gradient_start']} 0%, {theme['gradient_mid']} 50%, {theme['gradient_end']} 100%);
        padding: 3rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(59, 130, 246, 0.3);
        border: 1px solid rgba(59, 130, 246, 0.2);
    }}
    
    .main-header h1 {{
        color: {theme['insight_text']};
        margin: 0;
        font-size: 3.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.12);
    }}
    
    .main-header p {{
        color: {theme['insight_text']};
        margin-top: 0.5rem;
        font-size: 1.2rem;
    }}
    
    /* KPI cards */
    [data-testid="stMetricValue"] {{
        color: {theme['text_primary']} !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }}
    
    [data-testid="stMetricLabel"] {{
        color: {theme['text_secondary']} !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }}
    
    [data-testid="stMetricDelta"] {{
        color: {theme['accent_green']} !important;
        font-weight: 600 !important;
    }}
    
    div[data-testid="stMetric"] {{
        background: {theme['kpi_bg']};
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid {theme['kpi_border']};
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }}
    
    div[data-testid="stMetric"]:hover {{
        border-color: {theme['accent_blue']};
        box-shadow: 0 6px 24px rgba(37, 99, 235, 0.2);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }}
    
    /* Insight boxes */
    .insight-box {{
        background: {theme['insight_bg']};
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.15);
        border: 2px solid {theme['kpi_border']};
    }}
    
    .insight-box h3 {{
        margin: 0 0 0.5rem 0;
        font-size: 1.3rem;
        color: {theme['insight_text']};
        font-weight: 700;
    }}
    
    .insight-box p {{
        color: {theme['insight_text']};
        line-height: 1.6;
        font-weight: 500;
    }}
    
    .insight-box b {{
        color: {theme['insight_text']};
        font-weight: 800;
    }}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: transparent;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1.5rem;
        background-color: {theme['bg_secondary']};
        border: 2px solid {theme['border']};
        color: {theme['text_muted']};
        font-weight: 600;
    }}
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background: linear-gradient(135deg, {theme['accent_blue']} 0%, {theme['accent_purple']} 100%);
        border-color: {theme['accent_blue']};
        color: #ffffff;
    }}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background-color: {theme['bg_secondary']};
        border-right: 2px solid {theme['border']};
    }}
    
    [data-testid="stSidebar"] h3 {{
        color: {theme['text_primary']};
        font-weight: 700;
    }}
    
    /* Dataframe styling - Fix for light theme */
    div[data-testid="stDataFrame"] {{
        border-radius: 10px;
        overflow: hidden;
        border: 2px solid {theme['border']};
    }}
    
    /* Force dataframe text color */
    div[data-testid="stDataFrame"] * {{
        color: {theme['text_primary']} !important;
    }}
    
    /* Dataframe table cells and headers */
    div[data-testid="stDataFrame"] table {{
        background-color: {theme['chart_bg']};
        color: {theme['text_primary']} !important;
    }}
    div[data-testid="stDataFrame"] th, div[data-testid="stDataFrame"] td {{
        background-color: {theme['bg_secondary']};
        color: {theme['text_primary']} !important;
        border-color: {theme['border']};
    }}
    div[data-testid="stDataFrame"] thead th {{
        background-color: {theme['bg_tertiary']};
        color: {theme['text_primary']} !important;
        font-weight: 700;
    }}
    
    /* Headers and text */
    h1, h2, h3, h4 {{
        color: {theme['text_primary']} !important;
        font-weight: 700 !important;
    }}
    
    p {{
        color: {theme['text_secondary']};
    }}
    
    /* Divider */
    hr {{
        border-color: {theme['border']};
        border-width: 2px;
    }}
    
    /* Info/warning boxes */
    .stAlert {{
        background-color: {theme['bg_secondary']};
        border: 2px solid {theme['accent_blue']};
        color: {theme['text_primary']} !important;
        font-weight: 500;
    }}
    
    .stAlert * {{
        color: {theme['text_primary']} !important;
    }}
    
    /* Sidebar text and labels */
    [data-testid="stSidebar"] label {{
        color: {theme['text_primary']} !important;
        font-weight: 600 !important;
    }}
    
    [data-testid="stSidebar"] p {{
        color: {theme['text_secondary']} !important;
    }}
    
    /* Select boxes and inputs */
    [data-testid="stSidebar"] div[data-baseweb="select"] {{
        background-color: {theme['bg_primary']};
        border: 1px solid {theme['border']};
    }}
    
    /* Button styling */
    .stButton button {{
        background: linear-gradient(135deg, {theme['accent_blue']} 0%, {theme['accent_purple']} 100%);
        color: white !important;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }}
    
    .stButton button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.3);
    }}
    
    /* Radio button text */
    .stRadio label {{
        color: {theme['text_primary']} !important;
    }}
    
    /* Checkbox text */
    .stCheckbox label {{
        color: {theme['text_primary']} !important;
    }}
    
    /* Selectbox text */
    .stSelectbox label {{
        color: {theme['text_primary']} !important;
    }}
    </style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Load model and data
# --------------------------------------------------
@st.cache_resource
def load_model_and_data():
    model = load(os.path.join(PROJECT_ROOT, "models", "titanic_model.joblib"))
    feature_columns = load(os.path.join(PROJECT_ROOT, "models", "feature_columns.joblib"))
    data = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "train.csv"))
    data["Age"] = data["Age"].fillna(data["Age"].median())
    return model, feature_columns, data

model, feature_columns, data = load_model_and_data()

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown("""
    <div class="main-header">
        <h1>⚓ Titanic Survival Analyzer</h1>
        <p>AI-Powered Predictive Analytics | Historical Dataset Insights</p>
    </div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Sidebar with Theme Toggle at Top
# --------------------------------------------------
with st.sidebar:
    # Theme toggle as first element in sidebar
    theme_icon = "🌙" if st.session_state.theme == 'light' else "☀️"
    theme_label = "Switch to Dark Mode" if st.session_state.theme == 'light' else "Switch to Light Mode"
    
    if st.button(f"{theme_icon} {theme_label}", use_container_width=True, key="theme_toggle", type="primary"):
        st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("### 🎯 Filter Controls")
    st.markdown("---")
    
    cls = st.multiselect(
        "🎫 Passenger Class",
        sorted(data["Pclass"].unique()),
        default=sorted(data["Pclass"].unique()),
        help="1st, 2nd, or 3rd class passengers"
    )
    
    gender = st.multiselect(
        "👤 Gender",
        sorted(data["Sex"].unique()),
        default=sorted(data["Sex"].unique()),
    )
    
    age_min, age_max = st.slider(
        "🎂 Age Range",
        int(data["Age"].min()),
        int(data["Age"].max()),
        (int(data["Age"].min()), int(data["Age"].max()))
    )
    
    st.markdown("---")
    
    if st.button("🔄 Reset All Filters", use_container_width=True):
        st.rerun()
    
    st.markdown("---")
    st.markdown(f"""
        <div style='padding: 1rem; background: {theme['bg_tertiary']}; border-radius: 8px; margin-top: 2rem; border: 2px solid {theme['border']};'>
            <small style='color: {theme['text_secondary']}; font-weight: 500;'><b style='color: {theme['text_primary']};'>💡 Quick Tips</b><br>
            • Use filters to explore specific demographics<br>
            • Check the insights tab for key findings<br>
            • Higher class = better survival odds</small>
        </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# Apply filters
# --------------------------------------------------
filtered = data.copy()

if cls:
    filtered = filtered[filtered["Pclass"].isin(cls)]
if gender:
    filtered = filtered[filtered["Sex"].isin(gender)]

filtered = filtered[(filtered["Age"] >= age_min) & (filtered["Age"] <= age_max)]

# --------------------------------------------------
# Chart template based on theme
# --------------------------------------------------
def get_chart_template():
    return {
        'plot_bgcolor': theme['chart_bg'],
        'paper_bgcolor': theme['chart_bg'],
        'colorway': theme['chart_colors'],
        'font': {'color': theme['chart_text'], 'size': 13, 'family': 'Inter'},
        'hoverlabel': {
            'font': {'color': theme['chart_text']},
            'bgcolor': theme['chart_bg']
        },
        'xaxis': {
            'gridcolor': theme['grid_color'],
            'linecolor': theme['border'],
            'zerolinecolor': theme['border'],
            'color': theme['chart_text'],
            'tickfont': {'color': theme['chart_text'], 'size': 12}
        },
        'yaxis': {
            'gridcolor': theme['grid_color'],
            'linecolor': theme['border'],
            'zerolinecolor': theme['border'],
            'color': theme['chart_text'],
            'tickfont': {'color': theme['chart_text'], 'size': 12}
        },
        'title': {
            'font': {'color': theme['chart_text'], 'size': 16, 'family': 'Inter'}
        },
        'legend': {
            'font': {'color': theme['chart_text'], 'size': 12}
        }
    }

# --------------------------------------------------
# Results Section
# --------------------------------------------------
if filtered.empty:
    st.error("❌ No passengers match the selected filters. Please adjust your criteria.")
else:
    X_processed, _ = preprocess(
        filtered.drop(columns=["Survived"]),
        is_train=False,
        feature_columns=feature_columns
    )

    X_model = X_processed.drop(columns=["PassengerId"])
    filtered["Survival Probability (%)"] = (model.predict_proba(X_model)[:, 1] * 100).round(2)
    filtered["Prediction"] = filtered["Survival Probability (%)"].apply(
        lambda x: "✅ Survived" if x >= 50 else "❌ Perished"
    )

    # KPIs
    st.markdown("### 📊 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    avg_survival = filtered['Survival Probability (%)'].mean()
    high_survival = (filtered["Survival Probability (%)"] >= 70).sum()
    low_survival = (filtered["Survival Probability (%)"] < 30).sum()
    predicted_survivors = (filtered["Survival Probability (%)"] >= 50).sum()
    
    with col1:
        st.metric("👥 Total Passengers", f"{len(filtered):,}")
    
    with col2:
        st.metric("📈 Avg Survival Rate", f"{avg_survival:.1f}%", 
                 delta=f"{avg_survival - 38.4:.1f}% vs overall" if avg_survival != 38.4 else None)
    
    with col3:
        st.metric("⭐ High Confidence", high_survival, 
                 help="Passengers with ≥70% survival probability")
    
    with col4:
        st.metric("🎯 Predicted Survivors", predicted_survivors,
                 help="Passengers with ≥50% survival probability")

    st.markdown("---")

    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📈 Analytics", "💡 Insights", "📋 Data Table"])
    
    with tab1:
        # Row 1: Distribution Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Survival Probability Distribution")
            fig_hist = px.histogram(
                filtered, 
                x="Survival Probability (%)",
                nbins=20,
                color_discrete_sequence=[theme['chart_colors'][0]],
                title="Distribution of Survival Predictions"
            )
            fig_hist.update_layout(**get_chart_template(), height=350, showlegend=False)
            fig_hist.update_traces(marker_line_color=theme['border'], marker_line_width=0.5)
            fig_hist.update_traces(textfont=dict(color=theme['chart_text']))
            fig_hist.update_xaxes(color=theme['chart_text'], tickfont=dict(color=theme['chart_text']))
            fig_hist.update_yaxes(color=theme['chart_text'], tickfont=dict(color=theme['chart_text']))
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            st.markdown("#### ⚖️ Survival by Gender & Class")
            survival_by_group = filtered.groupby(['Sex', 'Pclass'])['Survival Probability (%)'].mean().reset_index()
            survival_by_group['Class'] = survival_by_group['Pclass'].map({1: '1st Class', 2: '2nd Class', 3: '3rd Class'})
            
            fig_bar = px.bar(
                survival_by_group,
                x='Class',
                y='Survival Probability (%)',
                color='Sex',
                barmode='group',
                color_discrete_map={'male': theme['male_color'], 'female': theme['female_color']},
                title="Average Survival Rate by Demographics"
            )
            fig_bar.update_layout(**get_chart_template(), height=350)
            fig_bar.update_traces(marker_line_color=theme['border'], marker_line_width=0.5)
            fig_bar.update_traces(textfont=dict(color=theme['chart_text']))
            fig_bar.update_xaxes(color=theme['chart_text'], tickfont=dict(color=theme['chart_text']))
            fig_bar.update_yaxes(color=theme['chart_text'], tickfont=dict(color=theme['chart_text']))
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Row 2: Detailed Charts
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("#### 🎂 Age vs Survival Probability")
            fig_scatter = px.scatter(
                filtered,
                x='Age',
                y='Survival Probability (%)',
                color='Sex',
                size='Pclass',
                color_discrete_map={'male': theme['male_color'], 'female': theme['female_color']},
                title="How Age and Gender Affect Survival",
                opacity=0.7
            )
            fig_scatter.update_layout(**get_chart_template(), height=350)
            fig_scatter.update_traces(marker=dict(line=dict(color=theme['border'], width=0.5)))
            fig_scatter.update_traces(textfont=dict(color=theme['chart_text']))
            fig_scatter.update_xaxes(color=theme['chart_text'], tickfont=dict(color=theme['chart_text']))
            fig_scatter.update_yaxes(color=theme['chart_text'], tickfont=dict(color=theme['chart_text']))
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col4:
            st.markdown("#### 🎫 Class Distribution")
            class_counts = filtered['Pclass'].value_counts().reset_index()
            class_counts.columns = ['Class', 'Count']
            class_counts['Class'] = class_counts['Class'].map({1: '1st Class', 2: '2nd Class', 3: '3rd Class'})
            
            fig_pie = px.pie(
                class_counts,
                values='Count',
                names='Class',
                color_discrete_sequence=theme['chart_colors'][:3],
                title="Passenger Distribution by Class"
            )
            fig_pie.update_layout(**get_chart_template(), height=350)
            fig_pie.update_traces(textfont=dict(color=theme['chart_text'], size=14, family='Inter'))
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab2:
        st.markdown("### 📈 Advanced Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Survival Rate by Age Group")
            filtered['Age Group'] = pd.cut(filtered['Age'], bins=[0, 12, 18, 35, 60, 100], 
                                          labels=['Child (0-12)', 'Teen (13-18)', 'Adult (19-35)', 
                                                 'Middle Age (36-60)', 'Senior (60+)'])
            age_survival = filtered.groupby('Age Group')['Survival Probability (%)'].mean().reset_index()
            
            fig_age = px.bar(
                age_survival,
                x='Age Group',
                y='Survival Probability (%)',
                color='Survival Probability (%)',
                color_continuous_scale='Blues',
                title="Survival Probability by Age Group"
            )
            fig_age.update_layout(**get_chart_template(), height=400)
            fig_age.update_traces(marker_line_color=theme['border'], marker_line_width=0.5)
            fig_age.update_traces(textfont=dict(color=theme['chart_text']))
            fig_age.update_xaxes(color=theme['chart_text'], tickfont=dict(color=theme['chart_text']))
            fig_age.update_yaxes(color=theme['chart_text'], tickfont=dict(color=theme['chart_text']))
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            st.markdown("#### 🎯 Prediction Confidence Levels")
            
            confidence_bins = pd.cut(filtered['Survival Probability (%)'], 
                                    bins=[0, 30, 50, 70, 100],
                                    labels=['Low (0-30%)', 'Moderate (30-50%)', 
                                           'High (50-70%)', 'Very High (70-100%)'])
            confidence_counts = confidence_bins.value_counts().reset_index()
            confidence_counts.columns = ['Confidence', 'Count']
            
            fig_confidence = px.bar(
                confidence_counts,
                x='Confidence',
                y='Count',
                color='Count',
                color_continuous_scale='Purples',
                title="Distribution of Prediction Confidence"
            )
            fig_confidence.update_layout(**get_chart_template(), height=400)
            fig_confidence.update_traces(marker_line_color=theme['border'], marker_line_width=0.5)
            fig_confidence.update_traces(textfont=dict(color=theme['chart_text']))
            fig_confidence.update_xaxes(color=theme['chart_text'], tickfont=dict(color=theme['chart_text']))
            fig_confidence.update_yaxes(color=theme['chart_text'], tickfont=dict(color=theme['chart_text']))
            st.plotly_chart(fig_confidence, use_container_width=True)
        
        # Correlation heatmap
        st.markdown("#### 🔥 Feature Correlation Analysis")
        corr_data = filtered[['Age', 'Pclass', 'Survival Probability (%)']].copy()
        corr_data['Gender_Numeric'] = filtered['Sex'].map({'male': 0, 'female': 1})
        corr_matrix = corr_data.corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title="Correlation Between Features and Survival"
        )
        fig_corr.update_layout(**get_chart_template(), height=400)
        fig_corr.update_traces(textfont=dict(color=theme['chart_text'], size=13))
        fig_corr.update_xaxes(color=theme['chart_text'], tickfont=dict(color=theme['chart_text']))
        fig_corr.update_yaxes(color=theme['chart_text'], tickfont=dict(color=theme['chart_text']))
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab3:
        st.markdown("### 💡 Key Insights & Findings")
        
        # Generate insights
        female_survival = filtered[filtered['Sex'] == 'female']['Survival Probability (%)'].mean()
        male_survival = filtered[filtered['Sex'] == 'male']['Survival Probability (%)'].mean()
        class1_survival = filtered[filtered['Pclass'] == 1]['Survival Probability (%)'].mean()
        class3_survival = filtered[filtered['Pclass'] == 3]['Survival Probability (%)'].mean()
        child_survival = filtered[filtered['Age'] <= 12]['Survival Probability (%)'].mean()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
                <div class="insight-box">
                    <h3>👩 Gender Impact</h3>
                    <p><b>Women</b> had a <b>{female_survival:.1f}%</b> average survival rate, compared to <b>{male_survival:.1f}%</b> for men.</p>
                    <p>📊 <b>{abs(female_survival - male_survival):.1f}%</b> difference — "Women and children first" protocol was clearly followed.</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class="insight-box">
                    <h3>🎫 Class Disparity</h3>
                    <p><b>1st Class</b> passengers: <b>{class1_survival:.1f}%</b> survival rate<br>
                    <b>3rd Class</b> passengers: <b>{class3_survival:.1f}%</b> survival rate</p>
                    <p>📊 Higher class meant <b>{(class1_survival - class3_survival):.1f}%</b> better survival odds.</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if not filtered[filtered['Age'] <= 12].empty:
                st.markdown(f"""
                    <div class="insight-box">
                        <h3>👶 Children's Survival</h3>
                        <p>Children (age ≤12) had a <b>{child_survival:.1f}%</b> average survival rate.</p>
                        <p>📊 Priority was given to younger passengers during evacuation.</p>
                    </div>
                """, unsafe_allow_html=True)
            
            top_survivor = filtered.nlargest(1, 'Survival Probability (%)')
            if not top_survivor.empty:
                st.markdown(f"""
                    <div class="insight-box">
                        <h3>⭐ Highest Probability</h3>
                        <p>Passenger ID <b>{top_survivor.iloc[0]['PassengerId']}</b>:<br>
                        {top_survivor.iloc[0]['Sex'].title()}, Age {top_survivor.iloc[0]['Age']:.0f}, Class {top_survivor.iloc[0]['Pclass']}</p>
                        <p>🎯 Survival Probability: <b>{top_survivor.iloc[0]['Survival Probability (%)']:.1f}%</b></p>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("#### 🔍 Model Performance Summary")
        st.info(f"""
            **Analysis Summary:**
            - Analyzed **{len(filtered)}** passengers matching your filters
            - Average survival probability: **{avg_survival:.1f}%**
            - **{predicted_survivors}** passengers predicted to survive (≥50% probability)
            - **{high_survival}** passengers with high confidence (≥70% probability)
        """)
    
    with tab4:
        st.markdown("### 📋 Detailed Passenger Data")
        
        col_a, col_b, col_c = st.columns([2, 2, 1])
        with col_a:
            sort_by = st.selectbox("Sort by", ["Survival Probability (%)", "Age", "Pclass", "PassengerId"], index=0)
        with col_b:
            sort_order = st.radio("Order", ["Descending", "Ascending"], horizontal=True)
        with col_c:
            show_all = st.checkbox("All columns", value=False)
        
        if show_all:
            display_cols = filtered.columns.tolist()
        else:
            display_cols = ["PassengerId", "Name", "Sex", "Age", "Pclass", 
                          "Survival Probability (%)", "Prediction"]
        
        sorted_data = filtered[display_cols].sort_values(
            sort_by, 
            ascending=(sort_order == "Ascending")
        )
        
        st.dataframe(
            sorted_data,
            use_container_width=True,
            height=550,
            column_config={
                "Survival Probability (%)": st.column_config.ProgressColumn(
                    "Survival %",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
                "Prediction": st.column_config.TextColumn("Prediction"),
                "Age": st.column_config.NumberColumn("Age", format="%.0f")
            }
        )
        
        # Download button
        csv = sorted_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Filtered Data (CSV)",
            csv,
            "titanic_filtered_predictions.csv",
            "text/csv",
            key='download-csv'
        )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown(f"""
    <p style="text-align: center; color: {theme['text_muted']}; font-size: 0.9rem; padding: 1rem 0;">
        Made with ❤️ using Streamlit | 
        <b style="color: {theme['text_primary']};">Titanic Dataset Analysis</b> | 
        Powered by Machine Learning
    </p>
""", unsafe_allow_html=True)