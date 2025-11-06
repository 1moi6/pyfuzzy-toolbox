"""
pyfuzzy-toolbox - Interactive Visual Interface
A modern, academic interface for creating and testing fuzzy systems

Author: Moiseis Cecconello
Version: 1.0.0
"""

import streamlit as st
from pathlib import Path

# Initialize session state first
def init_session_state():
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    if 'fuzzy_system' not in st.session_state:
        st.session_state.fuzzy_system = None

init_session_state()

# Page configuration
st.set_page_config(
    page_title="pyfuzzy-toolbox",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern academic look
def load_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global styles */
    * {
        font-family: 'Inter', sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    # header {visibility: hidden;}

    /* Reduce top padding */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 90%;
    }

    .stMainBlockContainer {
        padding-top: 2rem;
        padding-bottom: 0;
        max-width: 90%;
    }

    /* Custom navigation bar */
    .nav-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Cards */
    .feature-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        border: 1px solid #e5e7eb;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        cursor: pointer;
    }

    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.15);
        border-color: #667eea;
    }

    .card-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }

    .card-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.75rem;
    }

    .card-description {
        color: #6b7280;
        line-height: 1.6;
        font-size: 0.95rem;
    }

    /* Hero section */
    .hero-container {
        padding: 1.5rem 0 2rem 0;
        text-align: center;
    }

    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        text-align: center;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        color: #6b7280;
        text-align: center;
        margin: 0 auto 2.5rem auto;
        max-width: 100%;
        line-height: 1.6;
        padding: 0 1rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    @media (max-width: 768px) {
        .hero-subtitle {
            white-space: normal;
        }
    }

    /* Text animations */
    .animate-title {
        animation: fadeInUp 0.8s ease-out;
    }

    .animate-subtitle {
        animation: fadeInUp 1s ease-out 0.2s both;
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Stats */
    .stat-box {
        background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
    }

    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }

    .stat-label {
        color: #6b7280;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.5rem;
    }

    /* Buttons */
    .stButton {
        margin-top: 1.5rem !important;
    }

    /* All buttons - purple gradient by default */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
    }

    /* Sidebar buttons - action buttons styling */
    [data-testid="stSidebar"] .stButton {
        margin-top: 0.5rem !important;
    }

    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        width: 100% !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3) !important;
        transform: translateY(-1px) !important;
    }

    /* Individual button colors based on position */
    /* Save button (first action button) - green */
    [data-testid="stSidebar"] .stButton:nth-of-type(1) > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    }

    [data-testid="stSidebar"] .stButton:nth-of-type(1) > button:hover {
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3) !important;
    }

    /* Load button (second action button) - blue */
    [data-testid="stSidebar"] .stButton:nth-of-type(2) > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    }

    [data-testid="stSidebar"] .stButton:nth-of-type(2) > button:hover {
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3) !important;
    }

    /* Reset button (third action button) - orange */
    [data-testid="stSidebar"] .stButton:nth-of-type(3) > button {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
    }

    [data-testid="stSidebar"] .stButton:nth-of-type(3) > button:hover {
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.3) !important;
    }

    /* Sidebar columns - remove padding for home button */
    [data-testid="stSidebar"] [data-testid="column"] {
        padding: 0 !important;
    }

    /* Fix Streamlit column spacing */
    [data-testid="column"] > div {
        display: flex;
        flex-direction: column;
    }

    /* Animation keyframes */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .animate-fade-in {
        animation: fadeIn 0.6s ease-out;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    .animate-pulse {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }

    /* Section headers */
    .section-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
        margin-top: 3rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }

    .divider {
        width: 80px;
        height: 4px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin: 1rem auto 2rem;
        border-radius: 2px;
    }
    </style>
    """, unsafe_allow_html=True)

# Navigation function
def navigate_to(page_name):
    st.session_state.page = page_name
    st.rerun()

# Load custom CSS
load_css()

# Import modules
from modules import home, inference, learning, fuzzy_ode_module, dynamics_pfuzzy_discrete,dynamics_pfuzzy_continuous

# Define pages using st.Page with unique url_path
home_page = st.Page(home.run, title="Home", icon="🏠", url_path="home", default=True)

# Inference module pages
def inference_mamdani():
    st.session_state.inference_system_type = "Mamdani"
    inference.run()

def inference_sugeno():
    st.session_state.inference_system_type = "Sugeno"
    inference.run()

mamdani_page = st.Page(inference_mamdani, title="Mamdani", icon="🎯", url_path="mamdani")
st.session_state['app_pages'] = [mamdani_page]
sugeno_page = st.Page(inference_sugeno, title="Sugeno", icon="🎲", url_path="sugeno")

# Learning module pages
def learning_anfis():
    st.session_state.learning_method = "ANFIS"
    learning.run()

def learning_wang_mendel():
    st.session_state.learning_method = "Wang-Mendel"
    learning.run()

def learning_optimization():
    st.session_state.learning_method = "Rule Optimization"
    learning.run()

anfis_page = st.Page(learning_anfis, title="ANFIS", icon="🧠", url_path="anfis")
wang_mendel_page = st.Page(learning_wang_mendel, title="Wang-Mendel", icon="📚", url_path="wang-mendel")
optimization_page = st.Page(learning_optimization, title="Rule Optimization", icon="🔧", url_path="optimization")

# Dynamics module pages
def dynamics_discrete():
    st.session_state.dynamics_system_type = "p-Fuzzy Discrete"
    dynamics_pfuzzy_discrete.run()

def dynamics_continuous():
    st.session_state.dynamics_system_type = "p-Fuzzy Continuous"
    dynamics_pfuzzy_continuous.run()

def dynamics_fuzzy_ode():
    st.session_state.dynamics_system_type = "Fuzzy ODE"
    fuzzy_ode_module.run()

discrete_page = st.Page(dynamics_discrete, title="p-Fuzzy Discrete", icon="📊", url_path="pfuzzy-discrete")
continuous_page = st.Page(dynamics_continuous, title="p-Fuzzy Continuous", icon="📈", url_path="pfuzzy-continuous")
fuzzy_ode_page = st.Page(dynamics_fuzzy_ode, title="Fuzzy ODE", icon="🧮", url_path="fuzzy-ode")

# Organize pages with sections
pages = {
    "Home": [home_page],
    "Inference": [mamdani_page, sugeno_page],
    "Learning": [anfis_page, wang_mendel_page, optimization_page],
    "Dynamics": [discrete_page, continuous_page, fuzzy_ode_page]
}

# Navigation with sidebar control
pg = st.navigation(pages, position="top")
pg.run()
