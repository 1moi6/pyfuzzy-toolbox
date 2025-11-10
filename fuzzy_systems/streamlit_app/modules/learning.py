"""
Learning & Optimization Module
Interface for ANFIS, Wang-Mendel, and metaheuristic optimization
"""

import streamlit as st

def run():
    """Render learning & optimization page"""

    # Get method from session_state (set by navigation)
    method = st.session_state.get('learning_method', 'ANFIS')

    # Map method names
    method_map = {
        'ANFIS': 'ANFIS (Adaptive Neuro-Fuzzy Inference System)',
        'Wang-Mendel': 'Wang-Mendel (Rule Extraction)',
        'Rule Optimization': 'Mamdani Learning (Metaheuristic Optimization)'
    }

    algorithm = method_map.get(method, method_map['ANFIS'])

    # Sidebar
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; padding: 0.25rem 0 0.125rem 0; margin-top: 0.5rem;">
            <h2 style="margin: 0.25rem 0 0.125rem 0; color: #667eea;">{method}</h2>
            <p style="color: #6b7280; font-size: 0.9rem; margin: 0;">
                Learn fuzzy systems from data
            </p>
        </div>
        <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 0.25rem 0 0.5rem 0;">
        """, unsafe_allow_html=True)

    # Main content
    st.markdown(f"""
    <div style="text-align: center; padding: 0.5rem 0;">
        <h3 style="color: #6b7280; font-weight: 500; margin: 0; font-size: 1.1rem;">
            {method}
        </h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='border-bottom: 1px solid #e5e7eb; margin: 0.5rem 0 1.5rem 0;'></div>", unsafe_allow_html=True)

    # Algorithm-specific interface
    if "ANFIS" in algorithm:
        render_anfis_interface()
    elif "Wang-Mendel" in algorithm:
        render_wangmendel_interface()
    elif "Mamdani Learning" in algorithm:
        render_mamdani_learning_interface()

def render_anfis_interface():
    """Render ANFIS training interface"""

    st.markdown("### ANFIS Training")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.info("🚧 **Coming Soon**: ANFIS training interface")

        st.markdown("""
        **ANFIS Features**:
        - Hybrid learning (gradient descent + LSE)
        - Metaheuristic optimization (PSO, DE, GA)
        - Real-time training visualization
        - Automatic structure selection
        - Cross-validation support
        """)

    with col2:
        st.markdown("**Parameters**")
        n_rules = st.slider("Number of rules", 2, 10, 3)
        epochs = st.slider("Training epochs", 10, 1000, 100)
        learning_rate = st.slider("Learning rate", 0.001, 0.1, 0.01, format="%.3f")

    with st.expander("📊 Upload Training Data"):
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        if uploaded_file:
            st.success("Data uploaded successfully!")

    with st.expander("💻 View Example Code"):
        st.code("""
from fuzzy_systems.learning import ANFIS
import numpy as np

# Generate sample data
X = np.random.rand(100, 2) * 10
y = np.sin(X[:, 0]) + np.cos(X[:, 1])

# Create and train ANFIS
anfis = ANFIS(n_inputs=2, n_rules=3, task='regression')
anfis.fit(X, y, epochs=100, learning_rate=0.01)

# Predict
y_pred = anfis.predict(X)

# Evaluate
from sklearn.metrics import r2_score
print(f"R² Score: {r2_score(y, y_pred):.3f}")
        """, language='python')

def render_wangmendel_interface():
    """Render Wang-Mendel interface"""

    st.markdown("### Wang-Mendel Rule Extraction")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.info("🚧 **Coming Soon**: Wang-Mendel learning interface")

        st.markdown("""
        **Wang-Mendel Features**:
        - Single-pass rule extraction
        - Automatic membership function generation
        - Fast training (no iterations)
        - Interpretable rules
        - Handles regression and classification
        """)

    with col2:
        st.markdown("**Parameters**")
        n_mfs = st.slider("MFs per variable", 3, 7, 5)
        task = st.selectbox("Task", ["Regression", "Classification"])

    with st.expander("💻 View Example Code"):
        st.code("""
from fuzzy_systems.learning import WangMendelLearning
import numpy as np

# Generate sample data
X = np.random.rand(100, 2) * 10
y = X[:, 0] + 2 * X[:, 1]

# Create and train
wm = WangMendelLearning(n_mfs=5, task='regression')
system = wm.fit(X, y)

# Get rules
rules = system.export_rules()
print(f"Generated {len(rules)} rules")

# Predict
y_pred = system.predict(X)
        """, language='python')

def render_mamdani_learning_interface():
    """Render Mamdani optimization interface"""

    st.markdown("### Mamdani System Optimization")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.info("🚧 **Coming Soon**: Metaheuristic optimization interface")

        st.markdown("""
        **Optimization Features**:
        - Multiple algorithms (SA, GA, PSO, DE)
        - Optimize rule consequents
        - Preserve interpretability
        - Real-time convergence plot
        - Hyperparameter tuning
        """)

    with col2:
        st.markdown("**Parameters**")
        algorithm = st.selectbox("Algorithm", ["PSO", "DE", "GA", "SA"])
        pop_size = st.slider("Population size", 10, 100, 30)
        iterations = st.slider("Iterations", 10, 500, 100)

    with st.expander("💻 View Example Code"):
        st.code("""
from fuzzy_systems.learning import MamdaniLearning
from fuzzy_systems import MamdaniSystem

# Create initial system
system = MamdaniSystem()
# ... configure system ...

# Optimize with PSO
optimizer = MamdaniLearning(system=system)
optimized_system = optimizer.optimize(
    X_train, y_train,
    algorithm='PSO',
    pop_size=30,
    iterations=100
)

# Evaluate
y_pred = optimized_system.predict(X_test)
        """, language='python')
