"""
Overview Tab for Wang-Mendel Module
Provides theory, explanations, and examples of Wang-Mendel algorithm
"""

import streamlit as st


def render():
    """Render overview tab with Wang-Mendel theory"""

    st.markdown("# Wang-Mendel Algorithm")
    st.caption("Automatic Fuzzy Rule Generation from Data")

    st.markdown("---")

    # Introduction
    st.markdown("""
    The **Wang-Mendel algorithm** is a classic method for automatically generating fuzzy rule bases
    from numerical data. Proposed by L.X. Wang and J.M. Mendel in 1992, it provides a systematic
    approach to extract human-interpretable fuzzy rules directly from input-output examples.
    """)

    st.markdown("")

    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Theory", "Algorithm", "Advantages", "References"])

    with tab1:
        render_theory()

    with tab2:
        render_algorithm()

    with tab3:
        render_advantages()

    with tab4:
        render_references()


def render_theory():
    """Render theory section"""

    st.markdown("### Theoretical Foundation")

    st.markdown("""
    #### What is Wang-Mendel?

    The Wang-Mendel algorithm is a **one-pass learning method** that generates a Mamdani-type
    fuzzy inference system from numerical data. Unlike iterative learning methods (e.g., ANFIS),
    Wang-Mendel creates rules in a single pass through the training data.

    #### Key Concepts

    **1. Domain Partitioning**

    The input and output spaces are divided into fuzzy regions using membership functions:
    """)

    st.latex(r"""
    \text{Universe } U \rightarrow \{\text{Low}, \text{Medium}, \text{High}\}
    """)

    st.markdown("""
    **2. Rule Generation from Data**

    For each training sample $(x_1, x_2, ..., x_n, y)$:
    - Find the fuzzy term with **highest membership** for each variable
    - Create a candidate rule: IF $x_1$ is $A_1$ AND $x_2$ is $A_2$ ... THEN $y$ is $B$

    **3. Rule Degree**

    Each rule has a **degree** representing its strength:
    """)

    st.latex(r"""
    D(\text{rule}) = \prod_{i=1}^{n} \mu_{A_i}(x_i) \times \mu_B(y)
    """)

    st.markdown("""
    Where $\mu_{A_i}(x_i)$ is the membership degree of $x_i$ in fuzzy set $A_i$.

    **4. Conflict Resolution**

    When multiple rules have the same antecedent (IF part) but different consequents (THEN part):
    - Keep the rule with **highest degree**
    - Discard the others

    This ensures a consistent and non-contradictory rule base.

    **5. Final Inference System**

    The resulting Mamdani system uses:
    - **Min** for T-norm (AND operation)
    - **Max** for T-conorm (OR operation)
    - **Center of Gravity (COG)** for defuzzification
    """)


def render_algorithm():
    """Render algorithm steps"""

    st.markdown("### Algorithm Steps")

    st.markdown("""
    The Wang-Mendel algorithm consists of **5 main steps**:
    """)

    # Step 1
    with st.expander("**Step 1: Domain Partitioning**", expanded=False):
        st.markdown("""
        Divide each input and output variable into fuzzy regions.

        **Example:**
        - Temperature: {Cold, Warm, Hot}
        - Humidity: {Dry, Normal, Humid}
        - Fan Speed: {Slow, Medium, Fast}

        **Common approaches:**
        - Triangular membership functions
        - Uniformly distributed partitions
        - 3-11 partitions per variable (typically 5-7)
        """)

        st.code("""
# Example: 5 triangular partitions
system = MamdaniSystem()
system.add_input('temperature', (0, 40))
system.add_auto_mfs('temperature', n_mfs=5, mf_type='triangular')
        """, language='python')

    # Step 2
    with st.expander("**Step 2: Generate Candidate Rules**", expanded=False):
        st.markdown("""
        For each training sample, generate a candidate rule.

        **Process:**

        1. Given sample: $(x_1=25, x_2=60, y=75)$
        2. Find highest membership for each variable:
           - Temperature (25°C) → "Warm" (μ = 0.8)
           - Humidity (60%) → "Normal" (μ = 0.7)
           - Fan Speed (75%) → "Fast" (μ = 0.9)
        3. Create rule:
           ```
           IF Temperature is Warm AND Humidity is Normal
           THEN Fan Speed is Fast
           ```
        4. Calculate rule degree: $D = 0.8 \\times 0.7 \\times 0.9 = 0.504$

        **Result:** $N$ candidate rules (one per training sample)
        """)

    # Step 3
    with st.expander("**Step 3: Assign Rule Degrees**", expanded=False):
        st.markdown("""
        Calculate the degree for each candidate rule.

        **Formula:**
        """)

        st.latex(r"""
        D(\text{rule}_k) = \mu_{A_1^k}(x_1^k) \times \mu_{A_2^k}(x_2^k) \times ... \times \mu_{B^k}(y^k)
        """)

        st.markdown("""
        Where:
        - $k$ = training sample index
        - $\mu_{A_i^k}$ = membership of input $x_i$ in its assigned fuzzy set
        - $\mu_{B^k}$ = membership of output $y$ in its assigned fuzzy set

        **Interpretation:** Higher degree = rule is more representative of the data
        """)

    # Step 4
    with st.expander("**Step 4: Resolve Conflicts**", expanded=False):
        st.markdown("""
        Handle rules with identical antecedents but different consequents.

        **Conflict Example:**

        ```
        Rule 1: IF x is Low  THEN y is Small  (degree = 0.6)
        Rule 2: IF x is Low  THEN y is Medium (degree = 0.8)  ← Keep
        Rule 3: IF x is Low  THEN y is Large  (degree = 0.4)
        ```

        **Resolution:**
        - Compare degrees of conflicting rules
        - Keep rule with **maximum degree** (Rule 2)
        - Discard others

        **Why?** The rule with highest degree best represents the data for that input region.

        **Result:** Consistent, non-contradictory rule base
        """)

    # Step 5
    with st.expander("**Step 5: Create Final System**", expanded=False):
        st.markdown("""
        Combine all non-conflicting rules into the final Mamdani system.

        **Final Rule Base:**
        - Contains unique rules (no conflicts)
        - Typically much smaller than $N$ (number of training samples)
        - Reduction ratio: 2x - 10x compression

        **Example Output:**

        ```
        Final Rule Base (12 rules):
        ─────────────────────────────────────────────────
        1. IF temp is Cold  AND hum is Dry    THEN speed is Slow
        2. IF temp is Cold  AND hum is Normal THEN speed is Slow
        3. IF temp is Warm  AND hum is Dry    THEN speed is Medium
        4. IF temp is Warm  AND hum is Normal THEN speed is Medium
        5. IF temp is Warm  AND hum is Humid  THEN speed is Fast
        ...
        ```

        The system is now ready for inference!
        """)


def render_advantages():
    """Render advantages and limitations"""

    st.markdown("### Advantages & Limitations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Advantages")

        st.markdown("""
        **Speed**
        - Single pass through data
        - No iterative optimization
        - Fast training (seconds for 1000s of samples)

        **Interpretability**
        - Generates linguistic rules
        - Human-readable IF-THEN format
        - Easy to understand and validate

        **Simplicity**
        - Few hyperparameters
        - Automatic rule extraction
        - No gradient computation needed

        **Robustness**
        - Handles noise through conflict resolution
        - Works with incomplete data
        - Stable results

        **Versatility**
        - Regression and classification
        - Multi-input/multi-output
        - Any number of variables
        """)

    with col2:
        st.markdown("#### Limitations")

        st.markdown("""
        **Fixed Partitioning**
        - Membership functions not optimized
        - Manual selection of partitions
        - May not fit data perfectly

        **Rule Explosion**
        - High-dimensional data → many rules
        - $n$ inputs × $p$ partitions = $p^n$ possible rules
        - Example: 5 inputs × 5 partitions = 3,125 rules

        **Local Learning**
        - Each sample creates one rule
        - No global optimization
        - May miss underlying patterns

        **Consequent Limitations**
        - Only fuzzy consequents (Mamdani)
        - No functional outputs (unlike Sugeno)
        - Less flexible for approximation

        **Performance**
        - Lower accuracy than ANFIS/neural methods
        - Trade-off: speed vs precision
        - Best for interpretability-critical applications
        """)

    st.markdown("---")

    st.markdown("### When to Use Wang-Mendel?")

    st.markdown("""
    **Ideal for:**
    - **Explainable AI** requirements
    - Quick **baseline models**
    - Systems requiring **human validation**
    - **Low-dimensional** problems (2-5 inputs)
    - **Real-time** applications

    **Consider alternatives for:**
    - Maximum accuracy needed → Use **ANFIS** or **Neural Networks**
    - High-dimensional data (>10 inputs) → Use **feature selection** first
    - Complex nonlinear patterns → Use **ANFIS** with metaheuristics
    - Large datasets → Use **batch learning** or **deep learning**
    """)


def render_references():
    """Render references and resources"""

    st.markdown("### References & Resources")

    st.markdown("#### Original Paper")

    st.markdown("""
    **Wang, L. X., & Mendel, J. M. (1992)**
    *"Generating fuzzy rules by learning from examples"*
    IEEE Transactions on Systems, Man, and Cybernetics, 22(6), 1414-1427.

    [View on IEEE Xplore](https://ieeexplore.ieee.org/document/199466)
    """)

    st.markdown("---")

    st.markdown("#### Related Methods")

    with st.expander("**Comparison with Other Fuzzy Learning Methods**"):
        st.markdown("""
        | Method | Type | Speed | Accuracy | Interpretability |
        |--------|------|-------|----------|------------------|
        | **Wang-Mendel** | One-shot rule extraction | ⚡⚡⚡ Fast | ⭐⭐⭐ Good | ⭐⭐⭐ High |
        | **ANFIS** | Hybrid neuro-fuzzy | ⚡⚡ Medium | ⭐⭐⭐⭐ Excellent | ⭐⭐ Medium |
        | **Genetic Fuzzy** | Evolutionary optimization | ⚡ Slow | ⭐⭐⭐⭐ Excellent | ⭐⭐⭐ High |
        | **Clustering-based** | Data-driven partitioning | ⚡⚡ Medium | ⭐⭐⭐ Good | ⭐⭐ Medium |

        **Wang-Mendel vs ANFIS:**
        - Wang-Mendel: Fast, interpretable, good for baselines
        - ANFIS: Slower, more accurate, optimizes parameters

        **Wang-Mendel vs Genetic Fuzzy:**
        - Wang-Mendel: Deterministic, fast, no hyperparameters
        - Genetic: Stochastic, slow, optimizes structure + parameters
        """)

    st.markdown("---")

    st.markdown("#### Implementation Example")

    with st.expander("**Complete Python Example**"):
        st.code("""
import numpy as np
import fuzzy_systems as fs
from fuzzy_systems.learning import WangMendelLearning
from fuzzy_systems.inference import MamdaniSystem

# Generate sample data
X_train = np.random.uniform(0, 10, (100, 2))
y_train = (np.sin(X_train[:, 0]) + np.cos(X_train[:, 1])).reshape(-1, 1)

# Create Mamdani system
system = MamdaniSystem()

# Add inputs with automatic MFs
system.add_input('x1', (0, 10))
system.add_input('x2', (0, 10))
system.add_auto_mfs('x1', n_mfs=5, mf_type='triangular')
system.add_auto_mfs('x2', n_mfs=5, mf_type='triangular')

# Add output with automatic MFs
y_min, y_max = y_train.min(), y_train.max()
system.add_output('y', (y_min, y_max))
system.add_auto_mfs('y', n_mfs=5, mf_type='triangular')

# Run Wang-Mendel algorithm
wm = WangMendelLearning(system, X_train, y_train, task='regression')
trained_system = wm.fit(verbose=True)

# Get statistics
stats = wm.get_training_stats()
print(f"Candidate rules: {stats['candidate_rules']}")
print(f"Final rules: {stats['final_rules']}")
print(f"Conflicts resolved: {stats['conflicts_resolved']}")

# Make predictions
X_test = np.random.uniform(0, 10, (20, 2))
y_pred = wm.predict(X_test)

# Evaluate
from sklearn.metrics import r2_score
print(f"R² score: {r2_score(y_test, y_pred):.4f}")

# Visualize rules
trained_system.plot_rule_matrix()
        """, language='python')

    st.markdown("---")

    st.markdown("#### Additional Resources")

    st.markdown("""
    - **pyfuzzy-toolbox Documentation**: [https://1moi6.github.io/pyfuzzy-toolbox/](https://1moi6.github.io/pyfuzzy-toolbox/)
    - **Wang-Mendel Quick Start Guide**: [Online Guide](https://1moi6.github.io/pyfuzzy-toolbox/quick_start/wang_mendel/)
    - **Example Notebooks**: Available in the Examples Gallery
    - **API Reference**: [WangMendelLearning Class](https://1moi6.github.io/pyfuzzy-toolbox/api_reference/learning/)
    """)

    st.markdown("---")

    st.info("""
    **Tip:** Start with Wang-Mendel for quick rule extraction and interpretability.
    If you need higher accuracy, move to ANFIS or hybrid methods later!
    """)
