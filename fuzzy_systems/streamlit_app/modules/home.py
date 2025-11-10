"""
Home page for pyfuzzy-toolbox interface
Modern academic design with smooth animations
"""

import streamlit as st

def run():
    """Render home page"""

    # Hero section with text animation
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title animate-title">pyfuzzy-toolbox</h1>
        <p class="hero-subtitle animate-subtitle">
            A comprehensive Python library for fuzzy systems with applications
            in inference, learning, and dynamic modeling
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Divider
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Main feature cards (moved before stats)
    st.markdown("""
    <div class="section-header">Explore Modules</div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown("""
        <div class="feature-card animate-fade-in">
            <span class="card-icon">⚙️</span>
            <div class="card-title">Inference Systems</div>
            <div class="card-description">
                Build Mamdani and Sugeno fuzzy inference systems. Create linguistic
                rules, design membership functions, and perform real-time inference
                for control and decision-making applications.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card animate-fade-in" style="animation-delay: 0.15s;">
            <span class="card-icon">🧠</span>
            <div class="card-title">Learning & Optimization</div>
            <div class="card-description">
                Learn fuzzy systems from data using ANFIS and Wang-Mendel algorithms.
                Optimize parameters with metaheuristics (PSO, DE, GA, SA) for
                regression, classification, and control tasks.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card animate-fade-in" style="animation-delay: 0.3s;">
            <span class="card-icon">📊</span>
            <div class="card-title">Dynamic Systems</div>
            <div class="card-description">
                Model temporal evolution with fuzzy uncertainty. Solve differential
                equations with fuzzy parameters using p-fuzzy systems and fuzzy ODEs.
                Simulate discrete and continuous dynamics.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Stats section (moved after cards)
    st.markdown("""
    <div class="section-header">Library Overview</div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="stat-box animate-fade-in">
            <div class="stat-number">8+</div>
            <div class="stat-label">Core Algorithms</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="stat-box animate-fade-in" style="animation-delay: 0.1s;">
            <div class="stat-number">18+</div>
            <div class="stat-label">Example Notebooks</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="stat-box animate-fade-in" style="animation-delay: 0.2s;">
            <div class="stat-number">3</div>
            <div class="stat-label">Main Modules</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="stat-box animate-fade-in" style="animation-delay: 0.3s;">
            <div class="stat-number">100%</div>
            <div class="stat-label">Open Source</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Getting started section
    st.markdown("""
    <div class="section-header">Getting Started</div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        ### Installation

        Install pyfuzzy-toolbox using pip:

        ```bash
        pip install pyfuzzy-toolbox
        ```

        ### Quick Example

        Create a simple Mamdani system:

        ```python
        import fuzzy_systems as fs

        # Create system
        system = fs.MamdaniSystem()
        system.add_input('temperature', (0, 40))
        system.add_output('fan_speed', (0, 100))

        # Add terms
        system.add_term('temperature', 'cold', 'triangular', (0, 0, 20))
        system.add_term('temperature', 'hot', 'triangular', (20, 40, 40))
        system.add_term('fan_speed', 'slow', 'triangular', (0, 0, 50))
        system.add_term('fan_speed', 'fast', 'triangular', (50, 100, 100))

        # Add rules
        system.add_rules([
            ('cold', 'slow'),
            ('hot', 'fast')
        ])

        # Evaluate
        result = system.evaluate(temperature=25)
        print(f"Fan speed: {result['fan_speed']:.1f}%")
        ```
        """)

    with col2:
        st.markdown("""
        ### Resources

        **Documentation**
        [Read the full documentation →](https://1moi6.github.io/pyfuzzy-toolbox/)

        **Source Code**
        [GitHub Repository →](https://github.com/1moi6/pyfuzzy-toolbox)

        **PyPI Package**
        [View on PyPI →](https://pypi.org/project/pyfuzzy-toolbox/)

        **Examples**
        [18+ Colab Notebooks →](https://github.com/1moi6/pyfuzzy-toolbox/tree/main/notebooks_colab)

        ### Citation

        ```bibtex
        @software{pyfuzzy_toolbox,
          title = {pyfuzzy-toolbox},
          author = {Cecconello, Moiseis},
          year = {2025},
          url = {https://github.com/1moi6/pyfuzzy-toolbox}
        }
        ```
        """)

    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #9ca3af; padding: 2rem 0;">
        <p>pyfuzzy-toolbox v1.0.7 | MIT License | © 2025 Moiseis Cecconello</p>
    </div>
    """, unsafe_allow_html=True)
