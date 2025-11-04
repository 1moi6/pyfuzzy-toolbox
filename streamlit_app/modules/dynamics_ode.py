"""
Dynamic Systems Module
Interface for Fuzzy ODEs and p-Fuzzy systems
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fuzzy_systems.dynamics import PFuzzyDiscrete
from modules.inference_engine import InferenceEngine
from modules import fuzzy_ode_module


def run():

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 0.25rem 0 0.125rem 0; margin-top: 0.5rem;">
            <h2 style="margin: 0.25rem 0 0.125rem 0; color: #667eea;">Fuzzy ODE Solver</h2>
            <p style="color: #6b7280; font-size: 0.9rem; margin: 0;">
                Solve ODEs with fuzzy parameters
            </p>
        </div>
        <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 0.25rem 0 0.5rem 0;">
        """, unsafe_allow_html=True)

        # System type selection

        # st.markdown("<hr style='border: none; border-top: 1px solid #e5e7eb; margin: 0.5rem 0;'>", unsafe_allow_html=True)


    # Main content
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem 0;">
        <h3 style="color: #6b7280; font-weight: 500; margin: 0; font-size: 1.1rem;">
            Differencial Equations With Fuzzy Uncertainty
        </h3>
    </div>
    """, unsafe_allow_html=True)

    # st.markdown("<div style='border-bottom: 1px solid #e5e7eb; margin: 0.5rem 0 1.5rem 0;'></div>", unsafe_allow_html=True)

    # Render appropriate interface based on system type
    fuzzy_ode_module.run()



def render_fuzzy_ode_interface():
    """Render Fuzzy ODE solver interface with full implementation"""

    # Initialize session state
    if 'ode_system_type' not in st.session_state:
        st.session_state.ode_system_type = "Pre-defined"
    if 'selected_predefined_system' not in st.session_state:
        st.session_state.selected_predefined_system = "Logistic Growth"
    if 'n_custom_vars' not in st.session_state:
        st.session_state.n_custom_vars = 2
    if 'fuzzy_params_config' not in st.session_state:
        st.session_state.fuzzy_params_config = {}

    # Get ODE system configuration
    if st.session_state.ode_system_type == "Pre-defined":
        ode_config = get_predefined_ode_config(st.session_state.selected_predefined_system)
    else:
        ode_config = get_custom_ode_config()
        if ode_config is None:
            # Show custom ODE definition UI
            render_custom_ode_definition()
            return

    # Show system equations
    with st.expander(f"System Equations  - {ode_config['name']}", expanded=True):
        for i, (var, eq) in enumerate(zip(ode_config['vars'], ode_config['equations'])):
            st.code(f"d{var}/dt = {eq}", language="python")

    # Configuration and simulation
    render_ode_configuration_and_solve(ode_config)


def render_ode_sidebar():
    """Render Fuzzy ODE sidebar configuration"""

    st.markdown("**ODE System**")

    # System type selection
    system_type = st.radio(
        "Type",
        ["Pre-defined", "Custom"],
        key="ode_system_type",
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border: none; border-top: 1px solid #e5e7eb; margin: 0.5rem 0;'>", unsafe_allow_html=True)

    if system_type == "Pre-defined":
        predefined_systems = {
            "Logistic Growth": "1D Population model",
            "Lotka-Volterra": "2D Predator-prey",
            "SIR Model": "3D Epidemic model",
            "Van der Pol": "2D Oscillator"
        }

        system_name = st.selectbox(
            "Select System",
            list(predefined_systems.keys()),
            key="selected_predefined_system",
            format_func=lambda x: f"{x}",
            help=predefined_systems[st.session_state.get('selected_predefined_system', 'Logistic Growth')]
        )

    else:  # Custom
        n_vars = st.number_input(
            "Number of variables",
            min_value=1,
            max_value=5,
            value=st.session_state.n_custom_vars,
            key="n_custom_vars"
        )

        st.caption("Define equations in main panel →")


def get_predefined_ode_config(system_name):
    """Returns configuration for pre-defined ODE system"""

    systems = {
        "Logistic Growth": {
            "name": "Logistic Growth",
            "dim": 1,
            "vars": ["x"],
            "equations": ["r * x[0] * (1 - x[0] / K)"],
            "params": ["r", "K"],
            "default_params": {"r": 0.5, "K": 100},
            "default_ic": [10.0],
            "ic_ranges": [(0.0, 150.0)]
        },
        "Lotka-Volterra": {
            "name": "Lotka-Volterra (Predator-Prey)",
            "dim": 2,
            "vars": ["Prey", "Predator"],
            "equations": [
                "alpha * x[0] - beta * x[0] * x[1]",
                "delta * x[0] * x[1] - gamma * x[1]"
            ],
            "params": ["alpha", "beta", "delta", "gamma"],
            "default_params": {"alpha": 1.0, "beta": 0.1, "delta": 0.075, "gamma": 1.5},
            "default_ic": [40.0, 9.0],
            "ic_ranges": [(0.0, 100.0), (0.0, 50.0)]
        },
        "SIR Model": {
            "name": "SIR Epidemic Model",
            "dim": 3,
            "vars": ["S", "I", "R"],
            "equations": [
                "-beta * x[0] * x[1] / N",
                "beta * x[0] * x[1] / N - gamma * x[1]",
                "gamma * x[1]"
            ],
            "params": ["beta", "gamma", "N"],
            "default_params": {"beta": 0.5, "gamma": 0.1, "N": 1000},
            "default_ic": [990.0, 10.0, 0.0],
            "ic_ranges": [(0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0)]
        },
        "Van der Pol": {
            "name": "Van der Pol Oscillator",
            "dim": 2,
            "vars": ["x", "v"],
            "equations": [
                "x[1]",
                "mu * (1 - x[0]**2) * x[1] - x[0]"
            ],
            "params": ["mu"],
            "default_params": {"mu": 1.0},
            "default_ic": [1.0, 0.0],
            "ic_ranges": [(-3.0, 3.0), (-3.0, 3.0)]
        },
    }

    return systems.get(system_name)


def get_custom_ode_config():
    """Returns configuration for custom ODE system"""

    n_vars = st.session_state.n_custom_vars
    var_names = [st.session_state.get(f"custom_var_name_{i}", f"x{i}") for i in range(n_vars)]
    equations = [st.session_state.get(f"custom_equation_{i}", "") for i in range(n_vars)]

    # Check if all equations are provided
    if not all(equations):
        return None

    return {
        "name": "Custom ODE System",
        "dim": n_vars,
        "vars": var_names,
        "equations": equations,
        "params": [],
        "default_params": {},
        "default_ic": [1.0] * n_vars,
        "ic_ranges": [(-10.0, 10.0)] * n_vars
    }


def render_custom_ode_definition():
    """Render UI for custom ODE definition in main panel"""

    st.markdown("### 📝 Define Custom ODE System")

    n_vars = st.session_state.n_custom_vars

    st.caption(f"Define {n_vars} differential equation(s):")

    for i in range(n_vars):
        col1, col2 = st.columns([1, 4])

        with col1:
            var_name = st.text_input(
                f"Var {i+1}",
                value=st.session_state.get(f"custom_var_name_{i}", f"x{i}" if i > 0 else "x"),
                key=f"custom_var_name_{i}",
                placeholder=f"x{i}"
            )

        with col2:
            equation = st.text_input(
                f"d{var_name}/dt =",
                key=f"custom_equation_{i}",
                placeholder=f"e.g., r * x[{i}] * (1 - x[{i}] / K)",
                help="Use x[0], x[1], ... for state variables"
            )

    with st.expander("💡 How to write equations"):
        st.markdown("""
        **Syntax:**
        - **State variables**: `x[0]`, `x[1]`, `x[2]`, ...
        - **Parameters**: Use names directly: `r`, `K`, `alpha`, `beta`, etc.
        - **Operations**: `+`, `-`, `*`, `/`, `**` (power)
        - **Functions**: `sin()`, `cos()`, `exp()`, `log()`, `sqrt()`, `abs()`

        **Examples:**
        - 1D Logistic: `r * x[0] * (1 - x[0] / K)`
        - 2D Lotka-Volterra:
          - Prey: `alpha * x[0] - beta * x[0] * x[1]`
          - Predator: `delta * x[0] * x[1] - gamma * x[1]`
        """)


def render_ode_configuration_and_solve(ode_config):
    """Render configuration UI and solve Fuzzy ODE"""

    st.markdown("#### ⚙️ Simulation Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        t_end = st.number_input("Simulation time", min_value=1.0, max_value=1000.0, value=50.0, step=5.0)

    with col2:
        n_alpha = st.number_input("α-levels", min_value=3, max_value=21, value=11, step=2,
                                  help="Number of α-cut levels")

    with col3:
        method = st.selectbox("Solver", ["RK45", "RK23", "DOP853"],
                             help="Integration method")

    st.markdown("#### 🎯 Initial Conditions")

    initial_conditions = []
    cols = st.columns(min(ode_config['dim'], 3))

    for i in range(ode_config['dim']):
        with cols[i % len(cols)]:
            var_name = ode_config['vars'][i]
            default_val = ode_config['default_ic'][i]
            min_val, max_val = ode_config['ic_ranges'][i]

            ic_value = st.number_input(
                f"{var_name}₀",
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                key=f"ic_{var_name}"
            )
            initial_conditions.append(ic_value)

    st.markdown("#### 🌫️ Fuzzy Parameters")

    # Extract parameters
    import re
    all_params = set()
    for eq in ode_config['equations']:
        tokens = re.findall(r'\b[a-zA-Z_]\w*\b', eq)
        for token in tokens:
            if token not in ['x', 'sin', 'cos', 'exp', 'log', 'sqrt', 'abs', 't']:
                all_params.add(token)

    all_params = sorted(all_params)

    if not all_params:
        st.info("ℹ️ No parameters detected. Add parameters like 'r', 'K', etc.")
        return

    fuzzy_params = {}
    crisp_params = {}

    for param_name in all_params:
        with st.expander(f"📊 **{param_name}**", expanded=True):
            col1, col2 = st.columns([1, 3])

            with col1:
                is_fuzzy = st.checkbox("Fuzzy", value=True, key=f"fuzzy_{param_name}")

            with col2:
                if is_fuzzy:
                    subcol1, subcol2, subcol3, subcol4 = st.columns(4)

                    with subcol1:
                        mf_type = st.selectbox(
                            "Type",
                            ["triangular", "gaussian"],
                            key=f"mf_{param_name}",
                            label_visibility="collapsed"
                        )

                    default_val = ode_config['default_params'].get(param_name, 1.0)

                    if mf_type == "triangular":
                        with subcol2:
                            a = st.number_input("Min", value=default_val * 0.8, key=f"a_{param_name}", format="%.4f")
                        with subcol3:
                            b = st.number_input("Peak", value=default_val, key=f"b_{param_name}", format="%.4f")
                        with subcol4:
                            c = st.number_input("Max", value=default_val * 1.2, key=f"c_{param_name}", format="%.4f")

                        from fuzzy_systems.dynamics.fuzzy_ode import FuzzyNumber
                        fuzzy_params[param_name] = FuzzyNumber.triangular(center=b, spread=(c-a)/2, name=param_name)

                    else:  # gaussian
                        with subcol2:
                            mean = st.number_input("Mean", value=default_val, key=f"mean_{param_name}", format="%.4f")
                        with subcol3:
                            sigma = st.number_input("Sigma", value=default_val * 0.1, key=f"sigma_{param_name}", format="%.4f")

                        from fuzzy_systems.dynamics.fuzzy_ode import FuzzyNumber
                        fuzzy_params[param_name] = FuzzyNumber.gaussian(mean=mean, sigma=sigma, name=param_name)
                else:
                    crisp_value = st.number_input(
                        f"Value",
                        value=ode_config['default_params'].get(param_name, 1.0),
                        key=f"crisp_{param_name}",
                        format="%.4f",
                        label_visibility="collapsed"
                    )
                    crisp_params[param_name] = crisp_value

    all_params_dict = {**fuzzy_params, **crisp_params}

    # Solve
    if st.button("▶️ Solve Fuzzy ODE", type="primary", use_container_width=True):
        try:
            ode_func = build_ode_function(ode_config['equations'], all_params.union({'t'}))

            from fuzzy_systems.dynamics.fuzzy_ode import FuzzyODESolver

            with st.spinner("Solving Fuzzy ODE..."):
                solver = FuzzyODESolver(
                    ode_func=ode_func,
                    t_span=(0, t_end),
                    initial_condition=initial_conditions,
                    params=all_params_dict,
                    n_alpha_cuts=n_alpha,
                    method=method,
                    var_names=ode_config['vars']
                )

                solution = solver.solve(method='standard', n_grid_points=5, verbose=False)

            st.success("✅ Fuzzy ODE solved successfully!")
            render_fuzzy_ode_results(solution, ode_config)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            with st.expander("🐛 Debug"):
                st.code(traceback.format_exc())


def build_ode_function(equations, safe_names):
    """Build ODE function from string equations"""

    func_code = "def ode_system(t, x"
    param_names = sorted(safe_names - {'t', 'x'})

    if param_names:
        func_code += ", " + ", ".join(param_names)
    func_code += "):\n"
    func_code += "    from numpy import sin, cos, exp, log, sqrt, abs\n"
    func_code += "    import numpy as np\n"
    func_code += "    return np.array([\n"

    for eq in equations:
        func_code += f"        {eq},\n"
    func_code += "    ])\n"

    namespace = {}
    exec(func_code, namespace)
    return namespace['ode_system']


def render_fuzzy_ode_results(solution, ode_config):
    """Render Fuzzy ODE results - time evolution only"""

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    n_vars = len(ode_config['vars'])

    with st.expander("📈 Time Evolution", expanded=True):
        fig = make_subplots(
            rows=n_vars,
            cols=1,
            subplot_titles=[f"{var} vs Time" for var in ode_config['vars']],
            vertical_spacing=0.15 if n_vars > 1 else 0.1
        )

        for var_idx in range(n_vars):
            for alpha_idx, alpha in enumerate(solution.alphas):
                y_min, y_max = solution.get_alpha_level(alpha)

                opacity = 0.3 + 0.7 * alpha

                fig.add_trace(
                    go.Scatter(
                        x=solution.t,
                        y=y_min[var_idx],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=var_idx + 1,
                    col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=solution.t,
                        y=y_max[var_idx],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor=f'rgba(100, 150, 255, {opacity*0.4})',
                        name=f'α={alpha:.2f}' if var_idx == 0 and alpha_idx % 2 == 0 else None,
                        showlegend=var_idx == 0 and alpha_idx % 2 == 0,
                        hovertemplate=f'α={alpha:.2f}<br>t=%{{x:.2f}}<br>%{{y:.4f}}'
                    ),
                    row=var_idx + 1,
                    col=1
                )

            fig.update_xaxis(title_text="Time", row=var_idx + 1, col=1)
            fig.update_yaxis(title_text=ode_config['vars'][var_idx], row=var_idx + 1, col=1)

        fig.update_layout(height=300 * n_vars, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("💾 Export Data"):
        import pandas as pd

        alpha_export = st.selectbox("α-level", solution.alphas, index=len(solution.alphas)//2)
        df = solution.to_dataframe(alpha=alpha_export)

        st.dataframe(df.head(20), use_container_width=True)

        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"fuzzy_ode_alpha_{alpha_export:.2f}.csv",
            mime="text/csv"
        )
