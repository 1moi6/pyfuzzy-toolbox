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
    """Render dynamic systems page"""

    # Initialize session state for dynamics
    if 'dynamics_system_type' not in st.session_state:
        st.session_state.dynamics_system_type = "p-Fuzzy Discrete"
    if 'selected_fis_for_dynamics' not in st.session_state:
        st.session_state.selected_fis_for_dynamics = None

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 0.25rem 0 0.125rem 0; margin-top: 0.5rem;">
            <h2 style="margin: 0.25rem 0 0.125rem 0; color: #667eea;">Dynamic Systems</h2>
            <p style="color: #6b7280; font-size: 0.9rem; margin: 0;">
                Model temporal evolution with fuzzy uncertainty
            </p>
        </div>
        <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 0.25rem 0 0.5rem 0;">
        """, unsafe_allow_html=True)

        # System type selection
        st.markdown("**System Type**")
        system_type = st.selectbox(
            "Choose dynamics system",
            [
                "p-Fuzzy Discrete",
                "p-Fuzzy Continuous",
                "Fuzzy ODE"
            ],
            key="dynamics_system_type",
            label_visibility="collapsed"
        )

        st.markdown("<hr style='border: none; border-top: 1px solid #e5e7eb; margin: 0.5rem 0;'>", unsafe_allow_html=True)

        # System-specific sidebar content (only for p-Fuzzy systems)
        if system_type != "Fuzzy ODE":
            # FIS selection for p-Fuzzy systems
            st.markdown("**Select FIS for Dynamics**")

            # Check if there are FIS available from inference module
            if 'fis_list' in st.session_state and len(st.session_state.fis_list) > 0:
                fis_names = [f"{fis['name']} ({fis['type']})" for fis in st.session_state.fis_list]

                selected_fis_idx = st.selectbox(
                    "Available FIS from Inference",
                    range(len(st.session_state.fis_list)),
                    format_func=lambda x: fis_names[x],
                    key="selected_fis_for_dynamics",
                    label_visibility="collapsed",
                    index=0
                )

                # Show FIS info
                selected_fis = st.session_state.fis_list[selected_fis_idx]

                with st.expander("📋 FIS Info"):
                    st.caption(f"**Name:** {selected_fis['name']}")
                    st.caption(f"**Type:** {selected_fis['type']}")
                    st.caption(f"**Inputs:** {len(selected_fis['input_variables'])}")
                    st.caption(f"**Outputs:** {len(selected_fis['output_variables'])}")
                    st.caption(f"**Rules:** {len(selected_fis['fuzzy_rules'])}")

                # Show state variables
                st.markdown("<hr style='border: none; border-top: 1px solid #e5e7eb; margin: 0.5rem 0;'>", unsafe_allow_html=True)
                st.markdown("**State Variables**")
                state_var_names = [var['name'] for var in selected_fis['input_variables']]
                if state_var_names:
                    st.caption(f"{', '.join(state_var_names)}")
                else:
                    st.caption("No state variables defined")

            else:
                st.warning("⚠️ No FIS available")
                st.info("Go to **Inference** module to create or load a FIS first")

    # Main content
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem 0;">
        <h3 style="color: #6b7280; font-weight: 500; margin: 0; font-size: 1.1rem;">
            Dynamic Systems
        </h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='border-bottom: 1px solid #e5e7eb; margin: 0.5rem 0 1.5rem 0;'></div>", unsafe_allow_html=True)

    # Render appropriate interface based on system type
    if system_type == "p-Fuzzy Discrete":
        render_pfuzzy_discrete_interface()
    elif system_type == "p-Fuzzy Continuous":
        render_pfuzzy_continuous_interface()
    elif system_type == "Fuzzy ODE":
        fuzzy_ode_module.run()


def render_pfuzzy_discrete_interface():
    """Render p-Fuzzy discrete interface with full implementation"""

    # Check if FIS is available
    if 'fis_list' not in st.session_state or len(st.session_state.fis_list) == 0:
        st.warning("⚠️ **No FIS available**")
        st.info("Please go to the **Inference** module to create or load a FIS first")
        return

    # Get selected FIS
    fis_idx = st.session_state.selected_fis_for_dynamics
    selected_fis = st.session_state.fis_list[fis_idx]

    # Validate FIS for p-Fuzzy
    input_vars = selected_fis['input_variables']
    output_vars = selected_fis['output_variables']

    if len(input_vars) == 0 or len(output_vars) == 0:
        st.error("❌ FIS must have at least one input and one output variable")
        return

    # Configuration section
    st.markdown("### ⚙️ Simulation Configuration")

    col1, col2 = st.columns(2)

    with col1:
        # Mode selection
        mode = st.selectbox(
            "Mode",
            ["relative", "absolute"],
            help="Relative: x_{n+1} = x_n + f(x_n), Absolute: x_{n+1} = f(x_n)"
        )

    with col2:
        # Number of steps
        n_steps = st.number_input(
            "Number of time steps",
            min_value=10,
            max_value=10000,
            value=100,
            step=10
        )

    # Map each input variable to a state variable
    state_vars = []
    for var in input_vars:
        state_vars.append(var['name'])

    # Initial conditions
    st.markdown("### Initial Conditions")

    # Initial condition inputs
    initial_conditions = {}
    cols = st.columns(min(len(state_vars), 3))

    for idx, var in enumerate(input_vars):
        with cols[idx % len(cols)]:
            initial_conditions[var['name']] = st.number_input(
                f"{var['name']} (x0)",
                min_value=float(var['min']),
                max_value=float(var['max']),
                value=float((var['min'] + var['max']) / 2),
                key=f"ic_{var['name']}"
            )

    # Simulate button
    if st.button("▶️ Run Simulation", type="primary", width="stretch"):
        try:
            # Import p-fuzzy module
            # Build FIS using inference engine
            engine = InferenceEngine(selected_fis)

            # Create p-Fuzzy system
            pfuzzy = PFuzzyDiscrete(
                fis=engine.system,
                mode=mode,
                state_vars=state_vars
            )

            # Run simulation
            with st.spinner("Simulating..."):
                time, trajectory = pfuzzy.simulate(x0=initial_conditions, n_steps=n_steps)

            # Display results
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📊 Simulation Results")

            n_vars = len(state_vars)

            # Time Evolution (always show)
            with st.expander("📈 Time Evolution", expanded=True):
                fig = go.Figure()

                for i, var_name in enumerate(state_vars):
                    fig.add_trace(go.Scatter(
                        x=time,
                        y=trajectory[:, i],
                        mode='lines',
                        name=var_name,
                        line=dict(width=2)
                    ))

                fig.update_layout(
                    title="State Variables over Time",
                    xaxis_title="Time Step",
                    yaxis_title="Value",
                    hovermode='closest',
                    height=400
                )

                st.plotly_chart(fig, width="stretch")

            # Phase Space (only if 2+ variables)
            if n_vars >= 2:
                with st.expander("🔄 Phase Space", expanded=True):
                    # Variable selection for phase space
                    col1, col2 = st.columns(2)
                    with col1:
                        var_x_idx = st.selectbox(
                            "X-axis variable",
                            range(n_vars),
                            format_func=lambda x: state_vars[x],
                            key="phase_x_discrete"
                        )
                    with col2:
                        # Default to second variable, or first if only one
                        default_y = 1 if n_vars > 1 and var_x_idx != 1 else (0 if var_x_idx != 0 else 1 if n_vars > 1 else 0)
                        var_y_idx = st.selectbox(
                            "Y-axis variable",
                            range(n_vars),
                            format_func=lambda x: state_vars[x],
                            index=default_y,
                            key="phase_y_discrete"
                        )

                    # Phase space plot
                    fig_phase = go.Figure()

                    # Trajectory
                    fig_phase.add_trace(go.Scatter(
                        x=trajectory[:, var_x_idx],
                        y=trajectory[:, var_y_idx],
                        mode='lines',
                        name="Trajectory",
                        line=dict(width=2)
                    ))

                    # Initial condition marker
                    fig_phase.add_trace(go.Scatter(
                        x=[trajectory[0, var_x_idx]],
                        y=[trajectory[0, var_y_idx]],
                        mode='markers',
                        name="Initial",
                        marker=dict(size=12, color='green', symbol='star')
                    ))

                    # Final condition marker
                    fig_phase.add_trace(go.Scatter(
                        x=[trajectory[-1, var_x_idx]],
                        y=[trajectory[-1, var_y_idx]],
                        mode='markers',
                        name="Final",
                        marker=dict(size=12, color='red', symbol='square')
                    ))

                    fig_phase.update_layout(
                        title=f"Phase Space: {state_vars[var_x_idx]} vs {state_vars[var_y_idx]}",
                        xaxis_title=state_vars[var_x_idx],
                        yaxis_title=state_vars[var_y_idx],
                        hovermode='closest',
                        height=400
                    )

                    st.plotly_chart(fig_phase, width="stretch")

            # Export data
            with st.expander("💾 Export Data"):
                import pandas as pd

                # Create DataFrame
                data = {'time': time}
                for i, var_name in enumerate(state_vars):
                    data[var_name] = trajectory[:, i]

                df = pd.DataFrame(data)

                st.dataframe(df, width="stretch")

                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"{selected_fis['name']}_pfuzzy_discrete.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"❌ Error during simulation: {str(e)}")
            import traceback
            with st.expander("🐛 Debug Info"):
                st.code(traceback.format_exc())


def render_pfuzzy_continuous_interface():
    """Render p-Fuzzy continuous interface with full implementation"""

    # Check if FIS is available
    if 'fis_list' not in st.session_state or len(st.session_state.fis_list) == 0:
        st.warning("⚠️ **No FIS available**")
        st.info("Please go to the **Inference** module to create or load a FIS first")
        return

    # Get selected FIS
    fis_idx = st.session_state.selected_fis_for_dynamics
    selected_fis = st.session_state.fis_list[fis_idx]

    # Validate FIS for p-Fuzzy
    input_vars = selected_fis['input_variables']
    output_vars = selected_fis['output_variables']

    if len(input_vars) == 0 or len(output_vars) == 0:
        st.error("❌ FIS must have at least one input and one output variable")
        return

    # Configuration section
    st.markdown("### ⚙️ Simulation Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Time span
        t_end = st.number_input(
            "Simulation time",
            min_value=1.0,
            max_value=1000.0,
            value=50.0,
            step=1.0
        )

    with col2:
        # Time step
        dt = st.number_input(
            "Time step (dt)",
            min_value=0.001,
            max_value=1.0,
            value=0.1,
            step=0.01,
            format="%.3f"
        )

    with col3:
        # Integration method
        method = st.selectbox("Integration method", ["rk4", "euler"])

    # Map each input variable to a state variable
    state_vars = []
    for var in input_vars:
        state_vars.append(var['name'])

    # Initial conditions
    st.markdown("### Initial Conditions")

    # Initial condition inputs
    initial_conditions = {}
    cols = st.columns(min(len(state_vars), 3))

    for idx, var in enumerate(input_vars):
        with cols[idx % len(cols)]:
            initial_conditions[var['name']] = st.number_input(
                f"{var['name']} (x0)",
                min_value=float(var['min']),
                max_value=float(var['max']),
                value=float((var['min'] + var['max']) / 2),
                key=f"ic_cont_{var['name']}"
            )

    # Simulate button
    if st.button("▶️ Run Simulation", type="primary", width="stretch", key="run_cont"):
        try:
            # Import p-fuzzy module
            from fuzzy_systems.dynamics import PFuzzyContinuous
            from modules.inference_engine import InferenceEngine

            # Build FIS using inference engine
            engine = InferenceEngine(selected_fis)

            # Create p-Fuzzy system
            pfuzzy = PFuzzyContinuous(
                fis=engine.system,
                state_vars=state_vars
            )

            # Run simulation
            with st.spinner("Simulating..."):
                time, trajectory = pfuzzy.simulate(
                    x0=initial_conditions,
                    t_span=(0, t_end),
                    dt=dt,
                    method=method
                )

            # Display results
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📊 Simulation Results")

            n_vars = len(state_vars)

            # Time Evolution - Always shown
            with st.expander("📈 Time Evolution", expanded=True):
                fig = go.Figure()

                # Plot all variables
                for i, var_name in enumerate(state_vars):
                    fig.add_trace(go.Scatter(
                        x=time,
                        y=trajectory[:, i],
                        mode='lines',
                        name=var_name,
                        line=dict(width=2)
                    ))

                fig.update_layout(
                    title="State Variables Over Time",
                    xaxis_title="Time",
                    yaxis_title="Value",
                    hovermode='closest',
                    height=400
                )

                st.plotly_chart(fig, width="stretch")

            # Phase Space - Only for 2+ variables
            if n_vars >= 2:
                with st.expander("🔄 Phase Space", expanded=True):
                    # Variable selection for phase space
                    col1, col2 = st.columns(2)
                    with col1:
                        var_x_idx = st.selectbox(
                            "X-axis variable",
                            options=list(range(n_vars)),
                            format_func=lambda i: state_vars[i],
                            key="continuous_phase_x"
                        )
                    with col2:
                        var_y_idx = st.selectbox(
                            "Y-axis variable",
                            options=list(range(n_vars)),
                            format_func=lambda i: state_vars[i],
                            index=min(1, n_vars-1),
                            key="continuous_phase_y"
                        )

                    # Create phase space plot
                    fig_phase = go.Figure()

                    # Trajectory
                    fig_phase.add_trace(
                        go.Scatter(
                            x=trajectory[:, var_x_idx],
                            y=trajectory[:, var_y_idx],
                            mode='lines',
                            name="Trajectory",
                            line=dict(width=2, color='blue')
                        )
                    )

                    # Initial point
                    fig_phase.add_trace(
                        go.Scatter(
                            x=[trajectory[0, var_x_idx]],
                            y=[trajectory[0, var_y_idx]],
                            mode='markers',
                            name="Initial",
                            marker=dict(size=12, color='green', symbol='star')
                        )
                    )

                    # Final point
                    fig_phase.add_trace(
                        go.Scatter(
                            x=[trajectory[-1, var_x_idx]],
                            y=[trajectory[-1, var_y_idx]],
                            mode='markers',
                            name="Final",
                            marker=dict(size=12, color='red', symbol='square')
                        )
                    )

                    fig_phase.update_layout(
                        title=f"Phase Portrait: {state_vars[var_x_idx]} vs {state_vars[var_y_idx]}",
                        xaxis_title=state_vars[var_x_idx],
                        yaxis_title=state_vars[var_y_idx],
                        hovermode='closest',
                        height=400
                    )

                    st.plotly_chart(fig_phase, width="stretch")

            # Export data
            with st.expander("💾 Export Data"):
                import pandas as pd

                # Create DataFrame
                data = {'time': time}
                for i, var_name in enumerate(state_vars):
                    data[var_name] = trajectory[:, i]

                df = pd.DataFrame(data)

                st.dataframe(df, width="stretch")

                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"{selected_fis['name']}_pfuzzy_continuous.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"❌ Error during simulation: {str(e)}")
            import traceback
            with st.expander("🐛 Debug Info"):
                st.code(traceback.format_exc())


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

    # Main content
    st.markdown(f"""
    <div style="text-align: center; padding: 0.5rem 0;">
        <h3 style="color: #6b7280; font-weight: 500; margin: 0; font-size: 1.1rem;">
            {ode_config.get('name', 'Fuzzy ODE Solver')}
        </h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='border-bottom: 1px solid #e5e7eb; margin: 0.5rem 0 1.5rem 0;'></div>", unsafe_allow_html=True)

    # Show system equations
    with st.expander("📖 System Equations", expanded=False):
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
    if st.button("▶️ Solve Fuzzy ODE", type="primary", width="stretch"):
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
        st.plotly_chart(fig, width="stretch")

    with st.expander("💾 Export Data"):
        import pandas as pd

        alpha_export = st.selectbox("α-level", solution.alphas, index=len(solution.alphas)//2)
        df = solution.to_dataframe(alpha=alpha_export)

        st.dataframe(df.head(20), width="stretch")

        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"fuzzy_ode_alpha_{alpha_export:.2f}.csv",
            mime="text/csv"
        )
