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


def run():
    """Render dynamic systems page"""

    # Initialize session state for dynamics
    if 'dynamics_system_type' not in st.session_state:
        st.session_state.dynamics_system_type = "p-Fuzzy Discrete"
    if 'selected_fis_for_dynamics' not in st.session_state:
        st.session_state.selected_fis_for_dynamics = None

    # Sidebar
    with st.sidebar:
        # Navigation pills
        selected_page = st.pills(
            "Navigation",
            ["🏠 Home", "⚙️ Inference", "🧠 Learning", "📊 Dynamics"],
            selection_mode="single",
            default="📊 Dynamics",
            label_visibility="collapsed"
        )

        # Handle navigation
        if selected_page == "🏠 Home":
            st.session_state.page = 'home'
            st.rerun()
        elif selected_page == "⚙️ Inference":
            st.session_state.page = 'inference'
            st.rerun()
        elif selected_page == "🧠 Learning":
            st.session_state.page = 'learning'
            st.rerun()

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

        # FIS selection from inference module
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
        render_fuzzy_ode_interface()


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
    if st.button("▶️ Run Simulation", type="primary", use_container_width=True):
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

                st.plotly_chart(fig, use_container_width=True)

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

                    st.plotly_chart(fig_phase, use_container_width=True)

            # Export data
            with st.expander("💾 Export Data"):
                import pandas as pd

                # Create DataFrame
                data = {'time': time}
                for i, var_name in enumerate(state_vars):
                    data[var_name] = trajectory[:, i]

                df = pd.DataFrame(data)

                st.dataframe(df, use_container_width=True)

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
    if st.button("▶️ Run Simulation", type="primary", use_container_width=True, key="run_cont"):
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

                st.plotly_chart(fig, use_container_width=True)

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

                    st.plotly_chart(fig_phase, use_container_width=True)

            # Export data
            with st.expander("💾 Export Data"):
                import pandas as pd

                # Create DataFrame
                data = {'time': time}
                for i, var_name in enumerate(state_vars):
                    data[var_name] = trajectory[:, i]

                df = pd.DataFrame(data)

                st.dataframe(df, use_container_width=True)

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
    """Render Fuzzy ODE solver interface"""

    st.markdown("### Fuzzy ODE Solver")

    st.info("🚧 **Coming Soon**: Interactive Fuzzy ODE solver")

    st.markdown("""
    **Fuzzy ODE Features**:
    - α-level method for uncertainty propagation
    - Fuzzy initial conditions and parameters
    - Multiple ODE solvers (RK45, RK4, Euler)
    - Envelope visualization
    - Monte Carlo option for high dimensions
    """)

    with st.expander("💻 View Example Code"):
        st.code("""
from fuzzy_systems.dynamics import FuzzyODE
from fuzzy_systems.core import FuzzySet

# Define ODE
def logistic(t, x, r, K):
    return r * x * (1 - x / K)

# Create fuzzy parameters
r_fuzzy = FuzzySet(name='r', mf_type='triangular', params=(0.8, 1.0, 1.2))
K_fuzzy = FuzzySet(name='K', mf_type='triangular', params=(90, 100, 110))

# Solve
solver = FuzzyODE(
    ode_func=logistic,
    t_span=(0, 20),
    x0=10,
    fuzzy_params={'r': r_fuzzy, 'K': K_fuzzy},
    alpha_levels=[0, 0.25, 0.5, 0.75, 1.0]
)

results = solver.solve()
solver.plot_envelope()
        """, language='python')
