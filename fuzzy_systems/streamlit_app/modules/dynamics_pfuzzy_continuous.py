"""
Dynamic Systems Module
Interface for Fuzzy ODEs and p-Fuzzy systems
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fuzzy_systems.dynamics import PFuzzyContinuous
from modules.inference_engine import InferenceEngine
import traceback
import re
from typing import Tuple


def check_fis_available(show_styled_button: bool = False) -> bool:
    """Centraliza verificação de FIS disponível com mensagem padrão.

    Args:
        show_styled_button: Se True, mostra botão estilizado com gradiente

    Returns:
        True se FIS está disponível, False caso contrário (exibe mensagem)
    """
    if 'fis_list' not in st.session_state or len(st.session_state.fis_list) == 0:
        st.warning("⚠️ **No FIS available**")
        st.info("Please go to the **Inference** module to create or load a FIS first")

        if 'app_pages' in st.session_state:
            if show_styled_button:
                # Botão primário estilizado (página principal)
                if st.button("🚀 Go to Inference Module", type="primary", width="stretch", key="go_to_inference_continuous"):
                    st.switch_page(st.session_state['app_pages'][0])
            else:
                # Botão primário para sidebar também
                if st.button("Go to Inference Page", type="primary", width="stretch", key="sidebar_go_to_inference_continuous"):
                    st.switch_page(st.session_state['app_pages'][0])

        return False
    return True


def validate_equation(equation: str, n_vars: int) -> Tuple[bool, str]:
    """Valida sintaxe de equações antes de executar.

    Args:
        equation: String com a equação a ser validada
        n_vars: Número de variáveis de estado disponíveis

    Returns:
        (is_valid, error_message): Tupla com resultado da validação
    """
    if not equation or equation.strip() == "":
        return False, "Equação vazia"

    # Caracteres permitidos: letras, números, [], ., +, -, *, /, (), espaços
    allowed_pattern = r'^[\w\s\[\]\.+\-*/()]+$'

    if not re.match(allowed_pattern, equation):
        return False, "Caracteres inválidos detectados. Use apenas letras, números e operadores matemáticos"

    # Verifica se todos os índices x[i] estão no range válido
    x_indices = re.findall(r'x\[(\d+)\]', equation)
    for idx in x_indices:
        if int(idx) >= n_vars:
            return False, f"Índice x[{idx}] fora do range válido [0, {n_vars-1}]"

    # Verifica se todos os índices f[i] estão no range válido
    f_indices = re.findall(r'f\[(\d+)\]', equation)
    for idx in f_indices:
        if int(idx) >= n_vars:
            return False, f"Índice f[{idx}] fora do range válido [0, {n_vars-1}]"

    # Testa compilação da expressão
    try:
        # Cria variáveis dummy para teste
        test_vars = {f'x': [0.0] * n_vars, f'f': [0.0] * n_vars}
        test_code = f"lambda: {equation}"
        compile(test_code, '<string>', 'eval')
    except SyntaxError as e:
        return False, f"Erro de sintaxe: {str(e)}"

    return True, ""


@st.dialog('Custom p-Fuzzy definition')
def render_custom_p_fuzzy_definition(output_vars):
    """Render UI for custom p-Fuzzy definition"""

    st.markdown("<div style='border-bottom: 1px solid #e5e7eb; margin: 0.5rem 0 1.5rem 0;'></div>", unsafe_allow_html=True)
    st.markdown("#### Custom p-Fuzzy System")

    n_vars = len(output_vars)
    
    vars_aliases = ", ".join([f"$x_{i+1}=x[{i}]$" for i in range(n_vars)])
    output_aliases = ", ".join([f"fis_output[{i}]=$f[{i}]$" for i in range(n_vars)])
    st.caption(f"Define {n_vars} differential equation(s)")
    st.caption(f"Alias for state vars: "+vars_aliases)
    st.caption(f"Alias for FIS outputs: "+output_aliases)
    eqs = []
    for i in range(n_vars):
        equation = st.text_input(
            rf"$dx_{i+1}/dt$ =",
            key=f"custom_equation_{i}",
            placeholder=f"e.g., r*x[{i}]+f[{i}]**2",
            help="Use x[0], x[1], ... for state variables"
        )
        eqs.append(equation)
    if all(eqs):
        ode_config = {
                    "name": "Custom Continuous p-Fuzzy",
                    "dim": n_vars,
                    "vars": [ f"x_{i+1}" for i in range(n_vars)],
                    "equations": eqs,
                }
        # if 'custom_config' not in st.session_state:
        st.session_state['custom_config_continuous_p_fuzzy'] = ode_config
        st.rerun()


    
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

def build_p_fuzzy_function(equations):
    """Build continuous p-fuzzy function from string equations
    
    Parameters:
    -----------
    equations : list of str
        List of equations using 'x' for state and 'f' for fuzzy output
        
    Returns:
    --------
    function : callable
        Function with signature custom_pfuzzy(state, output_fis)
    """
    
    func_code = "def custom_pfuzzy(state, output_fis):\n"
    func_code += "    from numpy import sin, cos, exp, log, sqrt, abs\n"
    func_code += "    import numpy as np\n"
    func_code += "    x = state\n"
    func_code += "    f = output_fis\n"
    func_code += "    return np.array([\n"
    
    for eq in equations:
        func_code += f"        {eq},\n"
    func_code += "    ])\n"
    
    namespace = {}
    exec(func_code, namespace)
    return namespace['custom_pfuzzy']


def close_edit_dialog():
    st.session_state["edit_custom_equation"]=None

@st.dialog("Edit Custom Equation")
def edit_dialog():
    custom_config = st.session_state['custom_config_continuous_p_fuzzy']
    eqs = []
    i = 0
    for eq in custom_config['equations']:
        equation = st.text_input(
            rf"$dx_{i+1}/dt$ =",
            value = eq,
            key=f"custom_equation_edit_{i}",
            placeholder=f"e.g., r * x[{i}] * (1 - x[{i}] / K)",
            help="Use x[0], x[1], ... for state variables"
        )
        eqs.append(equation)
        i += 1
    if st.button('Save Changes',width = 'stretch'):
        if all(eqs):
            custom_config['equations'] = eqs
            # custom_config[vars] = extract_parameters(eqs)
            close_edit_dialog()
            st.rerun()


def render_pfuzzy_continuous_interface(selected_fis,mode,t_end,dt,method):
    """Render p-Fuzzy continuous interface with full implementation"""

    # Check if FIS is available (usa botão estilizado na página principal)
    if not check_fis_available(show_styled_button=True):
        return
       

    # Get selected FIS
    selected_fis = st.session_state.selected_fis_for_dynamics['fis']
    # # selected_fis = st.session_state.fis_list[fis_idx]

    # Validate FIS for p-Fuzzy
    input_vars = selected_fis['input_variables']
    output_vars = selected_fis['output_variables']

    if len(input_vars) == 0 or len(output_vars) == 0:
        st.error("❌ FIS must have at least one input and one output variable")
        return

    # Configuration section
    # with st.sidebar:
        

    # Map each input variable to a state variable
    state_vars = []
    for var in input_vars:
        state_vars.append(var['name'])
    
    if mode=='custom':
        ode_config = st.session_state.get('custom_config_continuous_p_fuzzy',None)
        if not ode_config is None:
            with st.expander(f"System Equations - {ode_config['name']}", expanded=True):
            
                code = ""
                for var, eq in zip(ode_config['vars'], ode_config['equations']):
                    code += f"d{var}/dt = {eq}\n"
                
                
                renames = ", ".join([f"x_{i+1}"  for i in range(len(ode_config['equations']))])
                real_vars = ", ".join([f"x[{i}]"  for i in range(len(ode_config['equations']))])
                st.code(f"{renames} = {real_vars}\n"+code, language="python")
                term_action = st.segmented_control(
                                                f"Actions",
                                                options=['Edit','Delete'],
                                                selection_mode="single",
                                                label_visibility="collapsed",
                                                key="edit_custom_equation",
                                            )
                dialog_opened = False
                if not dialog_opened and term_action:
                    if term_action=='Delete':
                        st.session_state.pop('custom_config_continuous_p_fuzzy',None)
                        st.rerun()
                    # dialog_opened = True
                    if term_action=='Edit':
                        edit_dialog()
                        dialog_opened = True
            
    
    # Initial conditions
    st.markdown("### Initial Conditions")

    # Initial condition inputs
    initial_conditions = {}
    cols = st.columns(min(len(state_vars), 3))

    for idx, var in enumerate(input_vars):
        with cols[idx % len(cols)]:
            initial_conditions[var['name']] = st.number_input(
                f"{var['name']}",
                min_value=float(var['min']),
                max_value=float(var['max']),
                value=float((var['min'] + var['max']) / 2),
                key=f"ic_{var['name']}"
            )
# Simulate button
    if st.button("▶️ Run Simulation", type="primary", width='stretch'):
        try:
            # Import p-fuzzy module
            # Build FIS using inference engine
            engine = InferenceEngine(selected_fis)

            # Create p-Fuzzy system
            if not mode=='custom':
                pfuzzy = PFuzzyContinuous(
                    fis=engine.system,
                    mode=mode,
                    state_vars=state_vars
                )
            else:
                custom_config = st.session_state.get('custom_config_continuous_p_fuzzy',None)
                pfuzzy_func = build_p_fuzzy_function(custom_config['equations'])
                pfuzzy = PFuzzyContinuous(
                    fis=engine.system,
                    dynamic_function=pfuzzy_func,
                    mode=mode,
                    state_vars=state_vars,
                    method=method
                )
                # return

            # Run simulation
            with st.spinner("Simulating..."):
                print(initial_conditions)
                time, trajectory = pfuzzy.simulate(x0=initial_conditions, t_span=(0,t_end),dt=dt,adaptive=True)

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

                st.plotly_chart(fig, width='stretch')

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
                            key="phase_x_continuous"
                        )
                    with col2:
                        # Default to second variable, or first if only one
                        default_y = 1 if n_vars > 1 and var_x_idx != 1 else (0 if var_x_idx != 0 else 1 if n_vars > 1 else 0)
                        var_y_idx = st.selectbox(
                            "Y-axis variable",
                            range(n_vars),
                            format_func=lambda x: state_vars[x],
                            index=default_y,
                            key="phase_y_continuous"
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

                    st.plotly_chart(fig_phase, width='stretch')

            # Export data
            with st.expander("💾 Export Data"):
                import pandas as pd

                # Create DataFrame
                data = {'time': time}
                for i, var_name in enumerate(state_vars):
                    data[var_name] = trajectory[:, i]

                df = pd.DataFrame(data)

                st.dataframe(df, width='stretch')

                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"{selected_fis['name']}_pfuzzy_continuous.csv",
                    mime="text/csv"
                )

            # Espaçamento final
            st.markdown("<br><br>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error during simulation: {str(e)}")
            import traceback
            with st.expander("🐛 Debug Info"):
                st.code(traceback.format_exc())

def run():
    """Render dynamic systems page"""

    # Verificar se há FIS disponível ANTES de renderizar sidebar
    has_fis = 'fis_list' in st.session_state and len(st.session_state.fis_list) > 0

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 0.25rem 0 0.125rem 0; margin-top: 0.5rem;">
            <h2 style="margin: 0.25rem 0 0.125rem 0; color: #667eea;">Continuous p-Fuzzy</h2>
            <p style="color: #6b7280; font-size: 0.9rem; margin: 0;">
                Model Continuous temporal evolution with FIS
            </p>
        </div>
        <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 0.25rem 0 0.5rem 0;">
        """, unsafe_allow_html=True)

        if has_fis:
            fis_options= [{'fis_index':i,'fis':fis} for i,fis in enumerate(st.session_state.fis_list)]
            selected_fis = st.selectbox('Select a FIS',fis_options,key="selected_fis_for_dynamics",format_func=lambda x: x['fis']['name'])
            st.markdown("#### Simulation Configuration")

            mode = st.selectbox(
                "Mode",
                ["absolute", "relative","custom"],
                help="Absolute: dx/dt = f(x)\n\nRelative: dx/dt = x*f(x)"
                )

            t_end = st.number_input(
                        "Simulation time",
                        min_value=1.0,
                        max_value=1000.0,
                        value=50.0,
                        step=1.0)

            dt = st.number_input(
                "Time step (dt)",
                min_value=0.001,
                max_value=1.0,
                value=0.1,
                step=0.01,
                format="%.3f")
            method = st.selectbox("Integration method", ["rk4", "euler"])
        else:
            # Sidebar usa botão simples (sem emoji)
            check_fis_available(show_styled_button=False)

    # PÁGINA PRINCIPAL - Sempre renderiza, mesmo sem FIS
    if not has_fis:
        # Mostra mensagem com botão estilizado na página principal
        check_fis_available(show_styled_button=True)
        return
       
    if mode == 'custom' and st.session_state.get('custom_config_continuous_p_fuzzy',None) is None:
        if st.button('Define Custom p-Fuzzy System',width='stretch'):
            output_vars = selected_fis['fis']['output_variables']
            render_custom_p_fuzzy_definition(output_vars)
        return
    render_pfuzzy_continuous_interface(selected_fis,mode,t_end,dt,method)
    
    




