"""
Fuzzy ODE Module
Specialized interface for Fuzzy Ordinary Differential Equations
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import traceback
from typing import Tuple, Dict, List, Any, Set, Callable


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

    # Testa compilação da expressão
    try:
        # Cria variáveis dummy para teste
        test_vars = {'x': [0.0] * n_vars}
        test_code = f"lambda: {equation}"
        compile(test_code, '<string>', 'eval')
    except SyntaxError as e:
        return False, f"Erro de sintaxe: {str(e)}"

    return True, ""


# ========== SIDEBAR ==========

def render_sidebar():
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
            key="selected_predefined_system"
        )

        st.markdown("<div style='border-bottom: 1px solid #e5e7eb; margin: 0.5rem 0 1.5rem 0;'></div>", unsafe_allow_html=True)

    else:  # Custom
        st.number_input(
            "Number of variables",
            min_value=1,
            max_value=5,
            value=st.session_state.get('n_custom_vars', 2),
            key="n_custom_vars"
        )

        st.caption("Define equations in main panel")


# ========== SYSTEM CONFIGURATIONS ==========

def get_predefined_ode_config(system_name):
    """Returns configuration for pre-defined ODE system"""

    systems = {
        "Logistic Growth": {
            "name": "Logistic Growth",
            "dim": 1,
            "vars": ["x"],
            "equations": ["r * x[0] * (1 - x[0] / K)"],
            "params": ["r", "K"],
            "default_params": {"r": 0.5, "K": 100.0},
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
            "default_params": {"beta": 0.5, "gamma": 0.1, "N": 1000.0},
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

def close_edit_dialog():
    st.session_state["edit_custom_equation"]=None

@st.dialog("Edit Custom Equation")
def edit_dialog():
    custom_config = st.session_state['custom_config']
    n_vars = custom_config['dim']
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
    if st.button('Save Changes', width='stretch'):
        if all(eqs):
            # Validar todas as equações antes de salvar
            all_valid = True
            for i, eq in enumerate(eqs):
                valid, msg = validate_equation(eq, n_vars)
                if not valid:
                    st.error(f"❌ Equation {i+1} invalid: {msg}")
                    all_valid = False

            if all_valid:
                custom_config['equations'] = eqs
                custom_config['params'] = extract_parameters(eqs)
                close_edit_dialog()
                st.rerun()


def get_custom_ode_config():
    """Returns configuration for custom ODE system"""

    n_vars = st.session_state.get('n_custom_vars', 2)
    var_names = [st.session_state.get(f"custom_var_name_{i}", f"x_{i+1}") for i in range(n_vars)]
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


# ========== CUSTOM ODE DEFINITION UI ==========

def render_custom_ode_definition():
    """Render UI for custom ODE definition"""

    st.markdown("<div style='border-bottom: 1px solid #e5e7eb; margin: 0.5rem 0 1.5rem 0;'></div>", unsafe_allow_html=True)
    st.markdown("#### Custom ODE System")

    n_vars = st.session_state.get('n_custom_vars', 2)
    
    renames = ", ".join([f"$x_{i+1}=x[{i}]$" for i in range(n_vars)])
    st.caption(f"Define {n_vars} differential equation(s): "+renames)
    eqs = []
    for i in range(n_vars):
        equation = st.text_input(
            rf"$dx_{i+1}/dt$ =",
            key=f"custom_equation_{i}",
            placeholder=f"e.g., r * x[{i}] * (1 - x[{i}] / K)",
            help="Use x[0], x[1], ... for state variables"
        )
        eqs.append(equation)

    if all(eqs):
        # Validar todas as equações antes de salvar
        all_valid = True
        for i, eq in enumerate(eqs):
            valid, msg = validate_equation(eq, n_vars)
            if not valid:
                st.error(f"❌ Equation {i+1} invalid: {msg}")
                all_valid = False

        if all_valid:
            params = extract_parameters(eqs)
            ode_config = {
                        "name": "Custom ODE System",
                        "dim": n_vars,
                        "vars": [ f"x_{i+1}" for i in range(n_vars)],
                        "equations": eqs,
                        "params": params,
                        "default_params": {},
                        "default_ic": [1.0] * n_vars,
                        "ic_ranges": [(-10.0, 10.0)] * n_vars
                    }
            st.session_state['custom_config'] = ode_config
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


# ========== HELPER FUNCTIONS ==========

def extract_parameters(equations):
    """Extract parameter names from equations"""
    all_params = set()
    for eq in equations:
        tokens = re.findall(r'\b[a-zA-Z_]\w*\b', eq)
        for token in tokens:
            if token not in ['x', 'sin', 'cos', 'exp', 'log', 'sqrt', 'abs', 't']:
                all_params.add(token)
    return sorted(all_params)


def build_fuzzy_number(config):
    """Build FuzzyNumber from config dict"""
    from fuzzy_systems.dynamics.fuzzy_ode import FuzzyNumber

    if config['type'] == 'triangular':
        center = config['b']
        spread = (config['c'] - config['a']) / 2
        return FuzzyNumber.triangular(center=center, spread=spread, name="fuzzy")

    elif config['type'] == 'gaussian':
        return FuzzyNumber.gaussian(mean=config['mean'], sigma=config['sigma'], name="fuzzy")

    else:  # trapezoidal
        return FuzzyNumber.trapezoidal(
            a=config['a'],
            b=config['b'],
            c=config['c'],
            d=config['d'],
            name="fuzzy"
        )


def build_ode_function(equations: List[str], safe_names: Set[str]) -> Callable:
    """Build ODE function from string equations (SAFE VERSION)

    Uses isolated namespace to prevent execution of arbitrary code.

    Args:
        equations: List of equation strings
        safe_names: Set of safe parameter names

    Returns:
        Callable ODE function
    """
    # Namespace restrito com apenas funções seguras
    safe_namespace = {
        'sin': np.sin,
        'cos': np.cos,
        'exp': np.exp,
        'log': np.log,
        'sqrt': np.sqrt,
        'abs': np.abs,
        'tan': np.tan,
        'arcsin': np.arcsin,
        'arccos': np.arccos,
        'arctan': np.arctan,
        'sinh': np.sinh,
        'cosh': np.cosh,
        'tanh': np.tanh,
        'np': np,
        '__builtins__': {}  # Bloqueia acesso a funções built-in perigosas
    }

    func_code = "def ode_system(t, x"
    param_names = sorted(safe_names - {'t', 'x'})

    if param_names:
        func_code += ", " + ", ".join(param_names)
    func_code += "):\n"
    func_code += "    return np.array([\n"

    for eq in equations:
        func_code += f"        {eq},\n"
    func_code += "    ])\n"

    # Executar em namespace isolado
    exec(func_code, safe_namespace)
    return safe_namespace['ode_system']


# ========== MAIN CONFIGURATION UI ==========

def render_configuration_and_solve(ode_config):
    """Render configuration UI and solve"""
    with st.sidebar:
        st.markdown("#### Solver Configuration")

        t_end = st.number_input("Simulation time", min_value=1.0, max_value=1000.0, value=10.0, step=5.0)

        n_alpha = st.number_input("α-levels", min_value=3, max_value=21, value=11, step=2)

        ode_method = st.selectbox(
                "ODE Method",
                ["RK45", "RK23", "DOP853", "Radau", "BDF"],
                help="Integration method for ODEs"
            )

        fuzzy_method = st.selectbox(
                "Fuzzy Method",
                ["standard", "monte_carlo"],
                help="Method for fuzzy propagation"
            )
        if fuzzy_method=='monte_carlo':
            num_of_points_mc = st.number_input("No of samples", min_value=100, max_value=10000, value=1000, step=100)
        else:
            num_of_points = st.number_input("Grid points", min_value=2, max_value=100, value=10, step=1)

    # Extract parameters
    all_params = extract_parameters(ode_config['equations'])

    # Configuration
    

    # Initialize session state
    if 'fuzzy_ics_config' not in st.session_state:
        st.session_state.fuzzy_ics_config = {}
    if 'fuzzy_params_config' not in st.session_state:
        st.session_state.fuzzy_params_config = {}

    # Initial Conditions
    # st.caption("Check 'Fuzzy' to define fuzzy initial conditions")

    initial_conditions = []


    for i in range(ode_config['dim']):
        var_name = ode_config['vars'][i]
        default_val = ode_config['default_ic'][i]
        min_val, max_val = ode_config['ic_ranges'][i]

        with st.expander(f'Initial condition - ${var_name}$',expanded = True):
        
            ic_cols = st.columns([1/3,2/3])
            # Checkbox for fuzzy
            is_fuzzy = ic_cols[0].checkbox(
                    f"Is fuzzy?",
                    value=var_name in st.session_state.fuzzy_ics_config,
                    key=f"fuzzy_ic_checkbox_{var_name}"
                )
            
            if is_fuzzy:
                mf_type = ic_cols[0].selectbox(
                                    "Membership Function Type",
                                    ["Triangular", "Gaussian", "Trapezoidal"],
                                    help="Shape of the fuzzy number",
                                    key=f"fuzzy_ci_selectbox_{var_name}"
                                )
                if mf_type=='Triangular':
                    a = ic_cols[0].number_input("Value for $a$", value=float(default_val * 0.8), format="%.4f", step=0.1,key=f"fuzzy_ci_tri_a_{var_name}")
                    b = ic_cols[0].number_input("Value for $b$", value=float(default_val *1), format="%.4f", step=0.1,key=f"fuzzy_ci_tri_b_{var_name}")
                    c = ic_cols[0].number_input("Value for $c$", value=float(default_val * 1.2), format="%.4f", step=0.1, key=f"fuzzy_ci_tri_c_{var_name}")
                    config = {'type': 'triangular', 'a': a, 'b': b, 'c': c}
                    
                    x = np.linspace(a-0.2*np.abs(a),c+0.2*np.abs(c),200)
                    y = np.minimum(np.maximum(0, (x - a) / (b - a)), np.maximum(0, (c - x) / (c - b)))


                elif mf_type=='Gaussian':
                    mean = ic_cols[0].number_input("Mean $\mu$", value=float(default_val), format="%.4f", step=0.1,key=f"fuzzy_ci_gauss_mu_{var_name}")
                    sigma = ic_cols[0].number_input("Std Dev $\sigma$", value=float(abs(default_val) * 0.1), format="%.4f", step=0.01,key=f"fuzzy_ci_gauss_sigma_{var_name}")
                    config = {'type': 'gaussian', 'mean': mean, 'sigma': sigma}
                    x = np.linspace(mean-5*sigma,mean+5*sigma,500)
                    y = np.exp(-0.5 * ((x - mean) /sigma)**2)
    
                else:
                    a = ic_cols[0].number_input("Value for $a$", value=float(default_val * 0.7), format="%.4f", step=0.1,key=f"fuzzy_ci_trap_a_{var_name}")
                    b = ic_cols[0].number_input("Value for $b$", value=float(default_val *0.9), format="%.4f", step=0.1,key=f"fuzzy_ci_trap_b_{var_name}")
                    c = ic_cols[0].number_input("Value for $c$", value=float(default_val * 1.1), format="%.4f", step=0.1,key=f"fuzzy_ci_trap_c_{var_name}")
                    d= ic_cols[0].number_input("Value for $d$", value=float(default_val * 1.3), format="%.4f", step=0.1,key=f"fuzzy_ci_trap_d_{var_name}")
                    config = {'type': 'trapezoidal', 'a': a, 'b': b, 'c': c,'d': d}

                    x = np.linspace(a-0.2*np.abs(a),d+0.2*np.abs(d),200)
                    y = np.minimum(np.maximum(0, (x - a) / (b - a)), np.minimum(1, np.maximum(0, (d - x) / (d - c)) if d > c else 1))

                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                                        x=x, y=y,
                                        mode='lines',
                                        name=var_name,
                                        hovertemplate=f"{var_name}<br>x=%{{x:.2f}}<br>μ=%{{y:.3f}}<extra></extra>"
                                    ))
                fig.update_layout(
                                title=rf"Triangular Membership Function",
                                xaxis_title=var_name,
                                yaxis_title="Membership Degree (μ)",
                                hovermode='closest',
                                height=400
                            )

                ic_cols[1].plotly_chart(fig, use_container_width=True,key=f"fuzzy_ci_fig_{var_name}")    

                initial_conditions.append(build_fuzzy_number(config))
            else:
                # Crisp value
                ic_cols[0].caption('Check above for fuzzy uncertainty')
                ic_value = ic_cols[1].number_input(
                    f"Value",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default_val),
                    key=f"ic_{var_name}",
                    step=0.1
                )
                initial_conditions.append(ic_value)


    # Parameters
    if not all_params:
        st.info("ℹ️ No parameters detected in equations")
    else:
        fuzzy_params = {}
        crisp_params = {}

        for idx, param_name in enumerate(all_params):
            default_val = ode_config['default_params'].get(param_name, 1.0)

            with st.expander(f'Parameter - ${param_name}$',expanded = True):
                param_cols = st.columns([1/3,2/3])
                # Checkbox for fuzzy
                is_fuzzy = param_cols[0].checkbox(
                        f"Is fuzzy?",
                        value=param_name in st.session_state.fuzzy_params_config,
                        key=f"fuzzy_param_checkbox_{param_name}"
                    )
                
                if is_fuzzy:
                    mf_type = param_cols[0].selectbox(
                                        "Membership Function Type",
                                        ["Triangular", "Gaussian", "Trapezoidal"],
                                        help="Shape of the fuzzy number",key = f"fuzzy_param_select_{param_name}")
                    if mf_type=='Triangular':
                        a = param_cols[0].number_input("Value for $a$", value=float(default_val * 0.8), format="%.4f", step=0.1)
                        b = param_cols[0].number_input("Value for $b$", value=float(default_val *1), format="%.4f", step=0.1)
                        c = param_cols[0].number_input("Value for $c$", value=float(default_val * 1.2), format="%.4f", step=0.1)
                        config = {'type': 'triangular', 'a': a, 'b': b, 'c': c}
                        
                        x = np.linspace(a-0.2*np.abs(a),c+0.2*np.abs(c),200)
                        y = np.minimum(np.maximum(0, (x - a) / (b - a)), np.maximum(0, (c - x) / (c - b)))


                    elif mf_type=='Gaussian':
                        mean = param_cols[0].number_input("Mean $\mu$", value=float(default_val), format="%.4f", step=0.1)
                        sigma = param_cols[0].number_input("Std Dev $\sigma$", value=float(abs(default_val) * 0.1), format="%.4f", step=0.01)
                        config = {'type': 'gaussian', 'mean': mean, 'sigma': sigma}
                        x = np.linspace(mean-5*sigma,mean+5*sigma,500)
                        y = np.exp(-0.5 * ((x - mean) /sigma)**2)
        
                    else:
                        a = param_cols[0].number_input("Value for $a$", value=float(default_val * 0.7), format="%.4f", step=0.1)
                        b = param_cols[0].number_input("Value for $b$", value=float(default_val *0.9), format="%.4f", step=0.1)
                        c = param_cols[0].number_input("Value for $c$", value=float(default_val * 1.1), format="%.4f", step=0.1)
                        d= param_cols[0].number_input("Value for $d$", value=float(default_val * 1.3), format="%.4f", step=0.1)
                        config = {'type': 'trapezoidal', 'a': a, 'b': b, 'c': c,'d': d}

                        x = np.linspace(a-0.2*np.abs(a),d+0.2*np.abs(d),200)
                        y = np.minimum(np.maximum(0, (x - a) / (b - a)), np.minimum(1, np.maximum(0, (d - x) / (d - c)) if d > c else 1))

                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                                            x=x, y=y,
                                            mode='lines',
                                            name=param_name,
                                            hovertemplate=f"{param_name}<br>x=%{{x:.2f}}<br>μ=%{{y:.3f}}<extra></extra>"
                                        ))
                    fig.update_layout(
                                    title=rf"Triangular Membership Function",
                                    xaxis_title=param_name,
                                    yaxis_title="Membership Degree (μ)",
                                    hovermode='closest',
                                    height=400
                                )

                    param_cols[1].plotly_chart(fig, use_container_width=True,key=f"fuzzy_param_fig_{param_name}")    

                    fuzzy_params[param_name] = build_fuzzy_number(config)
                else:
                    # Crisp value
                    param_cols[0].caption('Check above for fuzzy uncertainty')
                    # Crisp value
                    value = param_cols[1].number_input(
                        f"Value",
                        value=float(default_val),
                        key=f"crisp_param_{param_name}",
                        step=0.01,
                        format="%.4f"
                    )
                    crisp_params[param_name] = value

        all_params_dict = {**fuzzy_params, **crisp_params}

    # Detectar se há algum parâmetro ou IC fuzzy
    has_fuzzy_params = len(fuzzy_params) > 0
    has_fuzzy_ics = len([ic for ic in initial_conditions if hasattr(ic, 'alpha_cut')]) > 0
    is_fuzzy_problem = has_fuzzy_params or has_fuzzy_ics


    # Solve button
    st.markdown("<br>", unsafe_allow_html=True)

    button_label = "Solve Fuzzy ODE" if is_fuzzy_problem else "Solve ODE"
    if st.button(button_label, type="primary", use_container_width=True):
        # Validate fuzzy ICs
        for var_name in st.session_state.fuzzy_ics_config.keys():
            if f"fuzzy_ic_{var_name}" in st.session_state and st.session_state[f"fuzzy_ic_{var_name}"]:
                if var_name not in st.session_state.fuzzy_ics_config:
                    st.error(f"❌ Please configure fuzzy IC for {var_name}")
                    return

        # Validate fuzzy params
        for param_name in st.session_state.fuzzy_params_config.keys():
            if f"fuzzy_param_{param_name}" in st.session_state and st.session_state[f"fuzzy_param_{param_name}"]:
                if param_name not in st.session_state.fuzzy_params_config:
                    st.error(f"❌ Please configure fuzzy parameter {param_name}")
                    return

        try:
            # Build ODE function
            ode_func = build_ode_function(ode_config['equations'], set(all_params).union({'t'}))

            if not is_fuzzy_problem:
                # SOLUÇÃO NORMAL (SEM FUZZY)
                from scipy.integrate import solve_ivp

                # Converter ICs para valores numéricos
                crisp_ics = [float(ic) if not hasattr(ic, 'alpha_cut') else float(ic.center)
                            for ic in initial_conditions]

                with st.spinner("Solving ODE..."):
                    # Resolver EDO normal com scipy
                    if all_params:
                        # Com parâmetros
                        sol = solve_ivp(
                            lambda t, y: ode_func(t, y, **all_params_dict),
                            (0.0, t_end),
                            crisp_ics,
                            method='RK45' if ode_method == 'rk4' else 'RK23',
                            dense_output=True,
                            max_step=0.1
                        )
                    else:
                        # Sem parâmetros
                        sol = solve_ivp(
                            ode_func,
                            (0.0, t_end),
                            crisp_ics,
                            method='RK45' if ode_method == 'rk4' else 'RK23',
                            dense_output=True,
                            max_step=0.1
                        )

                    # Criar objeto de solução compatível
                    class CrispSolution:
                        def __init__(self, t, y, var_names):
                            self.t = t
                            self.y = y
                            self.var_names = var_names
                            self.is_fuzzy = False

                        def to_dataframe(self, alpha=None):
                            import pandas as pd
                            data = {'time': self.t}
                            for i, var in enumerate(self.var_names):
                                data[var] = self.y[i, :]
                            return pd.DataFrame(data)

                    solution = CrispSolution(sol.t, sol.y, ode_config['vars'])

                # Store solution in session state
                st.session_state.ode_solution = solution
                st.session_state.ode_config = ode_config
                st.toast("✅ ODE solved successfully! (Crisp/Deterministic solution)")

            else:
                # SOLUÇÃO FUZZY (ORIGINAL)
                from fuzzy_systems.dynamics.fuzzy_ode import FuzzyODESolver

                with st.spinner("Solving Fuzzy ODE..."):
                    solver = FuzzyODESolver(
                        ode_func=ode_func,
                        t_span=(0.0, t_end),
                        initial_condition=initial_conditions,
                        params=all_params_dict if all_params else None,
                        n_alpha_cuts=n_alpha,
                        method=ode_method,  # ODE integration method
                        var_names=ode_config['vars']
                    )

                    # Solve with fuzzy method
                    if fuzzy_method == 'monte_carlo':
                        try:
                            solution = solver.solve(method='monte_carlo', n_samples=num_of_points_mc, verbose=False)
                        except:
                            solution = solver.solve(method='monte_carlo', n_samples=1000, verbose=False)
                    elif fuzzy_method == 'hierarchical':
                        try:
                            solution = solver.solve(method='hierarchical', verbose=False)
                        except:
                            solution = solver.solve(method='hierarchical', verbose=False)
                    else:  # standard
                        try:
                            solution = solver.solve(method='standard', n_grid_points=5, verbose=False)
                        except:
                            solution = solver.solve(method='standard', n_grid_points=num_of_points, verbose=False)

                # Store solution in session state
                st.session_state.ode_solution = solution
                st.session_state.ode_config = ode_config
                st.toast("✅ Fuzzy ODE solved successfully!")

        except ValueError as e:
            st.error(f"❌ Invalid value: {str(e)}")
            st.info("💡 Check if initial conditions and parameters are valid")
        except KeyError as e:
            st.error(f"❌ Variable not found: {str(e)}")
            st.info("💡 Verify that all parameters are properly defined")
        except ZeroDivisionError:
            st.error(f"❌ Division by zero in equations")
            st.info("💡 Check your equations for potential division by zero")
        except Exception as e:
            st.error(f"❌ Unexpected error during simulation")
            with st.expander("🐛 Technical details"):
                st.code(traceback.format_exc())
                st.caption("Please report this error with the details above")

    # Check if solver configuration changed (should reset solution)
    current_solver_config = {
        't_end': t_end,
        'n_alpha': n_alpha,
        'ode_method': ode_method,
        'fuzzy_method': fuzzy_method,
        'num_of_points_mc': num_of_points_mc if fuzzy_method == 'monte_carlo' else None,
        'num_of_points': num_of_points if fuzzy_method != 'monte_carlo' else None,
        'initial_conditions': str(initial_conditions),  # Convert to string for comparison
        'params': str(all_params_dict if all_params else {})
    }

    # Compare with previous config
    if 'prev_solver_config' not in st.session_state:
        st.session_state.prev_solver_config = current_solver_config
    elif st.session_state.prev_solver_config != current_solver_config:
        # Configuration changed, reset solution
        if 'ode_solution' in st.session_state:
            st.session_state.ode_solution = None
            st.session_state.ode_config = None
        st.session_state.prev_solver_config = current_solver_config

    # Display results if solution exists
    if 'ode_solution' in st.session_state and st.session_state.ode_solution is not None:
        render_results(st.session_state.ode_solution, st.session_state.ode_config)




# ========== RESULTS VISUALIZATION ==========

def render_results(solution, ode_config):
    """Render ODE results (Fuzzy or Crisp)"""

    n_vars = len(ode_config['vars'])
    is_crisp = hasattr(solution, 'is_fuzzy') and solution.is_fuzzy == False

    # Time evolution
    with st.expander("Time Evolution", expanded=True):
        if is_crisp:
            # SOLUÇÃO CRISP (NÃO FUZZY)
            st.info("ℹ️ Displaying deterministic (crisp) solution - No fuzzy parameters were selected")

            for var_idx in range(n_vars):
                fig = go.Figure()

                # Plot single trajectory
                fig.add_trace(
                    go.Scatter(
                        x=solution.t,
                        y=solution.y[var_idx],
                        mode='lines',
                        name=ode_config['vars'][var_idx],
                        line=dict(color='#667eea', width=2),
                        hovertemplate=f't=%{{x:.2f}}<br>{ode_config["vars"][var_idx]}=%{{y:.4f}}<extra></extra>'
                    )
                )

                fig.update_xaxes(title_text="Time")
                fig.update_yaxes(title_text=ode_config['vars'][var_idx])
                fig.update_layout(
                    height=500,
                    showlegend=True,
                    title=f"Solution - {ode_config['vars'][var_idx]}",
                    title_font_size=16,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font_size=12
                )

                st.plotly_chart(fig, use_container_width=True)

        else:
            # SOLUÇÃO FUZZY (ORIGINAL)
            # Get figure settings
            fill_color = st.session_state.get('ode_fig_fill_color', "#01050D")
            alpha_opacity = st.session_state.get('ode_fig_alpha_opacity', 0.4)
            alpha_skip = st.session_state.get('ode_fig_alpha_skip', 5)

            # Convert hex color to RGB
            fill_color_rgb = tuple(int(fill_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            if 'ode_fig_title' not in st.session_state:
                st.session_state['ode_fig_title'] = f"Fuzzy Solution - {ode_config['name']}"

            for var_idx in range(n_vars):
                fig = go.Figure()
                for alpha_idx, alpha in enumerate(solution.alphas):
                    y_min, y_max = solution.get_alpha_level(alpha)

                    # Calculate opacity based on alpha level and user setting
                    opacity = (0.1 + 0.9 * alpha) * alpha_opacity

                    # Lower bound
                    fig.add_trace(
                        go.Scatter(
                            x=solution.t,
                            y=y_min[var_idx],
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip'
                        )
                    )

                    # Upper bound with fill
                    show_in_legend =  alpha_idx % alpha_skip == 0
                    fig.add_trace(
                        go.Scatter(
                            x=solution.t,
                            y=y_max[var_idx],
                            mode='lines',
                            line=dict(width=0),
                            fill='tonexty',
                            fillcolor=f'rgba({fill_color_rgb[0]}, {fill_color_rgb[1]}, {fill_color_rgb[2]}, {opacity})',
                            name=f'α={alpha:.2f}' if show_in_legend else None,
                            showlegend=show_in_legend,
                            hovertemplate=f'α={alpha:.2f}<br>t=%{{x:.2f}}<br>%{{y:.4f}}'
                        )
                    )

                fig.update_xaxes(title_text="Time")
                fig.update_yaxes(title_text=ode_config['vars'][var_idx])

                # Apply figure layout with session state settings
                fig.update_layout(
                    height=st.session_state.get('ode_fig_height', 500),
                    showlegend=st.session_state.get('ode_fig_showlegend', True),
                    title=st.session_state.get('ode_fig_title', ''),
                    title_font_size=st.session_state.get('ode_fig_title_size', 16),
                    plot_bgcolor=st.session_state.get('ode_fig_bgcolor', 'white'),
                    paper_bgcolor=st.session_state.get('ode_fig_paper_bgcolor', 'white'),
                    font_size=st.session_state.get('ode_fig_font_size', 12)
                )

                st.plotly_chart(fig, use_container_width=True)

        with st.popover('Figure Options', width="stretch"):
            st.markdown("#### Figure Customization")

            # Layout options
            st.markdown("**Layout**")
            col1, col2 = st.columns(2)

            with col1:
                st.number_input(
                    "Height per subplot (px)",
                    min_value=200,
                    max_value=1000,
                    value=st.session_state.get('ode_fig_height', 500),
                    step=50,
                    key='ode_fig_height',
                    help="Height of each subplot"
                )

                st.checkbox(
                    "Show legend",
                    value=st.session_state.get('ode_fig_showlegend', True),
                    key='ode_fig_showlegend'
                )

            with col2:
                st.number_input(
                    "Font size",
                    min_value=8,
                    max_value=24,
                    value=st.session_state.get('ode_fig_font_size', 12),
                    step=1,
                    key='ode_fig_font_size'
                )

                st.number_input(
                    "Title font size",
                    min_value=10,
                    max_value=32,
                    value=st.session_state.get('ode_fig_title_size', 16),
                    step=1,
                    key='ode_fig_title_size'
                )

            st.markdown("---")

            # Title
            st.markdown("**Title**")
            st.text_input(
                "Figure title",
                value=st.session_state.get('ode_fig_title'),
                key='ode_fig_title',
                placeholder="Leave empty for no title"
            )

            st.markdown("---")

            # Colors
            st.markdown("**Colors**")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.color_picker(
                    "Fill color",
                    value=st.session_state.get('ode_fig_fill_color', "#01050D"),
                    key='ode_fig_fill_color',
                    help="Color for α-level fills"
                )

            with col2:
                st.color_picker(
                    "Plot background",
                    value=st.session_state.get('ode_fig_bgcolor', '#FFFFFF'),
                    key='ode_fig_bgcolor'
                )

            with col3:
                st.color_picker(
                    "Paper background",
                    value=st.session_state.get('ode_fig_paper_bgcolor', '#FFFFFF'),
                    key='ode_fig_paper_bgcolor'
                )

            st.markdown("---")

            # Alpha level display
            st.markdown("**Alpha Levels**")
            col1, col2 = st.columns(2)

            with col1:
                st.slider(
                    "Opacity multiplier",
                    min_value=0.1,
                    max_value=1.0,
                    value=st.session_state.get('ode_fig_alpha_opacity', 0.4),
                    step=0.05,
                    key='ode_fig_alpha_opacity',
                    help="Transparency of α-level fills"
                )

            with col2:
                st.number_input(
                    "Show every n-th α",
                    min_value=1,
                    max_value=10,
                    value=st.session_state.get('ode_fig_alpha_skip', 5),
                    step=1,
                    key='ode_fig_alpha_skip',
                    help="Display legend for every n-th alpha level"
                )

            st.markdown("---")

    # Fixed Time - APENAS PARA SOLUÇÃO FUZZY
    if not is_crisp:
        with st.expander("Fixed Time",expanded=True):
            times = solution.t

            time_selected = st.select_slider('Selected time',times,
                                            key=f"time_selector_for_{ode_config['name']}",
                                            format_func = lambda x:f"{x:,.2f}")
            time_idx = [i for i in range(len(times)) if times[i]==time_selected][0]

            st.session_state['ode_time_fig_title'] = f"Fuzzy Solution at t = {time_selected:.2f}"

            for var_idx in range(n_vars):
                x_left,y_left,x_right,y_right = [],[],[],[]
                for alpha_idx, alpha in enumerate(solution.alphas):
                    y_min, y_max = solution.get_alpha_level(alpha)
                    x_left.append(y_min[var_idx,time_idx])
                    x_right.append(y_max[var_idx,time_idx])
                    y_left.append(alpha)
                    y_right.append(alpha)
                x = x_left+x_right[::-1]
                y = y_left+y_right[::-1]
                difx = max(x)-min(x)
                fig = go.Figure()
                x = [min(x)-difx]+x+[max(x)+difx]
                y = [0]+y+[0]

                fill_color = st.session_state.get('ode_time_fig_fill_color', "#295BC9")
                fill_color = tuple(int(fill_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            mode='lines',
                            line=dict(color=f'rgba({fill_color[0]}, {fill_color[1]}, {fill_color[2]}, 1)',
                            width=st.session_state.get('ode_time_fig_line_width', 2)),
                        )
                    )

                fig.update_yaxes(title_text="Membership")
                fig.update_xaxes(title_text=ode_config['vars'][var_idx])

                # Apply figure layout with session state settings
                fig.update_layout(
                    height=400,
                    title=st.session_state.get('ode_time_fig_title',''),
                    title_font_size=st.session_state.get('ode_time_fig_title_size', 16),
                    plot_bgcolor=st.session_state.get('ode_time_fig_bgcolor', 'white'),
                    paper_bgcolor=st.session_state.get('ode_time_fig_paper_bgcolor', 'white'),
                    font_size=st.session_state.get('ode_time_fig_font_size', 12)
                )

                st.plotly_chart(fig, width="stretch")

            with st.popover('Figure Options', width="stretch"):
                st.markdown("#### Figure Customization")

                # Layout options
                st.markdown("**Layout**")
                col1, col2 = st.columns(2)

                with col1:
                    st.number_input(
                        "Height per subplot (px)",
                        min_value=200,
                        max_value=1000,
                        value=st.session_state.get('ode_time_fig_height', 400),
                        step=50,
                        key='ode_time_fig_height',
                        help="Height of each subplot"
                    )

                    st.checkbox(
                        "Show legend",
                        value=st.session_state.get('ode_time_fig_showlegend', True),
                        key='ode_time_fig_showlegend'
                    )

                with col2:
                    st.number_input(
                        "Font size",
                        min_value=8,
                        max_value=24,
                        value=st.session_state.get('ode_time_fig_font_size', 12),
                        step=1,
                        key='ode_time_fig_font_size'
                    )

                    st.number_input(
                        "Title font size",
                        min_value=10,
                        max_value=32,
                        value=st.session_state.get('ode_time_fig_title_size', 16),
                        step=1,
                        key='ode_time_fig_title_size'
                    )

                st.markdown("---")

                # Title
                st.markdown("**Title**")
                st.text_input(
                    "Figure title",
                    value=st.session_state.get('ode_time_fig_title',''),
                    key='ode_time_fig_title',
                    placeholder="Leave empty for no title"
                )

                st.markdown("---")

                # Colors
                st.markdown("**Colors**")
                col1, col2, col3, col4,= st.columns(4)

                with col1:
                    st.color_picker(
                        "Fill color",
                        value=st.session_state.get('ode_time_fig_fill_color', "#295BC9"),
                        key='ode_time_fig_fill_color',
                        help="Color for α-level fills"
                    )

                with col2:
                    st.color_picker(
                        "Plot background",
                        value=st.session_state.get('ode_time_fig_bgcolor', '#FFFFFF'),
                        key='ode_time_fig_bgcolor'
                    )

                with col3:
                    st.color_picker(
                        "Paper background",
                        value=st.session_state.get('ode_time_fig_paper_bgcolor', '#FFFFFF'),
                        key='ode_time_fig_paper_bgcolor'
                    )
                with col4:
                    st.number_input(
                        "Line Width",
                        min_value=1,
                        max_value=10,
                        value=st.session_state.get('ode_time_fig_line_width', 2),
                        step=1,
                        key='ode_time_fig_line_width'
                    )


                st.markdown("---")

    

    # Export data
    with st.expander("Export Data"):
        import pandas as pd

        if is_crisp:
            # Para solução crisp, exportar diretamente
            df = solution.to_dataframe()
            st.dataframe(df.head(20), use_container_width=True)

            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"ode_solution.csv",
                mime="text/csv"
            )
        else:
            # Para solução fuzzy, selecionar α-level
            alpha_export = st.selectbox("Select α-level to export", solution.alphas, index=len(solution.alphas)//2)
            df = solution.to_dataframe(alpha=alpha_export)

            st.dataframe(df.head(20), use_container_width=True)

            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"fuzzy_ode_alpha_{alpha_export:.2f}.csv",
                mime="text/csv"
            )

    # Espaçamento final
    st.markdown("<br><br>", unsafe_allow_html=True)


# ========== MAIN RENDER FUNCTION ==========

def run():
    """Main render function for Fuzzy ODE module"""

    # Initialize session state
    if 'ode_system_type' not in st.session_state:
        st.session_state.ode_system_type = "Pre-defined"
    if 'selected_predefined_system' not in st.session_state:
        st.session_state.selected_predefined_system = "Logistic Growth"
    if 'n_custom_vars' not in st.session_state:
        st.session_state.n_custom_vars = 2
    if 'fuzzy_params_config' not in st.session_state:
        st.session_state.fuzzy_params_config = {}
    if 'fuzzy_ics_config' not in st.session_state:
        st.session_state.fuzzy_ics_config = {}

    # Sidebar
    with st.sidebar:
        render_sidebar()

    # Get ODE configuration
    if st.session_state.ode_system_type == "Pre-defined":
        ode_config = get_predefined_ode_config(st.session_state.selected_predefined_system)
    else:
        # ode_config = get_custom_ode_config()
        ode_config = st.session_state.get('custom_config',None)
        if ode_config is None:
            render_custom_ode_definition()
            return

    st.markdown("<div style='border-bottom: 1px solid #e5e7eb; margin: 0.5rem 0 1.5rem 0;'></div>", unsafe_allow_html=True)

    # Show equations
    with st.expander(f"System Equations - {ode_config['name']}", expanded=True):
    
        code = ""
        for var, eq in zip(ode_config['vars'], ode_config['equations']):
            if st.session_state.ode_system_type == "Pre-defined":
                st.code(f"d{var}/dt = {eq}", language="python")
            else:
                code += f"d{var}/dt = {eq}\n"
        
        if st.session_state.ode_system_type == "Pre-defined":
            pass
        else:
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
                    st.session_state.pop('custom_config',None)
                    st.rerun()
                if term_action=='Edit':
                    edit_dialog()
                    dialog_opened = True

                

    # Configuration and solve
    render_configuration_and_solve(ode_config)

    st.markdown("<div style='border-bottom: 1px solid #e5e7eb; margin: 0.5rem 0 1.5rem 0;'></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)