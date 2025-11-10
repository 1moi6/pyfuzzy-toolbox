"""
Inference Systems Module
Interactive interface for creating Mamdani and Sugeno fuzzy systems
"""

import streamlit as st

import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Import inference engine
from modules.inference_engine import InferenceEngine

def close_dialog(variable_idx):
    """Callback to reset action selection"""
    st.session_state[f"actions_{variable_idx}"] = None

def rescale_term_params(old_min, old_max, new_min, new_max, params, mf_type):
    """Rescale term parameters when variable domain changes"""

    def rescale_value(val):
        """Rescale a single value from old range to new range"""
        # Normalize to [0, 1]
        normalized = (val - old_min) / (old_max - old_min) if old_max != old_min else 0.5
        # Scale to new range
        return new_min + normalized * (new_max - new_min)

    if mf_type == "triangular":
        # (a, b, c)
        return (rescale_value(params[0]), rescale_value(params[1]), rescale_value(params[2]))

    elif mf_type == "trapezoidal":
        # (a, b, c, d)
        return (rescale_value(params[0]), rescale_value(params[1]),
                rescale_value(params[2]), rescale_value(params[3]))

    elif mf_type == "gaussian":
        # (mean, std)
        new_mean = rescale_value(params[0])
        # Scale std proportionally to range change
        range_ratio = (new_max - new_min) / (old_max - old_min) if old_max != old_min else 1
        new_std = params[1] * range_ratio
        return (new_mean, new_std)

    elif mf_type == "sigmoid":
        # (a, c) - a is slope (invariant), c is center (rescale)
        return (params[0], rescale_value(params[1]))

    return params

def close_term_dialog(variable_idx, term_idx):
    """Callback to reset term action selection"""
    st.session_state[f"term_actions_{variable_idx}_{term_idx}"] = None

def close_output_dialog(variable_idx):
    """Callback to reset output action selection"""
    st.session_state[f"output_actions_{variable_idx}"] = None

def close_output_term_dialog(variable_idx, term_idx):
    """Callback to reset output term action selection"""
    st.session_state[f"output_term_actions_{variable_idx}_{term_idx}"] = None

# ========== OUTPUT VARIABLE DIALOGS ==========

@st.dialog("Edit Output Variable")
def edit_output_variable_dialog(variable_idx):
    """Dialog for editing an output variable"""
    variable = st.session_state.output_variables[variable_idx]

    st.markdown(f"**Editing Output Variable**")

    new_name = st.text_input("Variable Name", value=variable['name'])
    col1, col2 = st.columns(2)
    with col1:
        new_min = st.number_input("Min", value=float(variable['min']))
    with col2:
        new_max = st.number_input("Max", value=float(variable['max']))

    domain_changed = (new_min != variable['min'] or new_max != variable['max'])
    if domain_changed and variable['terms']:
        st.warning(f"⚠️ Changing the domain will automatically rescale all {len(variable['terms'])} term(s) parameters.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✓ Save Changes", use_container_width=True, type="primary"):
            if new_name and (new_name == variable['name'] or new_name not in [v['name'] for v in st.session_state.output_variables]):
                st.session_state.output_variables[variable_idx]['name'] = new_name

                if domain_changed:
                    old_min, old_max = variable['min'], variable['max']
                    for term_idx, term in enumerate(st.session_state.output_variables[variable_idx]['terms']):
                        new_params = rescale_term_params(
                            old_min, old_max, new_min, new_max,
                            term['params'], term['mf_type']
                        )
                        st.session_state.output_variables[variable_idx]['terms'][term_idx]['params'] = new_params

                st.session_state.output_variables[variable_idx]['min'] = new_min
                st.session_state.output_variables[variable_idx]['max'] = new_max
                close_output_dialog(variable_idx)
                st.rerun()
            elif not new_name:
                st.error("Please enter a variable name")
            else:
                st.error("Variable name already exists")
    with col2:
        if st.button("Cancel", use_container_width=True, on_click=close_output_dialog, args=(variable_idx,)):
            pass

@st.dialog("View Output Variable Details")
def view_output_variable_dialog(variable_idx):
    """Dialog for viewing output variable details"""
    variable = st.session_state.output_variables[variable_idx]

    st.markdown(f"### {variable['name']}")
    st.markdown(f"**Range:** [{variable['min']}, {variable['max']}]")
    st.markdown(f"**Number of Terms:** {len(variable['terms'])}")

    if variable['terms']:
        st.markdown("---")
        st.markdown("**Fuzzy Terms:**")
        for term in variable['terms']:
            with st.container():
                st.markdown(f"**{term['name']}**")
                st.caption(f"Type: {term['mf_type']} | Parameters: {term['params']}")
    else:
        st.info("No terms defined yet")

    if st.button("Close", use_container_width=True):
        close_output_dialog(variable_idx)
        st.rerun()

@st.dialog("Delete Output Variable")
def delete_output_variable_dialog(variable_idx):
    """Dialog for confirming output variable deletion"""
    variable = st.session_state.output_variables[variable_idx]

    st.warning(f"Are you sure you want to delete variable **'{variable['name']}'**?")
    if variable['terms']:
        st.error(f"This will also delete {len(variable['terms'])} term(s)!")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Yes, Delete", use_container_width=True, type="primary"):
            st.session_state.output_variables.pop(variable_idx)
            close_output_dialog(variable_idx)
            st.rerun()
    with col2:
        if st.button("Cancel", use_container_width=True, on_click=close_output_dialog, args=(variable_idx,)):
            pass

@st.dialog("Add Output Term")
def add_output_term_dialog(variable_idx, variable):
    """Dialog for adding a new fuzzy term to output variable"""

    st.markdown(f"**Adding term to variable: `{variable['name']}`**")
    st.caption(f"Range: [{variable['min']}, {variable['max']}]")

    term_name = st.text_input("Term Name", placeholder="e.g., slow, medium, fast")

    mf_icons = {
        "triangular": "△",
        "trapezoidal": "⬠",
        "gaussian": "⌢",
        "sigmoid": "∫"
    }

    mf_type = st.segmented_control(
        "Membership Function Type",
        options=["triangular", "trapezoidal", "gaussian", "sigmoid"],
        format_func=lambda x: f"{mf_icons[x]} {x.title()}",
        default="triangular",
        selection_mode="single"
    )

    st.markdown("---")
    st.markdown("**Parameters**")

    if mf_type == "triangular":
        st.caption("Triangular function: three points (a, b, c)")
        col1, col2, col3 = st.columns(3)
        with col1:
            p1 = st.number_input("a (left)", value=variable['min'])
        with col2:
            p2 = st.number_input("b (peak)", value=(variable['min'] + variable['max'])/2)
        with col3:
            p3 = st.number_input("c (right)", value=variable['max'])
        params = (p1, p2, p3)

    elif mf_type == "trapezoidal":
        st.caption("Trapezoidal function: four points (a, b, c, d)")
        col1, col2 = st.columns(2)
        with col1:
            p1 = st.number_input("a (left)", value=variable['min'])
            p2 = st.number_input("b (left peak)", value=variable['min'] + (variable['max']-variable['min'])*0.25)
        with col2:
            p3 = st.number_input("c (right peak)", value=variable['min'] + (variable['max']-variable['min'])*0.75)
            p4 = st.number_input("d (right)", value=variable['max'])
        params = (p1, p2, p3, p4)

    elif mf_type == "gaussian":
        st.caption("Gaussian function: mean and standard deviation")
        col1, col2 = st.columns(2)
        with col1:
            p1 = st.number_input("μ (mean)", value=(variable['min'] + variable['max'])/2)
        with col2:
            p2 = st.number_input("σ (std dev)", value=(variable['max']-variable['min'])/6)
        params = (p1, p2)

    else:  # sigmoid
        st.caption("Sigmoid function: slope and center")
        col1, col2 = st.columns(2)
        with col1:
            p1 = st.number_input("a (slope)", value=1.0)
        with col2:
            p2 = st.number_input("c (center)", value=(variable['min'] + variable['max'])/2)
        params = (p1, p2)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✓ Add Term", use_container_width=True, type="primary"):
            if term_name and term_name not in [t['name'] for t in variable['terms']]:
                st.session_state.output_variables[variable_idx]['terms'].append({
                    'name': term_name,
                    'mf_type': mf_type,
                    'params': params
                })
                close_output_dialog(variable_idx)
                st.rerun()
            elif not term_name:
                st.error("Please enter a term name")
            else:
                st.error("Term name already exists")

    with col2:
        if st.button("Cancel", use_container_width=True, on_click=close_output_dialog, args=(variable_idx,)):
            pass

@st.dialog("Edit Output Term")
def edit_output_term_dialog(variable_idx, term_idx):
    """Dialog for editing an output fuzzy term"""
    variable = st.session_state.output_variables[variable_idx]
    term = variable['terms'][term_idx]

    st.markdown(f"**Editing term in variable: `{variable['name']}`**")
    st.caption(f"Range: [{variable['min']}, {variable['max']}]")

    new_term_name = st.text_input("Term Name", value=term['name'])

    mf_icons = {
        "triangular": "△",
        "trapezoidal": "⬠",
        "gaussian": "⌢",
        "sigmoid": "∫"
    }

    mf_type = st.segmented_control(
        "Membership Function Type",
        options=["triangular", "trapezoidal", "gaussian", "sigmoid"],
        format_func=lambda x: f"{mf_icons[x]} {x.title()}",
        default=term['mf_type'],
        selection_mode="single"
    )

    st.markdown("---")
    st.markdown("**Parameters**")

    current_params = term['params'] if term['mf_type'] == mf_type else None

    if mf_type == "triangular":
        st.caption("Triangular function: three points (a, b, c)")
        col1, col2, col3 = st.columns(3)
        with col1:
            p1 = st.number_input("a (left)", value=float(current_params[0]) if current_params else variable['min'])
        with col2:
            p2 = st.number_input("b (peak)", value=float(current_params[1]) if current_params else (variable['min'] + variable['max'])/2)
        with col3:
            p3 = st.number_input("c (right)", value=float(current_params[2]) if current_params else variable['max'])
        params = (p1, p2, p3)

    elif mf_type == "trapezoidal":
        st.caption("Trapezoidal function: four points (a, b, c, d)")
        col1, col2 = st.columns(2)
        with col1:
            p1 = st.number_input("a (left)", value=float(current_params[0]) if current_params else variable['min'])
            p2 = st.number_input("b (left peak)", value=float(current_params[1]) if current_params else variable['min'] + (variable['max']-variable['min'])*0.25)
        with col2:
            p3 = st.number_input("c (right peak)", value=float(current_params[2]) if current_params else variable['min'] + (variable['max']-variable['min'])*0.75)
            p4 = st.number_input("d (right)", value=float(current_params[3]) if current_params else variable['max'])
        params = (p1, p2, p3, p4)

    elif mf_type == "gaussian":
        st.caption("Gaussian function: mean and standard deviation")
        col1, col2 = st.columns(2)
        with col1:
            p1 = st.number_input("μ (mean)", value=float(current_params[0]) if current_params else (variable['min'] + variable['max'])/2)
        with col2:
            p2 = st.number_input("σ (std dev)", value=float(current_params[1]) if current_params else (variable['max']-variable['min'])/6)
        params = (p1, p2)

    else:  # sigmoid
        st.caption("Sigmoid function: slope and center")
        col1, col2 = st.columns(2)
        with col1:
            p1 = st.number_input("a (slope)", value=float(current_params[0]) if current_params else 1.0)
        with col2:
            p2 = st.number_input("c (center)", value=float(current_params[1]) if current_params else (variable['min'] + variable['max'])/2)
        params = (p1, p2)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✓ Save Changes", use_container_width=True, type="primary"):
            other_terms = [t['name'] for i, t in enumerate(variable['terms']) if i != term_idx]
            if new_term_name and new_term_name not in other_terms:
                st.session_state.output_variables[variable_idx]['terms'][term_idx] = {
                    'name': new_term_name,
                    'mf_type': mf_type,
                    'params': params
                }
                close_output_term_dialog(variable_idx, term_idx)
                st.rerun()
            elif not new_term_name:
                st.error("Please enter a term name")
            else:
                st.error("Term name already exists")
    with col2:
        if st.button("Cancel", use_container_width=True):
            close_output_term_dialog(variable_idx, term_idx)
            st.rerun()

@st.dialog("Delete Output Term")
def delete_output_term_dialog(variable_idx, term_idx):
    """Dialog for confirming output term deletion"""
    variable = st.session_state.output_variables[variable_idx]
    term = variable['terms'][term_idx]

    st.warning(f"Are you sure you want to delete term **'{term['name']}'**?")
    st.caption(f"Type: {term['mf_type']} | Parameters: {term['params']}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Yes, Delete", use_container_width=True, type="primary"):
            st.session_state.output_variables[variable_idx]['terms'].pop(term_idx)
            close_output_term_dialog(variable_idx, term_idx)
            st.rerun()
    with col2:
        if st.button("Cancel", use_container_width=True):
            close_output_term_dialog(variable_idx, term_idx)
            st.rerun()

@st.dialog("Edit Term")
def edit_term_dialog(variable_idx, term_idx):
    """Dialog for editing a fuzzy term"""
    variable = st.session_state.input_variables[variable_idx]
    term = variable['terms'][term_idx]

    st.markdown(f"**Editing term in variable: `{variable['name']}`**")
    st.caption(f"Range: [{variable['min']}, {variable['max']}]")

    new_term_name = st.text_input("Term Name", value=term['name'])

    # Segmented control for membership function type
    mf_icons = {
        "triangular": "△",
        "trapezoidal": "⬠",
        "gaussian": "⌢",
        "sigmoid": "∫"
    }

    mf_type = st.segmented_control(
        "Membership Function Type",
        options=["triangular", "trapezoidal", "gaussian", "sigmoid"],
        format_func=lambda x: f"{mf_icons[x]} {x.title()}",
        default=term['mf_type'],
        selection_mode="single"
    )

    st.markdown("---")
    st.markdown("**Parameters**")

    # Get current params or defaults
    current_params = term['params'] if term['mf_type'] == mf_type else None

    if mf_type == "triangular":
        st.caption("Triangular function: three points (a, b, c)")
        col1, col2, col3 = st.columns(3)
        with col1:
            p1 = st.number_input("a (left)", value=float(current_params[0]) if current_params else variable['min'])
        with col2:
            p2 = st.number_input("b (peak)", value=float(current_params[1]) if current_params else (variable['min'] + variable['max'])/2)
        with col3:
            p3 = st.number_input("c (right)", value=float(current_params[2]) if current_params else variable['max'])
        params = (p1, p2, p3)

    elif mf_type == "trapezoidal":
        st.caption("Trapezoidal function: four points (a, b, c, d)")
        col1, col2 = st.columns(2)
        with col1:
            p1 = st.number_input("a (left)", value=float(current_params[0]) if current_params else variable['min'])
            p2 = st.number_input("b (left peak)", value=float(current_params[1]) if current_params else variable['min'] + (variable['max']-variable['min'])*0.25)
        with col2:
            p3 = st.number_input("c (right peak)", value=float(current_params[2]) if current_params else variable['min'] + (variable['max']-variable['min'])*0.75)
            p4 = st.number_input("d (right)", value=float(current_params[3]) if current_params else variable['max'])
        params = (p1, p2, p3, p4)

    elif mf_type == "gaussian":
        st.caption("Gaussian function: mean and standard deviation")
        col1, col2 = st.columns(2)
        with col1:
            p1 = st.number_input("μ (mean)", value=float(current_params[0]) if current_params else (variable['min'] + variable['max'])/2)
        with col2:
            p2 = st.number_input("σ (std dev)", value=float(current_params[1]) if current_params else (variable['max']-variable['min'])/6)
        params = (p1, p2)

    else:  # sigmoid
        st.caption("Sigmoid function: slope and center")
        col1, col2 = st.columns(2)
        with col1:
            p1 = st.number_input("a (slope)", value=float(current_params[0]) if current_params else 1.0)
        with col2:
            p2 = st.number_input("c (center)", value=float(current_params[1]) if current_params else (variable['min'] + variable['max'])/2)
        params = (p1, p2)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✓ Save Changes", use_container_width=True, type="primary"):
            other_terms = [t['name'] for i, t in enumerate(variable['terms']) if i != term_idx]
            if new_term_name and new_term_name not in other_terms:
                st.session_state.input_variables[variable_idx]['terms'][term_idx] = {
                    'name': new_term_name,
                    'mf_type': mf_type,
                    'params': params
                }
                close_term_dialog(variable_idx, term_idx)
                st.rerun()
            elif not new_term_name:
                st.error("Please enter a term name")
            else:
                st.error("Term name already exists")
    with col2:
        if st.button("Cancel", use_container_width=True):
            close_term_dialog(variable_idx, term_idx)
            st.rerun()

@st.dialog("Delete Term")
def delete_term_dialog(variable_idx, term_idx):
    """Dialog for confirming term deletion"""
    variable = st.session_state.input_variables[variable_idx]
    term = variable['terms'][term_idx]

    st.warning(f"Are you sure you want to delete term **'{term['name']}'**?")
    st.caption(f"Type: {term['mf_type']} | Parameters: {term['params']}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Yes, Delete", use_container_width=True, type="primary"):
            st.session_state.input_variables[variable_idx]['terms'].pop(term_idx)
            close_term_dialog(variable_idx, term_idx)
            st.rerun()
    with col2:
        if st.button("Cancel", use_container_width=True):
            close_term_dialog(variable_idx, term_idx)
            st.rerun()

@st.dialog("Edit Variable")
def edit_variable_dialog(variable_idx):
    """Dialog for editing a variable"""
    variable = st.session_state.input_variables[variable_idx]

    st.markdown(f"**Editing Variable**")

    new_name = st.text_input("Variable Name", value=variable['name'])
    col1, col2 = st.columns(2)
    with col1:
        new_min = st.number_input("Min", value=float(variable['min']))
    with col2:
        new_max = st.number_input("Max", value=float(variable['max']))

    # Show warning if domain changed and there are terms
    domain_changed = (new_min != variable['min'] or new_max != variable['max'])
    if domain_changed and variable['terms']:
        st.warning(f"⚠️ Changing the domain will automatically rescale all {len(variable['terms'])} term(s) parameters.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✓ Save Changes", use_container_width=True, type="primary"):
            if new_name and (new_name == variable['name'] or new_name not in [v['name'] for v in st.session_state.input_variables]):
                # Update name and domain
                st.session_state.input_variables[variable_idx]['name'] = new_name

                # If domain changed, rescale all term parameters
                if domain_changed:
                    old_min, old_max = variable['min'], variable['max']
                    for term_idx, term in enumerate(st.session_state.input_variables[variable_idx]['terms']):
                        new_params = rescale_term_params(
                            old_min, old_max, new_min, new_max,
                            term['params'], term['mf_type']
                        )
                        st.session_state.input_variables[variable_idx]['terms'][term_idx]['params'] = new_params

                st.session_state.input_variables[variable_idx]['min'] = new_min
                st.session_state.input_variables[variable_idx]['max'] = new_max
                close_dialog(variable_idx)
                st.rerun()
            elif not new_name:
                st.error("Please enter a variable name")
            else:
                st.error("Variable name already exists")
    with col2:
        if st.button("Cancel", use_container_width=True, on_click=close_dialog, args=(variable_idx,)):
            pass

@st.dialog("View Variable Details")
def view_variable_dialog(variable_idx):
    """Dialog for viewing variable details"""
    variable = st.session_state.input_variables[variable_idx]

    st.markdown(f"### {variable['name']}")
    st.markdown(f"**Range:** [{variable['min']}, {variable['max']}]")
    st.markdown(f"**Number of Terms:** {len(variable['terms'])}")

    if variable['terms']:
        st.markdown("---")
        st.markdown("**Fuzzy Terms:**")
        for term in variable['terms']:
            with st.container():
                st.markdown(f"**{term['name']}**")
                st.caption(f"Type: {term['mf_type']} | Parameters: {term['params']}")
    else:
        st.info("No terms defined yet")

    if st.button("Close", use_container_width=True):
        close_dialog(variable_idx)
        st.rerun()

@st.dialog("Delete Variable")
def delete_variable_dialog(variable_idx):
    """Dialog for confirming variable deletion"""
    variable = st.session_state.input_variables[variable_idx]

    st.warning(f"Are you sure you want to delete variable **'{variable['name']}'**?")
    if variable['terms']:
        st.error(f"This will also delete {len(variable['terms'])} term(s)!")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Yes, Delete", use_container_width=True, type="primary"):
            st.session_state.input_variables.pop(variable_idx)
            close_dialog(variable_idx)
            st.rerun()
    with col2:
        if st.button("Cancel", use_container_width=True, on_click=close_dialog, args=(variable_idx,)):
            pass

@st.dialog("Add Fuzzy Term")
def add_term_dialog(variable_idx, variable):
    """Dialog for adding a new fuzzy term to a variable"""

    st.markdown(f"**Adding term to variable: `{variable['name']}`**")
    st.caption(f"Range: [{variable['min']}, {variable['max']}]")

    term_name = st.text_input("Term Name", placeholder="e.g., low, medium, high")

    # Use segmented control for membership function selection
    mf_icons = {
        "triangular": "△",
        "trapezoidal": "⬠",
        "gaussian": "⌢",
        "sigmoid": "∫"
    }

    mf_type = st.segmented_control(
        "Membership Function Type",
        options=["triangular", "trapezoidal", "gaussian", "sigmoid"],
        format_func=lambda x: f"{mf_icons[x]} {x.title()}",
        default="triangular",
        selection_mode="single"
    )

    st.markdown("---")
    st.markdown("**Parameters**")

    if mf_type == "triangular":
        st.caption("Triangular function: three points (a, b, c)")
        col1, col2, col3 = st.columns(3)
        with col1:
            p1 = st.number_input("a (left)", value=variable['min'])
        with col2:
            p2 = st.number_input("b (peak)", value=(variable['min'] + variable['max'])/2)
        with col3:
            p3 = st.number_input("c (right)", value=variable['max'])
        params = (p1, p2, p3)

    elif mf_type == "trapezoidal":
        st.caption("Trapezoidal function: four points (a, b, c, d)")
        col1, col2 = st.columns(2)
        with col1:
            p1 = st.number_input("a (left)", value=variable['min'])
            p2 = st.number_input("b (left peak)", value=variable['min'] + (variable['max']-variable['min'])*0.25)
        with col2:
            p3 = st.number_input("c (right peak)", value=variable['min'] + (variable['max']-variable['min'])*0.75)
            p4 = st.number_input("d (right)", value=variable['max'])
        params = (p1, p2, p3, p4)

    elif mf_type == "gaussian":
        st.caption("Gaussian function: mean and standard deviation")
        col1, col2 = st.columns(2)
        with col1:
            p1 = st.number_input("μ (mean)", value=(variable['min'] + variable['max'])/2)
        with col2:
            p2 = st.number_input("σ (std dev)", value=(variable['max']-variable['min'])/6)
        params = (p1, p2)

    else:  # sigmoid
        st.caption("Sigmoid function: slope and center")
        col1, col2 = st.columns(2)
        with col1:
            p1 = st.number_input("a (slope)", value=1.0)
        with col2:
            p2 = st.number_input("c (center)", value=(variable['min'] + variable['max'])/2)
        params = (p1, p2)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✓ Add Term", use_container_width=True, type="primary"):
            if term_name and term_name not in [t['name'] for t in variable['terms']]:
                st.session_state.input_variables[variable_idx]['terms'].append({
                    'name': term_name,
                    'mf_type': mf_type,
                    'params': params
                })
                close_dialog(variable_idx)
                st.rerun()
            elif not term_name:
                st.error("Please enter a term name")
            else:
                st.error("Term name already exists")

    with col2:
        if st.button("Cancel", use_container_width=True, on_click=close_dialog, args=(variable_idx,)):
            pass

# ========== RULE DIALOGS ==========

def close_rule_dialog(rule_idx):
    """Callback to reset rule action selection"""
    st.session_state[f"rule_actions_{rule_idx}"] = None

@st.dialog("Edit Fuzzy Rule")
def edit_rule_dialog(rule_idx):
    """Dialog for editing a fuzzy rule"""
    rule = st.session_state.fuzzy_rules[rule_idx]

    st.markdown(f"**Editing Rule {rule_idx + 1}**")

    # IF part (antecedents)
    st.markdown("**IF** (Antecedents)")
    new_antecedents = {}

    for var in st.session_state.input_variables:
        if var['terms']:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"`{var['name']}`")
            with col2:
                term_options = ["(any)"] + [term['name'] for term in var['terms']]
                current_value = rule['antecedents'].get(var['name'], "(any)")
                selected_term = st.selectbox(
                    f"is",
                    term_options,
                    index=term_options.index(current_value) if current_value in term_options else 0,
                    key=f"edit_rule_input_{rule_idx}_{var['name']}",
                    label_visibility="collapsed"
                )
                if selected_term != "(any)":
                    new_antecedents[var['name']] = selected_term

    if not new_antecedents:
        st.warning("⚠️ Select at least one input term")
    else:
        st.markdown("---")
        # THEN part (consequents)
        st.markdown("**THEN** (Consequents)")
        new_consequents = {}

        for var in st.session_state.output_variables:
            if var['terms']:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"`{var['name']}`")
                with col2:
                    term_options = [term['name'] for term in var['terms']]
                    current_value = rule['consequents'].get(var['name'], term_options[0])
                    selected_term = st.selectbox(
                        f"is ",
                        term_options,
                        index=term_options.index(current_value) if current_value in term_options else 0,
                        key=f"edit_rule_output_{rule_idx}_{var['name']}",
                        label_visibility="collapsed"
                    )
                    new_consequents[var['name']] = selected_term

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✓ Save Changes", use_container_width=True, type="primary"):
                # Check for duplicate rules (excluding current rule)
                rule_exists = any(
                    i != rule_idx and r['antecedents'] == new_antecedents and r['consequents'] == new_consequents
                    for i, r in enumerate(st.session_state.fuzzy_rules)
                )

                if rule_exists:
                    st.error("⚠️ This rule already exists!")
                else:
                    st.session_state.fuzzy_rules[rule_idx] = {
                        'antecedents': new_antecedents,
                        'consequents': new_consequents
                    }
                    close_rule_dialog(rule_idx)
                    st.rerun()

        with col2:
            if st.button("Cancel", use_container_width=True, on_click=close_rule_dialog, args=(rule_idx,)):
                pass

@st.dialog("Delete Fuzzy Rule")
def delete_rule_dialog(rule_idx):
    """Dialog for confirming rule deletion"""
    rule = st.session_state.fuzzy_rules[rule_idx]

    st.warning(f"Are you sure you want to delete **Rule {rule_idx + 1}**?")

    # Show rule details
    ant_str = " AND ".join([f"**{var}** is `{term}`" for var, term in rule['antecedents'].items()])
    st.markdown(f"**IF** {ant_str}")

    cons_str = ", ".join([f"**{var}** is `{term}`" for var, term in rule['consequents'].items()])
    st.markdown(f"**THEN** {cons_str}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Yes, Delete", use_container_width=True, type="primary"):
            st.session_state.fuzzy_rules.pop(rule_idx)
            close_rule_dialog(rule_idx)
            st.rerun()

    with col2:
        if st.button("Cancel", use_container_width=True, on_click=close_rule_dialog, args=(rule_idx,)):
            pass

@st.dialog("Edit Rule (Table View)")
def edit_rule_table_dialog():
    """Dialog for selecting a rule to edit in table view"""
    st.markdown("**Select a rule to edit:**")

    rule_options = [f"R{i+1}: {format_rule_compact(rule)}" for i, rule in enumerate(st.session_state.fuzzy_rules)]

    selected = st.selectbox(
        "Rule",
        options=range(len(st.session_state.fuzzy_rules)),
        format_func=lambda x: rule_options[x],
        label_visibility="collapsed"
    )

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✏️ Edit Selected", use_container_width=True, type="primary"):
            st.session_state.editing_rule_idx = selected
            st.rerun()

    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()

@st.dialog("Delete Rules (Table View)")
def delete_rules_table_dialog():
    """Dialog for selecting multiple rules to delete in table view"""
    st.markdown("**Select rules to delete:**")

    # Create checkboxes for each rule
    to_delete = []
    for i, rule in enumerate(st.session_state.fuzzy_rules):
        rule_str = format_rule_compact(rule)
        if st.checkbox(f"R{i+1}: {rule_str}", key=f"delete_check_{i}"):
            to_delete.append(i)

    st.markdown("---")

    if to_delete:
        st.warning(f"⚠️ You are about to delete {len(to_delete)} rule(s)")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Delete Selected", use_container_width=True, type="primary", disabled=len(to_delete)==0):
            # Delete in reverse order to maintain indices
            for idx in sorted(to_delete, reverse=True):
                st.session_state.fuzzy_rules.pop(idx)
            st.success(f"✓ Deleted {len(to_delete)} rule(s)")
            st.rerun()

    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()

def format_rule_compact(rule):
    """Format a rule in compact form for display"""
    ant_parts = [f"{var}={term}" for var, term in rule['antecedents'].items()]
    cons_parts = [f"{var}={term}" for var, term in rule['consequents'].items()]
    return f"IF {' AND '.join(ant_parts)} THEN {', '.join(cons_parts)}"

# ========== FIS MANAGEMENT DIALOGS ==========

@st.dialog("New Fuzzy Inference System")
def new_fis_dialog():
    """Dialog for creating a new FIS"""
    st.markdown("**Create a new Fuzzy Inference System**")

    fis_name = st.text_input("FIS Name", placeholder="e.g., Temperature Controller")

    # Get default system type from navigation (if set)
    default_type = st.session_state.get('inference_system_type', 'Mamdani')
    default_idx = 0 if default_type == "Mamdani" else 1

    fis_type = st.selectbox("System Type", ["Mamdani", "Sugeno (TSK)"], index=default_idx)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✓ Create FIS", use_container_width=True, type="primary"):
            if fis_name:
                # Check if name already exists
                if fis_name in [fis['name'] for fis in st.session_state.fis_list]:
                    st.error("A FIS with this name already exists!")
                else:
                    # Create new FIS
                    new_fis = {
                        'name': fis_name,
                        'type': fis_type,
                        'input_variables': [],
                        'output_variables': [],
                        'fuzzy_rules': []
                    }
                    st.session_state.fis_list.append(new_fis)
                    st.session_state.active_fis_idx = len(st.session_state.fis_list) - 1
                    st.success(f"✓ Created FIS: {fis_name}")
                    st.rerun()
            else:
                st.error("Please enter a FIS name")

    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()

@st.dialog("Rename FIS")
def rename_fis_dialog():
    """Dialog for renaming the active FIS"""
    active_fis = st.session_state.fis_list[st.session_state.active_fis_idx]

    st.markdown(f"**Renaming: {active_fis['name']}**")

    new_name = st.text_input("New Name", value=active_fis['name'])

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✓ Rename", use_container_width=True, type="primary"):
            if new_name and new_name != active_fis['name']:
                # Check if name already exists
                if new_name in [fis['name'] for i, fis in enumerate(st.session_state.fis_list) if i != st.session_state.active_fis_idx]:
                    st.error("A FIS with this name already exists!")
                else:
                    st.session_state.fis_list[st.session_state.active_fis_idx]['name'] = new_name
                    st.success(f"✓ Renamed to: {new_name}")
                    st.rerun()
            elif not new_name:
                st.error("Please enter a name")

    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()

@st.dialog("Delete FIS")
def delete_fis_dialog():
    """Dialog for deleting the active FIS"""
    active_fis = st.session_state.fis_list[st.session_state.active_fis_idx]

    st.warning(f"Are you sure you want to delete **{active_fis['name']}**?")

    st.markdown("This will permanently delete:")
    st.markdown(f"- {len(active_fis['input_variables'])} input variable(s)")
    st.markdown(f"- {len(active_fis['output_variables'])} output variable(s)")
    st.markdown(f"- {len(active_fis['fuzzy_rules'])} rule(s)")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Yes, Delete", use_container_width=True, type="primary"):
            st.session_state.fis_list.pop(st.session_state.active_fis_idx)

            # Adjust active index
            if len(st.session_state.fis_list) == 0:
                # Create default FIS if all deleted
                st.session_state.fis_list = [{
                    'name': 'FIS 1',
                    'type': 'Mamdani',
                    'input_variables': [],
                    'output_variables': [],
                    'fuzzy_rules': []
                }]
                st.session_state.active_fis_idx = 0
            elif st.session_state.active_fis_idx >= len(st.session_state.fis_list):
                st.session_state.active_fis_idx = len(st.session_state.fis_list) - 1

            st.success("✓ FIS deleted")
            st.rerun()

    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()

@st.dialog("Load FIS from JSON")
def load_fis_dialog():
    """Dialog for loading FIS from exported JSON file"""
    import json

    st.markdown("**Upload a FIS JSON file**")
    st.markdown("Upload a JSON file exported from MamdaniSystem or SugenoSystem")

    uploaded_file = st.file_uploader("Choose JSON file", type=['json'])

    if uploaded_file is not None:
        try:
            # Read and parse JSON
            json_data = json.loads(uploaded_file.getvalue().decode('utf-8'))

            # Display preview
            st.success("✓ File loaded successfully!")

            with st.expander("📋 Preview FIS Data"):
                st.markdown(f"**Name:** {json_data.get('name', 'Unnamed')}")
                st.markdown(f"**Type:** {json_data.get('system_type', 'Unknown')}")
                st.markdown(f"**Inputs:** {len(json_data.get('input_variables', {}))}")
                st.markdown(f"**Outputs:** {len(json_data.get('output_variables', {}))}")
                st.markdown(f"**Rules:** {len(json_data.get('rules', []))}")

                # Show input variables
                if json_data.get('input_variables'):
                    st.markdown("**Input Variables:**")
                    for var_name, var_data in json_data['input_variables'].items():
                        st.markdown(f"  - {var_name}: {len(var_data['terms'])} terms")

                # Show output variables
                if json_data.get('output_variables'):
                    st.markdown("**Output Variables:**")
                    for var_name, var_data in json_data['output_variables'].items():
                        st.markdown(f"  - {var_name}: {len(var_data['terms'])} terms")

            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✓ Import FIS", use_container_width=True, type="primary"):
                    # Convert JSON to internal format
                    fis_name = json_data.get('name', 'Imported FIS')
                    system_type = json_data.get('system_type', 'MamdaniSystem')

                    # Map system type
                    if 'Mamdani' in system_type:
                        fis_type = 'Mamdani'
                    elif 'Sugeno' in system_type or 'TSK' in system_type:
                        fis_type = 'Sugeno (TSK)'
                    else:
                        fis_type = 'Mamdani'

                    # Convert input variables
                    input_variables = []
                    for var_name, var_data in json_data.get('input_variables', {}).items():
                        universe = var_data.get('universe', [0, 100])
                        terms = []

                        for term_name, term_data in var_data.get('terms', {}).items():
                            terms.append({
                                'name': term_name,
                                'mf_type': term_data.get('mf_type', 'triangular'),
                                'params': term_data.get('params', [])
                            })

                        input_variables.append({
                            'name': var_name,
                            'min': universe[0],
                            'max': universe[1],
                            'terms': terms
                        })

                    # Convert output variables
                    output_variables = []
                    for var_name, var_data in json_data.get('output_variables', {}).items():
                        universe = var_data.get('universe', [0, 100])
                        terms = []

                        for term_name, term_data in var_data.get('terms', {}).items():
                            terms.append({
                                'name': term_name,
                                'mf_type': term_data.get('mf_type', 'triangular'),
                                'params': term_data.get('params', [])
                            })

                        output_variables.append({
                            'name': var_name,
                            'min': universe[0],
                            'max': universe[1],
                            'terms': terms
                        })

                    # Convert rules
                    fuzzy_rules = []
                    for rule in json_data.get('rules', []):
                        fuzzy_rules.append({
                            'antecedents': rule.get('antecedents', {}),
                            'consequents': rule.get('consequents', {})
                        })

                    # Create new FIS
                    new_fis = {
                        'name': fis_name,
                        'type': fis_type,
                        'input_variables': input_variables,
                        'output_variables': output_variables,
                        'fuzzy_rules': fuzzy_rules
                    }

                    # Check if name exists, add number suffix if needed
                    existing_names = [fis['name'] for fis in st.session_state.fis_list]
                    if fis_name in existing_names:
                        counter = 2
                        while f"{fis_name} ({counter})" in existing_names:
                            counter += 1
                        new_fis['name'] = f"{fis_name} ({counter})"

                    # Add to list and set as active
                    st.session_state.fis_list.append(new_fis)
                    st.session_state.active_fis_idx = len(st.session_state.fis_list) - 1

                    st.success(f"✓ Imported FIS: {new_fis['name']}")
                    st.rerun()

            with col2:
                if st.button("Cancel", use_container_width=True):
                    st.rerun()

        except json.JSONDecodeError:
            st.error("❌ Invalid JSON file")
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    else:
        st.info("👆 Upload a JSON file to import a FIS")

@st.dialog("Export FIS to JSON")
def export_fis_dialog():
    """Dialog for exporting FIS to JSON format compatible with MamdaniSystem/SugenoSystem"""
    import json
    from datetime import datetime

    # Get active FIS
    active_fis = st.session_state.fis_list[st.session_state.active_fis_idx]

    st.markdown("**Export your Fuzzy Inference System**")
    st.markdown("Generate a JSON file compatible with `MamdaniSystem.export_to_json()`")

    # Preview
    with st.expander("📋 System Preview"):
        st.markdown(f"**Name:** {active_fis['name']}")
        st.markdown(f"**Type:** {active_fis['type']}")
        st.markdown(f"**Input Variables:** {len(active_fis['input_variables'])}")
        st.markdown(f"**Output Variables:** {len(active_fis['output_variables'])}")
        st.markdown(f"**Rules:** {len(active_fis['fuzzy_rules'])}")

    # Convert to MamdaniSystem JSON format
    try:
        # Map system type
        if 'Sugeno' in active_fis['type'] or 'TSK' in active_fis['type']:
            system_type = "SugenoSystem"
        else:
            system_type = "MamdaniSystem"

        # Build JSON structure
        json_data = {
            "system_type": system_type,
            "name": active_fis['name'],
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "version": "1.0",
                "library": "fuzzy_systems"
            },
            "defuzzification_method": "centroid",
            "inference_config": {
                "and_method": "min",
                "or_method": "max",
                "implication_method": "min",
                "aggregation_method": "max"
            },
            "input_variables": {},
            "output_variables": {},
            "rules": []
        }

        # Convert input variables
        for var in active_fis['input_variables']:
            json_data['input_variables'][var['name']] = {
                "universe": [var['min'], var['max']],
                "terms": {}
            }

            for term in var['terms']:
                json_data['input_variables'][var['name']]['terms'][term['name']] = {
                    "mf_type": term['mf_type'],
                    "params": term['params']
                }

        # Convert output variables
        for var in active_fis['output_variables']:
            json_data['output_variables'][var['name']] = {
                "universe": [var['min'], var['max']],
                "terms": {}
            }

            for term in var['terms']:
                json_data['output_variables'][var['name']]['terms'][term['name']] = {
                    "mf_type": term['mf_type'],
                    "params": term['params']
                }

        # Convert rules
        for rule in active_fis['fuzzy_rules']:
            json_data['rules'].append({
                "antecedents": rule['antecedents'],
                "consequents": rule['consequents'],
                "operator": "AND",
                "weight": 1.0
            })

        # Generate JSON string
        json_string = json.dumps(json_data, indent=2)

        st.markdown("---")

        # Show JSON preview
        with st.expander("👁️ Preview JSON", expanded=False):
            st.code(json_string, language='json')

        # Download button
        filename = f"{active_fis['name'].replace(' ', '_').lower()}.json"

        st.download_button(
            label="💾 Download JSON File",
            data=json_string,
            file_name=filename,
            mime="application/json",
            use_container_width=True,
            type="primary"
        )

        st.success("✓ JSON ready for download!")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Copy to Clipboard", use_container_width=True):
                # Note: Clipboard API requires user interaction in browser
                st.info("💡 Use the download button or manually copy from the preview above")

        with col2:
            if st.button("Close", use_container_width=True):
                st.rerun()

    except Exception as e:
        st.error(f"❌ Error generating JSON: {str(e)}")

def run():
    """Render inference systems page"""

    # Initialize FIS management in session state
    if 'fis_list' not in st.session_state:
        st.session_state.fis_list = []  # Start with empty list
    if 'active_fis_idx' not in st.session_state:
        st.session_state.active_fis_idx = 0

    # Check if we have any FIS
    has_fis = len(st.session_state.fis_list) > 0

    # Get active FIS (only if exists)
    if has_fis:
        active_fis = st.session_state.fis_list[st.session_state.active_fis_idx]

        # Create aliases for easier access (backward compatibility)
        st.session_state.input_variables = active_fis['input_variables']
        st.session_state.output_variables = active_fis['output_variables']
        st.session_state.fuzzy_rules = active_fis['fuzzy_rules']
    else:
        active_fis = None
        # Create empty aliases
        st.session_state.input_variables = []
        st.session_state.output_variables = []
        st.session_state.fuzzy_rules = []

    # Sidebar configuration
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 0.25rem 0 0.125rem 0; margin-top: 0.5rem;">
            <h2 style="margin: 0.25rem 0 0.125rem 0; color: #667eea;">Inference Systems</h2>
            <p style="color: #6b7280; font-size: 0.9rem; margin: 0;">
                Build and test Mamdani and Sugeno fuzzy inference systems
            </p>
        </div>
        <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 0.25rem 0 0.5rem 0;">
        """, unsafe_allow_html=True)

        # FIS Management
        st.markdown("**Fuzzy Inference Systems**")

        # New FIS and Load FIS buttons in columns
       
        if st.button("New FIS", use_container_width=True):
                new_fis_dialog()
        if st.button("Load FIS", use_container_width=True):
                load_fis_dialog()

        # Only show FIS management controls if there's at least one FIS
        if has_fis:
            # FIS selector (only show if more than one FIS)
            if len(st.session_state.fis_list) > 1:
                fis_names = [f"{fis['name']} ({fis['type']})" for fis in st.session_state.fis_list]
                selected_idx = st.selectbox(
                    "Select FIS",
                    range(len(st.session_state.fis_list)),
                    format_func=lambda x: fis_names[x],
                    index=st.session_state.active_fis_idx,
                    label_visibility="collapsed"
                )

                if selected_idx != st.session_state.active_fis_idx:
                    st.session_state.active_fis_idx = selected_idx
                    st.rerun()
            else:
                # Show current FIS name (type is shown in System Info below)
                st.markdown(f"<div style='text-align: center; padding: 0.5rem; background: #f8f9fa; border-radius: 6px; margin-bottom: 0.5rem;'><strong>{active_fis['name']}</strong></div>", unsafe_allow_html=True)

            # FIS actions
            
            if st.button("Rename FIS", use_container_width=True):
                rename_fis_dialog()
            if st.button("Delete FIS", use_container_width=True, disabled=len(st.session_state.fis_list)==1):
                delete_fis_dialog()

            st.markdown("<hr style='border: none; border-top: 1px solid #e5e7eb; margin: 1rem 0;'>", unsafe_allow_html=True)

            # System info
            st.markdown("**System Info**")
            st.caption(f"Type: {active_fis['type']}")
            st.caption(f"Inputs: {len(active_fis['input_variables'])}")
            st.caption(f"Outputs: {len(active_fis['output_variables'])}")
            st.caption(f"Rules: {len(active_fis['fuzzy_rules'])}")

            st.markdown("<hr style='border: none; border-top: 1px solid #e5e7eb; margin: 1rem 0;'>", unsafe_allow_html=True)

            # Action buttons
            st.markdown("**Actions**")

            # Save/Export button
            if st.button("💾 Export JSON", use_container_width=True):
                export_fis_dialog()

            # Reset button
            if st.button("🔄 Reset FIS", use_container_width=True):
                # Reset current FIS
                st.session_state.fis_list[st.session_state.active_fis_idx] = {
                    'name': active_fis['name'],
                    'type': active_fis['type'],
                    'input_variables': [],
                    'output_variables': [],
                    'fuzzy_rules': []
                }
                st.rerun()
        else:
            # No FIS created yet - show instructions
            st.info("👆 Click **New** to create or **Load** to import a FIS!")

    # Main content - conditional on FIS existence
    if not has_fis:
        # Welcome screen when no FIS exists
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem;">
            <h1 style="color: #667eea; font-size: 2.5rem; margin-bottom: 1rem;">
                Welcome to Fuzzy Inference Systems
            </h1>
            <p style="color: #6b7280; font-size: 1.2rem; max-width: 600px; margin: 0 auto 2rem auto; line-height: 1.6;">
                Create and test Mamdani and Sugeno fuzzy inference systems with an intuitive interface.
                Define variables, configure membership functions, build rules, and evaluate your system.
            </p>
            <div style="background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
                        padding: 2rem; border-radius: 12px; max-width: 550px; margin: 0 auto;">
                <h3 style="color: #667eea; margin-bottom: 1rem;">🚀 Get Started</h3>
                <p style="color: #4b5563; margin-bottom: 0.5rem;">
                    <strong>Create New:</strong> Click <strong>New</strong> in the sidebar to build a system from scratch
                </p>
                <p style="color: #4b5563; margin-bottom: 0.5rem;">
                    <strong>Load Existing:</strong> Click <strong>Load</strong> to import a JSON file exported from MamdaniSystem
                </p>
                <p style="color: #4b5563; margin-bottom: 0.5rem;">
                    • Add input and output variables with membership functions
                </p>
                <p style="color: #4b5563; margin-bottom: 0.5rem;">
                    • Define fuzzy rules (IF-THEN logic)
                </p>
                <p style="color: #4b5563; margin: 0;">
                    • Test your system with real values
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Store system type for backward compatibility
        system_type = active_fis['type']

        # Main content - Just title
        st.markdown(f"""
        <div style="text-align: center; padding: 0.5rem 0;">
            <h3 style="color: #6b7280; font-weight: 500; margin: 0; font-size: 1.1rem;">
                {system_type} Fuzzy Inference System
            </h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='border-bottom: 1px solid #e5e7eb; margin: 0.5rem 0 1.5rem 0;'></div>", unsafe_allow_html=True)

        # Tabs for different stages
        tab1, tab2, tab3, tab4 = st.tabs([
            "📥 Input Variables",
            "📤 Output Variables",
            "📜 Fuzzy Rules",
            "⚡ Inference"
        ])

        with tab1:
            # st.markdown("##### ")
    
            # Initialize counters if needed
            if 'new_var_counter' not in st.session_state:
                st.session_state.new_var_counter = 0
    
            def add_variable():
                var_name = st.session_state.get('new_var_name_input', '')
                var_min = st.session_state.get('new_var_min_input', 0.0)
                var_max = st.session_state.get('new_var_max_input', 100.0)
    
                if var_name and var_name not in [v['name'] for v in st.session_state.input_variables]:
                    st.session_state.input_variables.append({
                        'name': var_name,
                        'min': var_min,
                        'max': var_max,
                        'terms': []
                    })
                    # Increment counter to force form reset
                    st.session_state.new_var_counter += 1
    
            # Add new variable section
            with st.expander("➕ Add New Input Variable", expanded=len(st.session_state.input_variables) == 0):
                col1, col2, col3 = st.columns(3)
                with col1:
                    var_name = st.text_input("Variable Name", placeholder="e.g., temperature",
                                            key=f"new_var_name_input_{st.session_state.new_var_counter}")
                with col2:
                    var_min = st.number_input("Min", value=0.0,
                                             key=f"new_var_min_input_{st.session_state.new_var_counter}")
                with col3:
                    var_max = st.number_input("Max", value=100.0,
                                             key=f"new_var_max_input_{st.session_state.new_var_counter}")
    
                # Store values in session state for callback
                st.session_state['new_var_name_input'] = var_name
                st.session_state['new_var_min_input'] = var_min
                st.session_state['new_var_max_input'] = var_max
    
                if st.button("✓ Add Variable", use_container_width=True, key="add_var_btn", on_click=add_variable):
                    if not var_name:
                        st.error("Please enter a variable name")
                    elif var_name in [v['name'] for v in st.session_state.input_variables]:
                        st.error("Variable already exists")
    
            st.markdown("<br>", unsafe_allow_html=True)
    
            # Display existing variables
            if st.session_state.input_variables:
                st.markdown("**Configured Variables**")
    
                # Track if any dialog has been opened (only one dialog per run)
                dialog_opened = False
    
                for idx, variable in enumerate(st.session_state.input_variables):
                    with st.container():
                        st.markdown(f"""
                        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 4px solid #667eea;">
                            <h4 style="margin: 0 0 0.5rem 0; color: #667eea;">{variable['name']}</h4>
                            <p style="margin: 0; color: #6b7280; font-size: 0.875rem;">
                                Range: [{variable['min']}, {variable['max']}] | Terms: {len(variable['terms'])}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
    
                        # Action buttons using icons
                        action_icons = {
                            # "view": "👁️ View",
                            "edit": "✏️ Edit",
                            "add_term": "➕ Add Term",
                            "delete": "🗑️ Delete"
                        }
    
                        action = st.segmented_control(
                            f"Actions for {variable['name']}",
                            options=list(action_icons.keys()),
                            format_func=lambda x: action_icons[x],
                            selection_mode="single",
                            key=f"actions_{idx}",
                            label_visibility="collapsed"
                        )
    
                        # Open dialog only if no other dialog has been opened yet
                        if not dialog_opened and action:
                            if action == "view":
                                view_variable_dialog(idx)
                                dialog_opened = True
                            elif action == "edit":
                                edit_variable_dialog(idx)
                                dialog_opened = True
                            elif action == "add_term":
                                add_term_dialog(idx, variable)
                                dialog_opened = True
                            elif action == "delete":
                                delete_variable_dialog(idx)
                                dialog_opened = True
    
                        # Display terms in expander
                        if variable['terms']:
                            with st.expander(f"📋 Terms ({len(variable['terms'])})", expanded=False):
                                # var_data = next(v for v in active_fis['input_variables'] if v['name'] == var_name)
                                engine = InferenceEngine(active_fis)
                                fig = go.Figure()

                                # Plot each term
                                for term in variable['terms']:
                                    x, y = engine.get_term_membership_curve(variable['name'], term['name'])
                                    fig.add_trace(go.Scatter(
                                        x=x, y=y,
                                        mode='lines',
                                        name=term['name'],
                                        hovertemplate=f"{term['name']}<br>x=%{{x:.2f}}<br>μ=%{{y:.3f}}<extra></extra>"
                                    ))

                                

                                fig.update_layout(
                                    title=f"Membership Functions - {variable['name']}",
                                    xaxis_title=var_name,
                                    yaxis_title="Membership Degree (μ)",
                                    hovermode='closest',
                                    height=350
                                )

                                st.plotly_chart(fig, use_container_width=True,key=f"input_chart_for_{variable['name']}")
                                
                                for t_idx, term in enumerate(variable['terms']):
                                    col_t1, col_t2 = st.columns([3, 1])
                                    with col_t1:
                                        st.markdown(f"**{term['name']}**")
                                        st.caption(f"`{term['mf_type']}` {term['params']}")
                                    with col_t2:
                                        term_action_icons = {
                                            "edit": "✏️",
                                            "delete": "🗑️"
                                        }

                                        term_action = st.segmented_control(
                                            f"Actions for term {term['name']}",
                                            options=list(term_action_icons.keys()),
                                            format_func=lambda x: term_action_icons[x],
                                            selection_mode="single",
                                            key=f"term_actions_{idx}_{t_idx}",
                                            label_visibility="collapsed"
                                        )

                                        # Open dialog only if no other dialog has been opened yet
                                        if not dialog_opened and term_action:
                                            if term_action == "edit":
                                                edit_term_dialog(idx, t_idx)
                                                dialog_opened = True
                                            elif term_action == "delete":
                                                delete_term_dialog(idx, t_idx)
                                                dialog_opened = True
            else:
                st.info("No input variables configured. Click '➕ Add New Input Variable' to get started.")
    
        with tab2:
            # st.markdown("### Output Variables")
    
            # Initialize counter if needed
            if 'new_output_var_counter' not in st.session_state:
                st.session_state.new_output_var_counter = 0
    
            def add_output_variable():
                var_name = st.session_state.get('new_output_var_name_input', '')
                var_min = st.session_state.get('new_output_var_min_input', 0.0)
                var_max = st.session_state.get('new_output_var_max_input', 100.0)
    
                if var_name and var_name not in [v['name'] for v in st.session_state.output_variables]:
                    st.session_state.output_variables.append({
                        'name': var_name,
                        'min': var_min,
                        'max': var_max,
                        'terms': []
                    })
                    # Increment counter to force form reset
                    st.session_state.new_output_var_counter += 1
    
            # Add new variable section
            with st.expander("➕ Add New Output Variable", expanded=len(st.session_state.output_variables) == 0):
                col1, col2, col3 = st.columns(3)
                with col1:
                    var_name = st.text_input("Variable Name", placeholder="e.g., fan_speed",
                                            key=f"new_output_var_name_input_{st.session_state.new_output_var_counter}")
                with col2:
                    var_min = st.number_input("Min", value=0.0,
                                             key=f"new_output_var_min_input_{st.session_state.new_output_var_counter}")
                with col3:
                    var_max = st.number_input("Max", value=100.0,
                                             key=f"new_output_var_max_input_{st.session_state.new_output_var_counter}")
    
                # Store values in session state for callback
                st.session_state['new_output_var_name_input'] = var_name
                st.session_state['new_output_var_min_input'] = var_min
                st.session_state['new_output_var_max_input'] = var_max
    
                if st.button("✓ Add Variable", use_container_width=True, key="add_output_var_btn", on_click=add_output_variable):
                    if not var_name:
                        st.error("Please enter a variable name")
                    elif var_name in [v['name'] for v in st.session_state.output_variables]:
                        st.error("Variable already exists")
    
            st.markdown("<br>", unsafe_allow_html=True)
    
            # Display existing variables
            if st.session_state.output_variables:
                st.markdown("**Configured Variables**")
    
                # Track if any dialog has been opened (only one dialog per run)
                dialog_opened = False
    
                for idx, variable in enumerate(st.session_state.output_variables):
                    with st.container():
                        st.markdown(f"""
                        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 4px solid #10b981;">
                            <h4 style="margin: 0 0 0.5rem 0; color: #10b981;">{variable['name']}</h4>
                            <p style="margin: 0; color: #6b7280; font-size: 0.875rem;">
                                Range: [{variable['min']}, {variable['max']}] | Terms: {len(variable['terms'])}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
    
                        # Action buttons using icons
                        action_icons = {
                            # "view": "👁️ View",
                            "edit": "✏️ Edit",
                            "add_term": "➕ Add Term",
                            "delete": "🗑️ Delete"
                        }
    
                        action = st.segmented_control(
                            f"Actions for {variable['name']}",
                            options=list(action_icons.keys()),
                            format_func=lambda x: action_icons[x],
                            selection_mode="single",
                            key=f"output_actions_{idx}",
                            label_visibility="collapsed"
                        )
    
                        # Open dialog only if no other dialog has been opened yet
                        if not dialog_opened and action:
                            if action == "view":
                                view_output_variable_dialog(idx)
                                dialog_opened = True
                            elif action == "edit":
                                edit_output_variable_dialog(idx)
                                dialog_opened = True
                            elif action == "add_term":
                                add_output_term_dialog(idx, variable)
                                dialog_opened = True
                            elif action == "delete":
                                delete_output_variable_dialog(idx)
                                dialog_opened = True
    
                        # Display terms in expander
                        if variable['terms']:
                            with st.expander(f"📋 Terms ({len(variable['terms'])})", expanded=False):

                                engine = InferenceEngine(active_fis)
                                fig = go.Figure()

                                # Plot each term
                                for term in variable['terms']:
                                    x, y = engine.get_term_membership_curve(variable['name'], term['name'])
                                    fig.add_trace(go.Scatter(
                                        x=x, y=y,
                                        mode='lines',
                                        name=term['name'],
                                        hovertemplate=f"{term['name']}<br>x=%{{x:.2f}}<br>μ=%{{y:.3f}}<extra></extra>"
                                    ))

                                

                                fig.update_layout(
                                    title=f"Membership Functions - {variable['name']}",
                                    xaxis_title=var_name,
                                    yaxis_title="Membership Degree (μ)",
                                    hovermode='closest',
                                    height=350
                                )

                                st.plotly_chart(fig, use_container_width=True,key=f"output_chart_for_{variable['name']}")


                                for t_idx, term in enumerate(variable['terms']):
                                    col_t1, col_t2 = st.columns([3, 1])
                                    with col_t1:
                                        st.markdown(f"**{term['name']}**")
                                        st.caption(f"`{term['mf_type']}` {term['params']}")
                                    with col_t2:
                                        term_action_icons = {
                                            "edit": "✏️",
                                            "delete": "🗑️"
                                        }

                                        term_action = st.segmented_control(
                                            f"Actions for term {term['name']}",
                                            options=list(term_action_icons.keys()),
                                            format_func=lambda x: term_action_icons[x],
                                            selection_mode="single",
                                            key=f"output_term_actions_{idx}_{t_idx}",
                                            label_visibility="collapsed"
                                        )

                                        # Open dialog only if no other dialog has been opened yet
                                        if not dialog_opened and term_action:
                                            if term_action == "edit":
                                                edit_output_term_dialog(idx, t_idx)
                                                dialog_opened = True
                                            elif term_action == "delete":
                                                delete_output_term_dialog(idx, t_idx)
                                                dialog_opened = True
            else:
                st.info("No output variables configured. Click '➕ Add New Output Variable' to get started.")
    
        with tab3:
            # st.markdown("### Fuzzy Rules")
    
            # Initialize counter for rule form reset
            if 'new_rule_counter' not in st.session_state:
                st.session_state.new_rule_counter = 0
    
            # Check if we have variables and terms configured
            has_inputs = len(st.session_state.input_variables) > 0
            has_outputs = len(st.session_state.output_variables) > 0
    
            if not has_inputs or not has_outputs:
                st.warning("⚠️ Please configure input and output variables first before creating rules.")
                if not has_inputs:
                    st.info("📥 Go to 'Input Variables' tab to add input variables and terms")
                if not has_outputs:
                    st.info("📤 Go to 'Output Variables' tab to add output variables and terms")
            else:
                # Check if all variables have terms
                missing_terms_inputs = [v['name'] for v in st.session_state.input_variables if len(v['terms']) == 0]
                missing_terms_outputs = [v['name'] for v in st.session_state.output_variables if len(v['terms']) == 0]
    
                if missing_terms_inputs or missing_terms_outputs:
                    st.warning("⚠️ Some variables don't have fuzzy terms defined:")
                    if missing_terms_inputs:
                        st.info(f"📥 Input variables without terms: {', '.join(missing_terms_inputs)}")
                    if missing_terms_outputs:
                        st.info(f"📤 Output variables without terms: {', '.join(missing_terms_outputs)}")
                    st.markdown("---")
    
                # Add Rule Interface
                with st.expander("➕ Add New Fuzzy Rule", expanded=False):
                    st.markdown("Build an IF-THEN fuzzy rule:")
    
                    # IF part (antecedents) - one condition per input variable
                    st.markdown("**IF** (Antecedents)")
                    antecedents = {}
    
                    for idx, var in enumerate(st.session_state.input_variables):
                        if var['terms']:
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                st.markdown(f"`{var['name']}`")
                            with col2:
                                term_options = ["(any)"] + [term['name'] for term in var['terms']]
                                selected_term = st.selectbox(
                                    f"is",
                                    term_options,
                                    key=f"rule_input_{var['name']}_{st.session_state.new_rule_counter}",
                                    label_visibility="collapsed"
                                )
                                if selected_term != "(any)":
                                    antecedents[var['name']] = selected_term
    
                    if not antecedents:
                        st.info("Select at least one input term to create a rule")
                    else:
                        st.markdown("---")
                        # THEN part (consequents) - one output per output variable
                        st.markdown("**THEN** (Consequents)")
                        consequents = {}
    
                        for var in st.session_state.output_variables:
                            if var['terms']:
                                col1, col2 = st.columns([1, 2])
                                with col1:
                                    st.markdown(f"`{var['name']}`")
                                with col2:
                                    term_options = [term['name'] for term in var['terms']]
                                    selected_term = st.selectbox(
                                        f"is ",
                                        term_options,
                                        key=f"rule_output_{var['name']}_{st.session_state.new_rule_counter}",
                                        label_visibility="collapsed"
                                    )
                                    consequents[var['name']] = selected_term
    
                        st.markdown("---")
    
                        if st.button("✓ Add Rule", type="primary", use_container_width=True):
                            # Check for duplicate rules
                            rule_exists = any(
                                r['antecedents'] == antecedents and r['consequents'] == consequents
                                for r in st.session_state.fuzzy_rules
                            )
    
                            if rule_exists:
                                st.error("⚠️ This rule already exists!")
                            else:
                                st.session_state.fuzzy_rules.append({
                                    'antecedents': antecedents,
                                    'consequents': consequents
                                })
                                # Increment counter to reset form
                                st.session_state.new_rule_counter += 1
                                st.success(f"✓ Rule added! Total rules: {len(st.session_state.fuzzy_rules)}")
                                st.rerun()
    
                st.markdown("<br>", unsafe_allow_html=True)
    
                # Display existing rules
                if st.session_state.fuzzy_rules:
                    col_header1, col_header2 = st.columns([3, 1])
                    with col_header1:
                        st.markdown(f"**Fuzzy Rules ({len(st.session_state.fuzzy_rules)})**")
                    with col_header2:
                        # Toggle between view modes
                        view_icons = {
                            "compact": "📝",
                            "table": "📊"
                        }
                        view_mode = st.segmented_control(
                            "View mode",
                            options=["compact", "table"],
                            format_func=lambda x: view_icons[x],
                            default="compact",
                            selection_mode="single",
                            key="rule_view_mode",
                            label_visibility="collapsed"
                        )
    
                    if view_mode == "table":
                        # Table view - Create DataFrame
                        import pandas as pd
    
                        # Prepare data for DataFrame
                        table_data = []
                        for idx, rule in enumerate(st.session_state.fuzzy_rules):
                            row = {"Rule": f"R{idx + 1}"}
    
                            # Add input variables
                            for var in st.session_state.input_variables:
                                row[f"IN: {var['name']}"] = rule['antecedents'].get(var['name'], "-")
    
                            # Add output variables
                            for var in st.session_state.output_variables:
                                row[f"OUT: {var['name']}"] = rule['consequents'].get(var['name'], "-")
    
                            table_data.append(row)
    
                        df = pd.DataFrame(table_data)
    
                        # Display dataframe
                        st.dataframe(
                            df,
                            use_container_width=True,
                            hide_index=True
                        )
    
                        st.markdown("<br>", unsafe_allow_html=True)
    
                        # Check if we need to open edit dialog for a specific rule
                        if 'editing_rule_idx' in st.session_state:
                            idx = st.session_state.editing_rule_idx
                            del st.session_state.editing_rule_idx
                            edit_rule_dialog(idx)
                        else:
                            # Actions below table - only show if not editing
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("✏️ Edit Rule", use_container_width=True):
                                    edit_rule_table_dialog()
    
                            with col2:
                                if st.button("🗑️ Delete Rules", use_container_width=True):
                                    delete_rules_table_dialog()
    
                    else:
                        # Compact view - Smaller cards
                        for idx, rule in enumerate(st.session_state.fuzzy_rules):
                            col1, col2 = st.columns([4, 1])
    
                            with col1:
                                # Format antecedents
                                ant_parts = [f"{var} = {term}" for var, term in rule['antecedents'].items()]
                                ant_str = " AND ".join(ant_parts)
    
                                # Format consequents
                                cons_parts = [f"{var} = {term}" for var, term in rule['consequents'].items()]
                                cons_str = ", ".join(cons_parts)
    
                                # Compact display
                                st.markdown(f"""
                                <div style="background: #f8f9fa; padding: 0.5rem 0.75rem; border-radius: 6px; margin-bottom: 0.5rem; border-left: 3px solid #667eea; font-size: 0.85rem;">
                                    <span style="color: #667eea; font-weight: 600;">R{idx + 1}:</span>
                                    <span style="font-weight: 600;">IF</span> {ant_str}
                                    <span style="font-weight: 600;">THEN</span> {cons_str}
                                </div>
                                """, unsafe_allow_html=True)
    
                            with col2:
                                # Action segmented control
                                action_icons = {
                                    "edit": "✏️",
                                    "delete": "🗑️"
                                }
    
                                action = st.segmented_control(
                                    f"Actions for rule {idx}",
                                    options=list(action_icons.keys()),
                                    format_func=lambda x: action_icons[x],
                                    selection_mode="single",
                                    key=f"rule_actions_{idx}",
                                    label_visibility="collapsed"
                                )
    
                                if action == "edit":
                                    edit_rule_dialog(idx)
                                elif action == "delete":
                                    delete_rule_dialog(idx)
                else:
                    st.info("No rules defined yet. Click '➕ Add New Fuzzy Rule' to get started.")
    
        with tab4:
            # st.markdown("### Inference Engine")

            
            # Validate FIS
            try:
                engine = InferenceEngine(active_fis)
                is_valid, message = engine.validate_fis()

                if not is_valid:
                    st.warning(f"⚠️ **FIS not ready**: {message}")
                    st.info("Please complete the configuration in the other tabs before running inference.")
                else:
                    st.success("✓ FIS is ready for inference!")

                    # Input values section
                    st.markdown("##### Set Input Values")

                    # Create input sliders/number inputs
                    input_values = {}
                    cols = st.columns(min(len(active_fis['input_variables']), 3))

                    for idx, var in enumerate(active_fis['input_variables']):
                        with cols[idx % len(cols)]:
                            input_values[var['name']] = st.slider(
                                f"{var['name']}",
                                min_value=float(var['min']),
                                max_value=float(var['max']),
                                value=float((var['min'] + var['max']) / 2),
                                step=float((var['max'] - var['min']) / 100)
                            )

                    # Compute inference button
                    if st.button("⚡ Run Inference", type="primary", use_container_width=True):
                        try:
                            # Compute output
                            result = engine.evaluate(input_values)

                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown("#### 📤 Output Results")

                            # Display results in metric cards
                            result_cols = st.columns(len(result))
                            for idx, (var_name, value) in enumerate(result.items()):
                                with result_cols[idx]:
                                    st.metric(
                                        label=var_name,
                                        value=f"{value:.3f}"
                                    )

                            # Fuzzification visualization
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown("#### 🔍 Fuzzification Analysis")

                            # Show fuzzification for each input
                            for var_name, value in input_values.items():
                                with st.expander(f"📥 {var_name} = {value:.2f}"):
                                    memberships = engine.get_fuzzification(var_name, value)

                                    # Plot membership functions with current value
                                    var_data = next(v for v in active_fis['input_variables'] if v['name'] == var_name)

                                    fig = go.Figure()

                                    # Plot each term
                                    for term in var_data['terms']:
                                        x, y = engine.get_term_membership_curve(var_name, term['name'])
                                        fig.add_trace(go.Scatter(
                                            x=x, y=y,
                                            mode='lines',
                                            name=term['name'],
                                            hovertemplate=f"{term['name']}<br>x=%{{x:.2f}}<br>μ=%{{y:.3f}}<extra></extra>"
                                        ))

                                    # Add vertical line for current value
                                    fig.add_vline(
                                        x=value,
                                        line_dash="dash",
                                        line_color="red",
                                        annotation_text=f"Input: {value:.2f}"
                                    )

                                    fig.update_layout(
                                        title=f"Membership Functions for '{var_name}'",
                                        xaxis_title=var_name,
                                        yaxis_title="Membership Degree (μ)",
                                        hovermode='closest',
                                        height=350
                                    )

                                    st.plotly_chart(fig, use_container_width=True)

                                    # Show membership degrees
                                    st.markdown("**Membership Degrees:**")
                                    mem_cols = st.columns(len(memberships))
                                    for idx, (term_name, degree) in enumerate(memberships.items()):
                                        with mem_cols[idx]:
                                            st.metric(
                                                label=term_name,
                                                value=f"{degree:.3f}",
                                                delta=None
                                            )

                            # Rule activation analysis
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown("#### ⚙️ Rule Activation Analysis")

                            activations = engine.get_rule_activations(input_values)

                            # Sort by activation (highest first)
                            activations.sort(key=lambda x: x[1], reverse=True)

                            # Show top activated rules
                            with st.expander("📜 Active Rules", expanded=True):
                                for rule_idx, activation, rule in activations[:10]:  # Top 10
                                    if activation > 0.01:  # Only show if significantly activated
                                        # Format rule
                                        ant_str = " AND ".join([f"{k}={v}" for k, v in rule['antecedents'].items()])
                                        cons_str = ", ".join([f"{k}={v}" for k, v in rule['consequents'].items()])

                                        # Color based on activation
                                        if activation > 0.7:
                                            color = "#10b981"  # green
                                        elif activation > 0.4:
                                            color = "#f59e0b"  # orange
                                        else:
                                            color = "#6b7280"  # gray

                                        st.markdown(f"""
                                        <div style="background: #f8f9fa; padding: 0.75rem; border-radius: 6px;
                                                    margin-bottom: 0.5rem; border-left: 4px solid {color};">
                                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                                <div>
                                                    <strong>Rule {rule_idx + 1}:</strong>
                                                    IF {ant_str} THEN {cons_str}
                                                </div>
                                                <div style="background: {color}; color: white; padding: 0.25rem 0.75rem;
                                                           border-radius: 12px; font-weight: 600; font-size: 0.875rem;">
                                                    {activation:.3f}
                                                </div>
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)

                                if not any(a[1] > 0.01 for a in activations):
                                    st.info("No rules significantly activated with current inputs")

                        except Exception as e:
                            st.error(f"❌ Error during inference: {str(e)}")

            except Exception as e:
                st.error(f"❌ Error initializing inference engine: {str(e)}")
    
        # Example code
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("##### Example Code")
    
        with st.expander("View example Mamdani system"):
            st.code("""
    import fuzzy_systems as fs
    
    # Create Mamdani system
    system = fs.MamdaniSystem()
    
    # Add input variable
    system.add_input('temperature', (0, 40))
    system.add_term('temperature', 'cold', 'triangular', (0, 0, 20))
    system.add_term('temperature', 'warm', 'triangular', (10, 20, 30))
    system.add_term('temperature', 'hot', 'triangular', (20, 40, 40))
    
    # Add output variable
    system.add_output('fan_speed', (0, 100))
    system.add_term('fan_speed', 'slow', 'triangular', (0, 0, 50))
    system.add_term('fan_speed', 'medium', 'triangular', (25, 50, 75))
    system.add_term('fan_speed', 'fast', 'triangular', (50, 100, 100))
    
    # Add rules
    system.add_rules([
        ('cold', 'slow'),
        ('warm', 'medium'),
        ('hot', 'fast')
    ])
    
    # Evaluate
    result = system.evaluate(temperature=25)
    print(f"Fan speed: {result['fan_speed']:.1f}%")
            """, language='python')
