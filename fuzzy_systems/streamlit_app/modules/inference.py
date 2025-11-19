"""
Inference Systems Module
Interactive interface for creating Mamdani and Sugeno fuzzy systems
"""

import streamlit as st
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Import inference engine
from modules.inference_engine import InferenceEngine
# Import system-specific output variable handlers
import modules.inference_mamdani as mamdani
import modules.inference_sugeno as sugeno
# Import fuzzy_systems for automatic FIS creation
import fuzzy_systems as fs

def close_dialog(variable_idx):
    st.session_state[f"actions_{variable_idx}"] = None

def reset_all_segment_controls():
    """Reset all segment control states efficiently (for on_dismiss callback)"""
    # Reset input variable actions
    if 'input_variables' in st.session_state:
        for idx in range(len(st.session_state.input_variables)):
            # Reset variable-level actions
            key = f"actions_{idx}"
            if key in st.session_state and st.session_state[key] is not None:
                st.session_state[key] = None

            # Reset term-level actions
            var = st.session_state.input_variables[idx]
            if 'terms' in var:
                for t_idx in range(len(var['terms'])):
                    key = f"term_actions_{idx}_{t_idx}"
                    if key in st.session_state and st.session_state[key] is not None:
                        st.session_state[key] = None

    # Reset output variable actions
    if 'output_variables' in st.session_state:
        for idx in range(len(st.session_state.output_variables)):
            key = f"output_actions_{idx}"
            if key in st.session_state and st.session_state[key] is not None:
                st.session_state[key] = None

            var = st.session_state.output_variables[idx]
            if 'terms' in var:
                for t_idx in range(len(var['terms'])):
                    key = f"output_term_actions_{idx}_{t_idx}"
                    if key in st.session_state and st.session_state[key] is not None:
                        st.session_state[key] = None

    # Reset rule actions
    if 'fuzzy_rules' in st.session_state:
        for idx in range(len(st.session_state.fuzzy_rules)):
            key = f"rule_actions_{idx}"
            if key in st.session_state and st.session_state[key] is not None:
                st.session_state[key] = None

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

# ========== OUTPUT VARIABLE DIALOGS (System-Specific) ==========
# These wrappers delegate to either Mamdani or Sugeno modules based on system type

def edit_output_variable_dialog(variable_idx):
    """Dialog for editing an output variable - delegates to system-specific module"""
    fis_type = st.session_state.fis_list[st.session_state.active_fis_idx]['type']

    if 'Sugeno' in fis_type or 'TSK' in fis_type:
        sugeno.edit_output_variable_dialog(variable_idx, reset_all_segment_controls)
    else:
        mamdani.edit_output_variable_dialog(variable_idx, reset_all_segment_controls)


def view_output_variable_dialog(variable_idx):
    """Dialog for viewing output variable details - delegates to system-specific module"""
    fis_type = st.session_state.fis_list[st.session_state.active_fis_idx]['type']

    if 'Sugeno' in fis_type or 'TSK' in fis_type:
        sugeno.view_output_variable_dialog(variable_idx, reset_all_segment_controls)
    else:
        mamdani.view_output_variable_dialog(variable_idx, reset_all_segment_controls)


def delete_output_variable_dialog(variable_idx):
    """Dialog for deleting output variable - delegates to system-specific module"""
    fis_type = st.session_state.fis_list[st.session_state.active_fis_idx]['type']

    if 'Sugeno' in fis_type or 'TSK' in fis_type:
        sugeno.delete_output_variable_dialog(variable_idx, reset_all_segment_controls)
    else:
        mamdani.delete_output_variable_dialog(variable_idx, reset_all_segment_controls)


def add_output_term_dialog(variable_idx, variable):
    """Dialog for adding output term - delegates to system-specific module"""
    fis_type = st.session_state.fis_list[st.session_state.active_fis_idx]['type']

    if 'Sugeno' in fis_type or 'TSK' in fis_type:
        sugeno.add_output_term_dialog(variable_idx, variable, reset_all_segment_controls)
    else:
        mamdani.add_output_term_dialog(variable_idx, variable, reset_all_segment_controls)


def edit_output_term_dialog(variable_idx, term_idx):
    """Dialog for editing output term - delegates to system-specific module"""
    fis_type = st.session_state.fis_list[st.session_state.active_fis_idx]['type']

    if 'Sugeno' in fis_type or 'TSK' in fis_type:
        sugeno.edit_output_term_dialog(variable_idx, term_idx, reset_all_segment_controls)
    else:
        mamdani.edit_output_term_dialog(variable_idx, term_idx, reset_all_segment_controls)


def delete_output_term_dialog(variable_idx, term_idx):
    """Dialog for deleting output term - delegates to system-specific module"""
    fis_type = st.session_state.fis_list[st.session_state.active_fis_idx]['type']

    if 'Sugeno' in fis_type or 'TSK' in fis_type:
        sugeno.delete_output_term_dialog(variable_idx, term_idx, reset_all_segment_controls)
    else:
        mamdani.delete_output_term_dialog(variable_idx, term_idx, reset_all_segment_controls)


@st.dialog("Edit Term", on_dismiss=reset_all_segment_controls)
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

    cols = st.columns([0.1,0.8,0.1])
    with cols[1]:
        if st.button("Save Changes", width="stretch", type="primary"):
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
    # with col2:
    #     if st.button("Cancel", width="stretch"):
    #         close_term_dialog(variable_idx, term_idx)
    #         st.rerun()

@st.dialog("Delete Term", on_dismiss=reset_all_segment_controls)
def delete_term_dialog(variable_idx, term_idx):
    """Dialog for confirming term deletion"""
    variable = st.session_state.input_variables[variable_idx]
    term = variable['terms'][term_idx]

    st.warning(f"Are you sure you want to delete term **'{term['name']}'**?")
    st.caption(f"Type: {term['mf_type']} | Parameters: {term['params']}")

    cols = st.columns([0.1,0.8,0.1])
    with cols[1]:
        if st.button("Yes, Delete!", width="stretch", type="primary"):
            st.session_state.input_variables[variable_idx]['terms'].pop(term_idx)
            close_term_dialog(variable_idx, term_idx)
            st.rerun()
    # with col2:
    #     if st.button("Cancel", width="stretch"):
    #         close_term_dialog(variable_idx, term_idx)
    #         st.rerun()

@st.dialog("Edit Variable", on_dismiss=reset_all_segment_controls)
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

    cols = st.columns([0.1,0.8,0.1])
    with cols[1]:
        if st.button("✓ Save Changes", width="stretch", type="primary"):
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
    # with col2:
    #     if st.button("Cancel", width="stretch", on_click=close_dialog, args=(variable_idx,)):
    #         pass

@st.dialog("View Variable Details", on_dismiss=reset_all_segment_controls)
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

    if st.button("Close", width="stretch"):
        close_dialog(variable_idx)
        st.rerun()

@st.dialog("Delete Variable", on_dismiss=reset_all_segment_controls)
def delete_variable_dialog(variable_idx):
    """Dialog for confirming variable deletion"""
    variable = st.session_state.input_variables[variable_idx]

    st.warning(f"Are you sure you want to delete variable **'{variable['name']}'**?")
    if variable['terms']:
        st.error(f"This will also delete {len(variable['terms'])} term(s)!")

    cols = st.columns([0.1,0.8,0.1])
    with cols[1]:
        if st.button("Yes, Delete!", width="stretch", type="primary"):
            st.session_state.input_variables.pop(variable_idx)
            close_dialog(variable_idx)
            st.rerun()
    # with col2:
    #     if st.button("Cancel", width="stretch", on_click=close_dialog, args=(variable_idx,)):
    #         pass

@st.dialog("Add Fuzzy Term", on_dismiss=reset_all_segment_controls)
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

    cols = st.columns([0.1,0.8,0.1])
    with col1:
        if st.button("Add Term", width="stretch", type="primary"):
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

    # with col2:
    #     if st.button("Cancel", width="stretch", on_click=close_dialog, args=(variable_idx,)):
    #         pass

# ========== RULE DIALOGS ==========

def close_rule_dialog(rule_idx):
    """Callback to reset rule action selection"""
    st.session_state[f"rule_actions_{rule_idx}"] = None

@st.dialog("Edit Fuzzy Rule", on_dismiss=reset_all_segment_controls)
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

        cols= st.columns([0.1,0.8,0.1])
        with cols[1]:
            if st.button("Save Changes", width="stretch", type="primary"):
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

        # with col2:
        #     if st.button("Cancel", width="stretch", on_click=close_rule_dialog, args=(rule_idx,)):
        #         pass

@st.dialog("Delete Fuzzy Rule", on_dismiss=reset_all_segment_controls)
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

    cols = st.columns([0.1,0.8,0.1])
    with cols[1]:
        if st.button("Yes, Delete!", width="stretch", type="primary"):
            st.session_state.fuzzy_rules.pop(rule_idx)
            close_rule_dialog(rule_idx)
            st.rerun()

    # with col2:
    #     if st.button("Cancel", width="stretch", on_click=close_rule_dialog, args=(rule_idx,)):
    #         pass

@st.dialog("Edit Rule (Table View)", on_dismiss=reset_all_segment_controls)
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

    cols = st.columns([0.1,0.8,0.1])
    with cols[1]:
        if st.button("✏️ Edit Selected", width="stretch", type="primary"):
            st.session_state.editing_rule_idx = selected
            st.rerun()

    # with col2:
    #     if st.button("Cancel", width="stretch"):
    #         st.rerun()

@st.dialog("Delete Rules (Table View)", on_dismiss=reset_all_segment_controls)
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

    cols = st.columns([0.1,0.8,0.1])
    with cols[1]:
        if st.button("Delete Selected", width="stretch", type="primary", disabled=len(to_delete)==0):
            # Delete in reverse order to maintain indices
            for idx in sorted(to_delete, reverse=True):
                st.session_state.fuzzy_rules.pop(idx)
            st.success(f"✓ Deleted {len(to_delete)} rule(s)")
            st.rerun()

    # with col2:
    #     if st.button("Cancel", width="stretch"):
    #         st.rerun()

def format_rule_compact(rule):
    """Format a rule in compact form for display"""
    ant_parts = [f"{var}={term}" for var, term in rule['antecedents'].items()]
    cons_parts = [f"{var}={term}" for var, term in rule['consequents'].items()]
    return f"IF {' AND '.join(ant_parts)} THEN {', '.join(cons_parts)}"

# ========== FIS MANAGEMENT DIALOGS ==========

@st.dialog("New Fuzzy Inference System", on_dismiss=reset_all_segment_controls)
def new_fis_dialog():
    """Dialog for creating a new FIS with optional automatic generation"""
    # Get system type from current page context
    system_type = st.session_state.get('inference_system_type', 'Mamdani')

    # Normalize type to include (TSK) suffix for Sugeno
    if system_type == 'Sugeno' or 'Sugeno' in system_type:
        fis_type = 'Sugeno (TSK)'
        type_display = 'Sugeno (TSK)'
    else:
        fis_type = 'Mamdani'
        type_display = 'Mamdani'

    st.markdown(f"**Create a new {type_display} Fuzzy Inference System**")
    st.caption(f"You are on the {type_display} page")

    # FIS Name
    fis_name = st.text_input("FIS Name", placeholder="e.g., Temperature Controller")

    st.markdown("---")

    # Quick Setup toggle
    auto_generate = st.checkbox("⚡ Auto-generate with membership functions", value=False,
                                help="Automatically create variables with evenly distributed membership functions")

    if auto_generate:
        st.markdown("**Quick Setup Configuration**")

        # Number of inputs and outputs
        col1, col2 = st.columns(2)
        with col1:
            n_inputs = st.number_input("Number of Inputs", min_value=1, max_value=10, value=2)
        with col2:
            n_outputs = st.number_input("Number of Outputs", min_value=1, max_value=5, value=1)

        # MFs configuration
        st.markdown("**Membership Functions**")

        mf_mode = st.radio(
            "Configuration mode",
            ["Same for all", "Custom per variable"],
            horizontal=True,
            help="Choose whether all variables have the same MF settings or customize each one"
        )

        if mf_mode == "Same for all":
            col1, col2 = st.columns(2)
            with col1:
                n_mfs = st.number_input("MFs per variable", min_value=2, max_value=7, value=3)
            with col2:
                mf_type = st.selectbox("MF Type", ["triangular", "gaussian", "trapezoidal", "bell"])

            n_mfs_list = None
            mf_type_list = None
        else:
            n_mfs_list = []
            mf_type_list = []

            # Inputs
            with st.expander(f"📥 Input Variables ({n_inputs})"):
                for i in range(n_inputs):
                    col1, col2 = st.columns(2)
                    with col1:
                        n_mf = st.number_input(f"Input {i+1} - MFs", min_value=2, max_value=7, value=3, key=f"input_mf_{i}")
                    with col2:
                        mf_t = st.selectbox(f"Input {i+1} - Type", ["triangular", "gaussian", "trapezoidal", "bell"], key=f"input_type_{i}")
                    n_mfs_list.append(n_mf)
                    mf_type_list.append(mf_t)

            # Outputs
            with st.expander(f"📤 Output Variables ({n_outputs})"):
                for i in range(n_outputs):
                    col1, col2 = st.columns(2)
                    with col1:
                        n_mf = st.number_input(f"Output {i+1} - MFs", min_value=2, max_value=7, value=3, key=f"output_mf_{i}")
                    with col2:
                        if fis_type == 'Mamdani':
                            mf_t = st.selectbox(f"Output {i+1} - Type", ["triangular", "gaussian", "trapezoidal", "bell"], key=f"output_type_{i}")
                        else:
                            mf_t = st.selectbox(f"Output {i+1} - Type", ["constant", "linear"], key=f"output_type_{i}")
                    n_mfs_list.append(n_mf)
                    mf_type_list.append(mf_t)

        # Optional: Universes configuration
        use_custom_universes = st.checkbox("Customize universes of discourse", value=False)

        input_universes = None
        output_universes = None

        if use_custom_universes:
            col1, col2 = st.columns(2)
            with col1:
                st.caption("Inputs")
                input_universes = []
                for i in range(n_inputs):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        u_min = st.number_input(f"Input {i+1} Min", value=0.0, key=f"input_min_{i}")
                    with col_b:
                        u_max = st.number_input(f"Input {i+1} Max", value=100.0, key=f"input_max_{i}")
                    input_universes.append((u_min, u_max))

            with col2:
                st.caption("Outputs")
                output_universes = []
                for i in range(n_outputs):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        u_min = st.number_input(f"Output {i+1} Min", value=0.0, key=f"output_min_{i}")
                    with col_b:
                        u_max = st.number_input(f"Output {i+1} Max", value=100.0, key=f"output_max_{i}")
                    output_universes.append((u_min, u_max))

    st.markdown("---")

    cols = st.columns([0.1,0.8,0.1])
    with cols[1]:
        button_label = "⚡ Generate FIS" if auto_generate else "✓ Create FIS"
        if st.button(button_label, width="stretch", type="primary"):
            if not fis_name:
                st.error("Please enter a FIS name")
                return

            if fis_name in [fis['name'] for fis in st.session_state.fis_list]:
                st.error("A FIS with this name already exists!")
                return

            try:
                if auto_generate:
                    # Auto-generate FIS
                    with st.spinner(f"Generating {type_display} system..."):
                        if fis_type == 'Mamdani':
                            auto_system = fs.MamdaniSystem.create_automatic(
                                n_inputs=n_inputs,
                                n_outputs=n_outputs,
                                n_mfs=n_mfs_list if n_mfs_list else n_mfs,
                                mf_type=mf_type_list if mf_type_list else mf_type,
                                input_universes=input_universes,
                                output_universes=output_universes,
                                name=fis_name
                            )
                        else:
                            auto_system = create_sugeno_automatic(
                                n_inputs=n_inputs,
                                n_outputs=n_outputs,
                                n_mfs=n_mfs_list if n_mfs_list else n_mfs,
                                mf_type=mf_type_list if mf_type_list else mf_type,
                                input_universes=input_universes,
                                output_universes=output_universes,
                                name=fis_name
                            )

                        # Convert to Streamlit format
                        new_fis = convert_fis_to_streamlit_format(auto_system, fis_type)

                        st.session_state.fis_list.append(new_fis)
                        st.session_state.active_fis_idx = len(st.session_state.fis_list) - 1

                        st.success(f"✓ Generated {type_display} FIS: {fis_name}")
                        st.info(f"📊 Created {n_inputs} inputs, {n_outputs} outputs with auto-generated membership functions")
                        st.rerun()
                else:
                    # Create empty FIS
                    new_fis = {
                        'name': fis_name,
                        'type': fis_type,
                        'input_variables': [],
                        'output_variables': [],
                        'fuzzy_rules': []
                    }
                    st.session_state.fis_list.append(new_fis)
                    st.session_state.active_fis_idx = len(st.session_state.fis_list) - 1
                    st.success(f"✓ Created {type_display} FIS: {fis_name}")
                    st.rerun()

            except Exception as e:
                st.error(f"Error creating FIS: {str(e)}")
                st.exception(e)

# Helper functions for automatic FIS generation

def create_sugeno_automatic(n_inputs, n_outputs, n_mfs, mf_type, input_universes, output_universes, name):
    """Create Sugeno system automatically (similar to Mamdani but with constant/linear outputs)"""
    import fuzzy_systems as fs

    # For now, create a basic Sugeno system structure
    # We'll use constant outputs (order 0) by default
    system = fs.SugenoSystem(name=name, order=0)

    # Process parameters similar to Mamdani.create_automatic
    if isinstance(n_mfs, int):
        n_mfs_list = [n_mfs] * (n_inputs + n_outputs)
    else:
        n_mfs_list = n_mfs

    if isinstance(mf_type, str):
        mf_types_list = [mf_type] * n_inputs + ['constant'] * n_outputs
    else:
        mf_types_list = mf_type

    # Universes
    if input_universes is None:
        input_universes_list = [(0.0, 1.0)] * n_inputs
    elif isinstance(input_universes, tuple):
        input_universes_list = [input_universes] * n_inputs
    else:
        input_universes_list = input_universes

    if output_universes is None:
        output_universes_list = [(0.0, 1.0)] * n_outputs
    elif isinstance(output_universes, tuple):
        output_universes_list = [output_universes] * n_outputs
    else:
        output_universes_list = output_universes

    # Add inputs with auto MFs
    input_names = [f"input_{i+1}" for i in range(n_inputs)]
    for i, (name_var, universe) in enumerate(zip(input_names, input_universes_list)):
        system.add_input(name_var, universe)
        system.add_auto_mfs(name_var, n_mfs=n_mfs_list[i], mf_type=mf_types_list[i])

    # Add outputs (for Sugeno, these are just placeholders)
    output_names = [f"output_{i+1}" for i in range(n_outputs)]
    for i, (name_var, universe) in enumerate(zip(output_names, output_universes_list)):
        system.add_output(name_var, universe)

    return system

def convert_fis_to_streamlit_format(auto_system, fis_type):
    """Convert automatically generated FIS to Streamlit session_state format"""

    # Convert input variables
    input_variables = []
    for var_name, var in auto_system.input_variables.items():
        universe = var.universe
        terms = []

        for term_name, term in var.terms.items():
            terms.append({
                'name': term_name,
                'mf_type': term.mf_type,
                'params': term.params
            })

        input_variables.append({
            'name': var_name,
            'min': float(universe[0]),
            'max': float(universe[-1]),
            'terms': terms
        })

    # Convert output variables
    output_variables = []
    for var_name, var in auto_system.output_variables.items():
        universe = var.universe
        terms = []

        # For Sugeno, handle constant/linear outputs differently
        if 'Sugeno' in fis_type or 'TSK' in fis_type:
            # For Sugeno order 0, create constant terms
            # We'll create one constant term per input combination (simplified)
            if len(var.terms) > 0:
                # If system already has terms defined
                for term_name, term in var.terms.items():
                    terms.append({
                        'name': term_name,
                        'mf_type': 'constant',
                        'params': (0.5,)  # Default constant value
                    })
            else:
                # Create a default constant term
                terms.append({
                    'name': 'output_constant',
                    'mf_type': 'constant',
                    'params': (0.5,)
                })
        else:
            # For Mamdani, use the fuzzy sets
            for term_name, term in var.terms.items():
                terms.append({
                    'name': term_name,
                    'mf_type': term.mf_type,
                    'params': term.params
                })

        output_variables.append({
            'name': var_name,
            'min': float(universe[0]),
            'max': float(universe[-1]),
            'terms': terms
        })

    # Convert rules (if any exist)
    fuzzy_rules = []
    # For now, don't auto-generate rules - user will add them manually
    # This could be enhanced later to generate all possible combinations

    return {
        'name': auto_system.name,
        'type': fis_type,
        'input_variables': input_variables,
        'output_variables': output_variables,
        'fuzzy_rules': fuzzy_rules
    }

@st.dialog("Rename FIS", on_dismiss=reset_all_segment_controls)
def rename_fis_dialog():
    """Dialog for renaming the active FIS"""
    active_fis = st.session_state.fis_list[st.session_state.active_fis_idx]

    st.markdown(f"**Renaming: {active_fis['name']}**")

    new_name = st.text_input("New Name", value=active_fis['name'])

    st.markdown("---")

    cols = st.columns([0.1,0.8,0.1])
    with cols[1]:
        if st.button("Rename", width="stretch", type="primary"):
            if new_name and new_name != active_fis['name']:
                # Check if name already exists
                if new_name in [fis['name'] for i, fis in enumerate(st.session_state.fis_list) if i != st.session_state.active_fis_idx]:
                    st.error("A FIS with this name already exists!")
                else:
                    st.session_state.fis_list[st.session_state.active_fis_idx]['name'] = new_name
                    st.success(f"Renamed to: {new_name}")
                    st.rerun()
            elif not new_name:
                st.error("Please enter a name")

    # with col2:
    #     if st.button("Cancel", width="stretch"):
    #         st.rerun()

@st.dialog("Delete FIS", on_dismiss=reset_all_segment_controls)
def delete_fis_dialog():
    """Dialog for deleting the active FIS"""
    active_fis = st.session_state.fis_list[st.session_state.active_fis_idx]

    st.warning(f"Are you sure you want to delete **{active_fis['name']}**?")

    st.markdown("This will permanently delete:")
    st.markdown(f"- {len(active_fis['input_variables'])} input variable(s)")
    st.markdown(f"- {len(active_fis['output_variables'])} output variable(s)")
    st.markdown(f"- {len(active_fis['fuzzy_rules'])} rule(s)")

    st.markdown("---")

    cols = st.columns([0.1,0.8,0.1])
    with cols[1]:
        if st.button("Yes, Delete!", width="stretch", type="primary"):
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

            st.success("FIS deleted")
            st.rerun()

    # with col2:
    #     if st.button("Cancel", width="stretch"):
    #         st.rerun()

@st.dialog("Load FIS from JSON", on_dismiss=reset_all_segment_controls)
def load_fis_dialog():
    """Dialog for loading FIS from exported JSON file"""
    import json

    # Get system type from current page context
    current_page_type = st.session_state.get('inference_system_type', 'Mamdani')
    if current_page_type == 'Sugeno' or 'Sugeno' in current_page_type:
        expected_type = 'Sugeno (TSK)'
        page_display = 'Sugeno (TSK)'
    else:
        expected_type = 'Mamdani'
        page_display = 'Mamdani'

    st.markdown(f"**Upload a {page_display} FIS JSON file**")
    st.caption(f"You are on the {page_display} page - only {page_display} FIS files can be loaded here")
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

            cols = st.columns([0.1,0.8,0.1])
            with cols[1]:
                if st.button("Import FIS", width="stretch", type="primary"):
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

                    # Validate FIS type matches current page
                    if fis_type != expected_type:
                        st.error(f"❌ Type mismatch: This is a **{fis_type}** FIS, but you are on the **{page_display}** page.")
                        st.warning(f"Please navigate to the **{fis_type}** page to load this file.")
                        return

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

            # with col2:
            #     if st.button("Cancel", width="stretch"):
            #         st.rerun()

        except json.JSONDecodeError:
            st.error("❌ Invalid JSON file")
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    else:
        st.info("👆 Upload a JSON file to import a FIS")

@st.dialog("Export FIS to JSON", on_dismiss=reset_all_segment_controls)
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
            width="stretch",
            type="primary"
        )

        st.success("✓ JSON ready for download!")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Copy to Clipboard", width="stretch"):
                # Note: Clipboard API requires user interaction in browser
                st.info("💡 Use the download button or manually copy from the preview above")

        with col2:
            if st.button("Close", width="stretch"):
                st.rerun()

    except Exception as e:
        st.error(f"❌ Error generating JSON: {str(e)}")


def render_rules_visual_matrix(rules, input_variables, output_variables, colormap='blues', fontsize=9):
    """
    Render fuzzy rules as a visual colored matrix using Plotly.

    Parameters:
    -----------
    rules : list
        List of fuzzy rules with 'antecedents' and 'consequents'
    input_variables : list
        List of input variable dictionaries
    output_variables : list
        List of output variable dictionaries
    colormap : str
        Plotly colorscale name (default: 'blues')
    fontsize : int
        Font size for cell text (default: 9)
    """
    import pandas as pd

    if not rules:
        st.info("No rules to display in matrix view.")
        return

    # Prepare data for matrix
    n_rules = len(rules)
    n_inputs = len(input_variables)
    n_outputs = len(output_variables)
    n_cols = n_inputs + n_outputs

    # Create column headers
    col_headers = [f"{v['name']}" for v in input_variables] + \
                  [f"{v['name']}" for v in output_variables]

    # Create row headers
    row_headers = [f"R{i+1}" for i in range(n_rules)]

    # Build matrix data and text
    matrix_data = np.zeros((n_rules, n_cols))
    cell_text = [['' for _ in range(n_cols)] for _ in range(n_rules)]

    for i, rule in enumerate(rules):
        # Process inputs (antecedents)
        for j, var in enumerate(input_variables):
            term = rule['antecedents'].get(var['name'], None)
            if term:
                # Find term index for coloring
                term_idx = next((idx for idx, t in enumerate(var['terms']) if t['name'] == term), 0)
                matrix_data[i, j] = term_idx + 1  # +1 to avoid zero (white)
                cell_text[i][j] = term
            else:
                cell_text[i][j] = '-'

        # Process outputs (consequents)
        for j, var in enumerate(output_variables):
            col_idx = n_inputs + j
            term = rule['consequents'].get(var['name'], None)
            if term:
                term_idx = next((idx for idx, t in enumerate(var['terms']) if t['name'] == term), 0)
                matrix_data[i, col_idx] = term_idx + 1
                cell_text[i][col_idx] = term
            else:
                cell_text[i][col_idx] = '-'

    # Create heatmap with Plotly
    fig = go.Figure(data=go.Heatmap(
        z=matrix_data,
        x=col_headers,
        y=row_headers,
        text=cell_text,
        texttemplate='%{text}',
        textfont={"size": fontsize,'weight':'bold'},
        colorscale=colormap,
        showscale=False,
        hovertemplate='<b>%{y}</b><br>%{x}: %{text}<extra></extra>',
        xgap=2,
        ygap=2
    ))

    # Update layout
    fig.update_layout(
        title={
            'text': f'Fuzzy Rules Matrix ({n_rules} rules)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 14}
        },
        xaxis={
            'side': 'top',
            'tickangle': 0,
            'tickfont': {'size': st.session_state.labels_fontsize,'weight':'bold'}
        },
        yaxis={
            'tickfont': {'size': st.session_state.labels_fontsize},
            'autorange': 'reversed'
        },
        height=max(400, n_rules * 25 + 100),
        margin=dict(l=60, r=20, t=100, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    # # Add statistics
    # col1, col2, col3 = st.columns(3)
    # with col1:
    #     st.metric("Total Rules", n_rules)
    # with col2:
    #     st.metric("Input Variables", n_inputs)
    # with col3:
    #     st.metric("Output Variables", n_outputs)


def render_rules_matrix_view():
    """Render rules as visual colored matrix with customization options"""

    # st.markdown("**Visual Rules Matrix**")
    # st.caption("Color-coded matrix showing antecedents and consequents for each rule")

    # Initialize colormap and font size in session state
    if 'matrix_colormap' not in st.session_state:
        st.session_state.matrix_colormap = 'blues'
    if 'matrix_fontsize' not in st.session_state:
        st.session_state.matrix_fontsize = 9
    if 'labels_fontsize' not in st.session_state:
        st.session_state.labels_fontsize = 10
    # Add color palette and font size selector in popover
    
    st.markdown("")

    # Check if suitable for matrix visualization
    n_inputs = len(st.session_state.input_variables)
    n_outputs = len(st.session_state.output_variables)

    if n_inputs > 5:
        st.info(f"Visual matrix works best with ≤5 inputs. Your system has {n_inputs} inputs.\n\n"
                "Consider using Table View for better readability.")

    try:
        render_rules_visual_matrix(
            st.session_state.fuzzy_rules,
            st.session_state.input_variables,
            st.session_state.output_variables,
            colormap=st.session_state.matrix_colormap,
            fontsize=st.session_state.matrix_fontsize
        )
    except Exception as e:
        st.error(f"❌ Error generating visual matrix: {str(e)}")
        import traceback
        with st.expander("See error details"):
            st.code(traceback.format_exc())

    with st.popover("Control Appearance", use_container_width=True):
        
        # Plotly colorscales organized by category
        sequential_cmaps = [
            'blues', 'greens', 'oranges', 'purples', 'reds', 'greys',
            'ylorbr', 'ylorrd', 'orrd', 'purd', 'rdpu', 'redor',
            'bupu', 'gnbu', 'pubu', 'ylgnbu', 'pubugn', 'bugn', 'ylgn', 'blugrn',
            'teal', 'mint', 'emrld', 'bluyl', 'peach', 'pinkyl', 'purp', 'purpor',
            'aggrnyl', 'agsunset', 'brwnyl', 'burgyl', 'burg', 'oryel', 'oxy',
            'tealgrn', 'darkmint', 'magenta', 'solar', 'amp', 'speed'
        ]

        diverging_cmaps = [
            'rdylbu', 'rdylgn', 'spectral', 'rdbu', 'rdgy',
            'piyg', 'prgn', 'puor', 'brbg', 'picnic', 'portland',
            'armyrose', 'fall', 'geyser', 'temps', 'tealrose',
            'balance', 'curl', 'delta', 'icefire', 'tropic'
        ]

        pastel_soft_cmaps = [
            'peach', 'pinkyl', 'mint', 'teal', 'purp', 'sunset', 'sunsetdark',
            'twilight', 'phase', 'mrybm', 'mygbm', 'earth', 'edge'
        ]

        other_cmaps = [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'turbo', 'rainbow', 'jet', 'hot', 'electric', 'blackbody', 'bluered',
            'thermal', 'ice', 'haline', 'hsv', 'plotly3',
            'deep', 'dense', 'matter', 'algae', 'tempo', 'turbid', 'gray'
        ]

        # All palettes combined for single selector
        all_palettes = {
            '── Sequential ──': None,
            **{name: name for name in sequential_cmaps},
            '── Diverging ──': None,
            **{name: name for name in diverging_cmaps},
            '── Pastel & Soft ──': None,
            **{name: name for name in pastel_soft_cmaps},
            '── Other ──': None,
            **{name: name for name in other_cmaps}
        }

        # Get current colormap index
        palette_list = [k for k in all_palettes.keys() if all_palettes[k] is not None]
        try:
            current_idx = palette_list.index(st.session_state.matrix_colormap)
        except ValueError:
            current_idx = 0

        cols = st.columns(2)
        # Single selectbox with all options
        with cols[0]:
            st.markdown("**Select Color Scheme**")

            st.selectbox(
                "Color palette",
                options=palette_list,
                index=current_idx,
                key='matrix_colormap',
                help="Choose a color scheme for the rule matrix"
            )

            st.markdown("**Labels Size**")
            st.number_input(
                "Labels Font size",
                min_value=6,
                max_value=28,
                value=st.session_state.labels_fontsize,
                step=1,
                key='labels_fontsize',
                help="Adjust the size of label's text")

        with cols[1]:
            # Font size selector
            st.markdown("**Text Size**")
            st.number_input(
                "Cells Font size",
                min_value=6,
                max_value=28,
                value=st.session_state.matrix_fontsize,
                step=1,
                key='matrix_fontsize',
                help="Adjust the size of text in the matrix cells"
            )

@st.cache_data(show_spinner=False)
def create_cached_inference_engine(_fis_data):
    """Create and cache InferenceEngine to avoid recreating on every interaction

    The _fis_data parameter uses underscore prefix to prevent hashing the entire FIS,
    instead we'll use a hash of the structure.
    """
    return InferenceEngine(_fis_data)


def get_fis_hash(fis_data):
    """Generate a hash of FIS structure for caching"""
    import hashlib
    import json
    # Create a simple string representation for hashing
    hash_str = json.dumps({
        'inputs': len(fis_data.get('input_variables', [])),
        'outputs': len(fis_data.get('output_variables', [])),
        'rules': len(fis_data.get('fuzzy_rules', [])),
        'type': fis_data.get('type', ''),
        'name': fis_data.get('name', '')
    }, sort_keys=True)
    return hashlib.md5(hash_str.encode()).hexdigest()


def get_cached_engine(fis_data):
    """Get or create cached InferenceEngine based on FIS hash"""
    fis_hash = get_fis_hash(fis_data)

    # Check if we have a cached engine for this FIS structure
    if 'engine_cache' not in st.session_state:
        st.session_state.engine_cache = {}

    if fis_hash not in st.session_state.engine_cache:
        # Create new engine and cache it
        st.session_state.engine_cache[fis_hash] = InferenceEngine(fis_data)

    return st.session_state.engine_cache[fis_hash]


def invalidate_engine_cache():
    """Clear the InferenceEngine cache - call this when FIS structure changes"""
    if 'engine_cache' in st.session_state:
        st.session_state.engine_cache = {}


def run():
    """Render inference systems page"""

    # Initialize FIS management in session state
    if 'fis_list' not in st.session_state:
        st.session_state.fis_list = []  # Start with empty list
    if 'active_fis_idx' not in st.session_state:
        st.session_state.active_fis_idx = 0

    # Get current page type
    current_page_type = st.session_state.get('inference_system_type', 'Mamdani')
    if current_page_type == 'Sugeno' or 'Sugeno' in current_page_type:
        expected_type = 'Sugeno (TSK)'
    else:
        expected_type = 'Mamdani'

    # Filter FIS list by type - each page shows only its type
    filtered_fis_list = []
    fis_index_map = {}  # Maps filtered index to original index

    for original_idx, fis in enumerate(st.session_state.fis_list):
        if fis['type'] == expected_type:
            filtered_idx = len(filtered_fis_list)
            fis_index_map[filtered_idx] = original_idx
            filtered_fis_list.append(fis)

    # Check if we have any FIS of this type
    has_fis = len(filtered_fis_list) > 0

    # Initialize per-page active index if needed
    page_active_idx_key = f'active_fis_idx_{expected_type}'
    if page_active_idx_key not in st.session_state:
        st.session_state[page_active_idx_key] = 0

    # Get active index for this page
    page_active_idx = st.session_state[page_active_idx_key]

    # Validate and adjust if needed
    if page_active_idx >= len(filtered_fis_list):
        page_active_idx = max(0, len(filtered_fis_list) - 1)
        st.session_state[page_active_idx_key] = page_active_idx

    # Get active FIS (only if exists)
    if has_fis and page_active_idx < len(filtered_fis_list):
        active_fis = filtered_fis_list[page_active_idx]
        # Update global active_fis_idx to point to original index
        st.session_state.active_fis_idx = fis_index_map[page_active_idx]

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

        # New FIS button
        if st.button("➕ New FIS", width="stretch"):
            new_fis_dialog()

        if not has_fis:
            if st.button("📂 Load FIS", width="stretch"):
                load_fis_dialog()
            

        # Only show FIS management controls if there's at least one FIS
        if has_fis:
            # FIS selector (only show if more than one FIS of this type)
            if len(filtered_fis_list) > 1:
                # Use only name since all FIS in filtered list are same type
                fis_names = [fis['name'] for fis in filtered_fis_list]
                selected_filtered_idx = st.selectbox(
                    "Select FIS",
                    range(len(filtered_fis_list)),
                    format_func=lambda x: fis_names[x],
                    index=page_active_idx,
                    label_visibility="collapsed"
                )

                if selected_filtered_idx != page_active_idx:
                    # Update page-specific active index
                    st.session_state[page_active_idx_key] = selected_filtered_idx
                    # Map to original index in fis_list
                    st.session_state.active_fis_idx = fis_index_map[selected_filtered_idx]
                    st.rerun()
            else:
                # Show current FIS name (type is shown in System Info below)
                st.markdown(f"<div style='text-align: center; padding: 0.5rem; background: #f8f9fa; border-radius: 6px; margin-bottom: 0.5rem;'><strong>{active_fis['name']}</strong></div>", unsafe_allow_html=True)

            # FIS actions
            
            

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


            if st.button("Rename FIS", width="stretch"):
                rename_fis_dialog()
            # Disable delete if this is the last FIS overall OR last of this type
            if st.button("Delete FIS", width="stretch", disabled=(len(st.session_state.fis_list)==1 or len(filtered_fis_list)==1)):
                delete_fis_dialog()

            if st.button("Load FIS", width="stretch"):
                load_fis_dialog()
            # Save/Export button
            if st.button("Export JSON", width="stretch"):
                export_fis_dialog()

            # Reset button
            if st.button("Reset FIS", width="stretch"):
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
            "**Input Variables**",
            "**Output Variables**",
            "**Fuzzy Rules**",
            "**Inference**"
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
    
                if st.button("✓ Add Variable", width="stretch", key="add_var_btn", on_click=add_variable):
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
                    with st.expander(f"**:blue-badge[{variable['name']}]** - Range: [{variable['min']}, {variable['max']}] | Terms: {len(variable['terms'])}" ):
                        # st.markdown(f"""
                        # <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 4px solid #667eea;">
                        #     <h4 style="margin: 0 0 0.5rem 0; color: #667eea;">{variable['name']}</h4>
                        #     <p style="margin: 0; color: #6b7280; font-size: 0.875rem;">
                        #         Range: [{variable['min']}, {variable['max']}] | Terms: {len(variable['terms'])}
                        #     </p>
                        # </div>
                        # """, unsafe_allow_html=True)
    
                        # Action buttons using icons
                        action_icons = {
                            # "view": "👁️ View",
                            "edit": "Edit Linguistic Variable",
                            "add_term": "Add Term to Variable",
                            "delete": "Delete Linguistic Variable"
                        }
    
                        action = st.segmented_control(
                            f"Actions for {variable['name']}",
                            options=list(action_icons.keys()),
                            format_func=lambda x:f"**{action_icons[x]}**",
                            selection_mode="single",
                            key=f"actions_{idx}",
                            label_visibility="collapsed",
                            width = 'stretch'
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
                            # with st.popover(f"📋 Terms ({len(variable['terms'])})"):
                                # var_data = next(v for v in active_fis['input_variables'] if v['name'] == var_name)
                            engine = get_cached_engine(active_fis)
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

                            st.plotly_chart(fig, width="stretch",key=f"input_chart_for_{variable['name']}")
                            
                            for t_idx, term in enumerate(variable['terms']):
                                col_t1, col_t2 = st.columns([3, 1])
                                with col_t1:
                                    st.markdown(f"""
                                <div style="background: #f8f9fa; padding: 0.5rem 0.75rem; border-radius: 6px; margin-bottom: 0.5rem; border-left: 3px solid #667eea; font-size: 0.85rem;">
                                    <span style="font-weight: 600;">{term['name']}</span>
                                    <span style="font-weight: 300;"> - </span>
                                    <span style="font-weight: 300;">{term['mf_type']} - [{', '.join([str(round(a, 2)) for a in term['params']])}]</span>
                                </div>
                                """, unsafe_allow_html=True)
                                    # st.markdown(f"**{term['name']}**")
                                    # st.caption(f"`{term['mf_type']}` {term['params']}")
                                with col_t2:
                                    term_action_icons = {
                                        "edit": "Edit Term",
                                        "delete": "Delete Term"
                                    }

                                    term_action = st.segmented_control(
                                        f"Actions for term {term['name']}",
                                        options=list(term_action_icons.keys()),
                                        format_func=lambda x: term_action_icons[x],
                                        selection_mode="single",
                                        key=f"term_actions_{idx}_{t_idx}",
                                        label_visibility="collapsed",
                                        width = 'stretch'
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
    
                if st.button("✓ Add Variable", width="stretch", key="add_output_var_btn", on_click=add_output_variable):
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
                    with st.expander(f"**:green-badge[{variable['name']}]** - Range: [{variable['min']}, {variable['max']}] | Terms: {len(variable['terms'])}" ):
                        # st.markdown(f"""
                        # <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 4px solid #10b981;">
                        #     <h4 style="margin: 0 0 0.5rem 0; color: #10b981;">{variable['name']}</h4>
                        #     <p style="margin: 0; color: #6b7280; font-size: 0.875rem;">
                        #         Range: [{variable['min']}, {variable['max']}] | Terms: {len(variable['terms'])}
                        #     </p>
                        # </div>
                        # """, unsafe_allow_html=True)
    
                        # Action buttons using icons
                        action_icons = {
                            # "view": "👁️ View",
                            "edit": "Edit Linguistic Variable",
                            "add_term": "Add Term",
                            "delete": "Delete Linguistic Variable"
                        }
    
                        action = st.segmented_control(
                            f"Actions for {variable['name']}",
                            options=list(action_icons.keys()),
                            format_func=lambda x: f"**{action_icons[x]}**",
                            selection_mode="single",
                            key=f"output_actions_{idx}",
                            label_visibility="collapsed",
                            width = 'stretch'
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
                            # with st.expander(f"📋 Terms ({len(variable['terms'])})", expanded=False):

                            # Only plot membership functions for Mamdani systems
                            # Sugeno systems use constant/linear functions instead
                            if 'Mamdani' in system_type:
                                engine = get_cached_engine(active_fis)
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

                                st.plotly_chart(fig, width="stretch",key=f"output_chart_for_{variable['name']}")
                            else:
                                # For Sugeno systems, show a table of output functions
                                st.markdown("**Output Functions:**")
                                st.caption("Sugeno systems use constant or linear functions instead of fuzzy sets")


                            for t_idx, term in enumerate(variable['terms']):
                                col_t1, col_t2 = st.columns([3, 1])
                                with col_t1:
                                    # Format term info based on system type
                                    if 'Sugeno' in system_type or 'TSK' in system_type:
                                        if term['mf_type'] == 'constant':
                                            term_info = f"Constant = {term['params'][0]:.2f}"
                                        else:  # linear
                                            input_names = [v['name'] for v in st.session_state.input_variables]
                                            coefs_str = ' + '.join([f"{term['params'][i]:.2f}·{input_names[i]}" for i in range(len(input_names))])
                                            term_info = f"Linear: {coefs_str} + {term['params'][-1]:.2f}"
                                    else:
                                        term_info = f"{term['mf_type']} - [{', '.join([str(round(a, 2)) for a in term['params']])}]"

                                    st.markdown(f"""
                                <div style="background: #f8f9fa; padding: 0.5rem 0.75rem; border-radius: 6px; margin-bottom: 0.5rem; border-left: 3px solid #059669; font-size: 0.85rem;">
                                    <span style="font-weight: 600;">{term['name']}</span>
                                    <span style="font-weight: 300;"> - </span>
                                    <span style="font-weight: 300;">{term_info}</span>
                                </div>
                                """, unsafe_allow_html=True)
                                    # st.markdown(f"**{term['name']}**")
                                    # st.caption(f"`{term['mf_type']}` {term['params']}")
                                with col_t2:
                                    term_action_icons = {
                                        "edit": "Edit Term",
                                        "delete": "Delete Term"
                                    }

                                    term_action = st.segmented_control(
                                        f"Actions for term {term['name']}",
                                        options=list(term_action_icons.keys()),
                                        format_func=lambda x: term_action_icons[x],
                                        selection_mode="single",
                                        key=f"output_term_actions_{idx}_{t_idx}",
                                        label_visibility="collapsed",
                                        width = 'stretch'
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

                    # Method selection
                    method_icons = {
                        "pills": "Pills",
                        "text": "Text",
                        "csv": "CSV"
                    }
                    add_method = st.segmented_control(
                        "Rule addition method",
                        options=["pills", "text", "csv"],
                        format_func=lambda x: method_icons[x],
                        default="pills",
                        selection_mode="single",
                        key=f"add_rule_method_{st.session_state.new_rule_counter}",
                        label_visibility="collapsed",
                        width='stretch'
                    )

                    st.markdown("<br>", unsafe_allow_html=True)

                    if add_method == "pills":  # pills method
                        # New method with pills in columns (single selection)
                        # IF part (antecedents)
                        st.markdown("**IF** (Antecedents)")

                        antecedents = {}

                        # Create columns for each input variable
                        if st.session_state.input_variables:
                            num_inputs = len(st.session_state.input_variables)
                            input_cols = st.columns(num_inputs)

                            for idx, var in enumerate(st.session_state.input_variables):
                                if var['terms']:
                                    with input_cols[idx]:
                                        st.markdown(f"**{var['name']}**")
                                        # Pills for each term
                                        term_names = [term['name'] for term in var['terms']]
                                        selected_term = st.pills(
                                            f"Select term for {var['name']}",
                                            options=term_names,
                                            selection_mode="single",
                                            key=f"ant_pills_{var['name']}_{st.session_state.new_rule_counter}",
                                            label_visibility="collapsed"
                                        )
                                        if selected_term:
                                            antecedents[var['name']] = selected_term

                        st.markdown("---")

                        # THEN part (consequents)
                        st.markdown("**THEN** (Consequents)")

                        consequents = {}

                        # Create columns for each output variable
                        if st.session_state.output_variables:
                            num_outputs = len(st.session_state.output_variables)
                            output_cols = st.columns(num_outputs)

                            for idx, var in enumerate(st.session_state.output_variables):
                                if var['terms']:
                                    with output_cols[idx]:
                                        st.markdown(f"**{var['name']}**")
                                        # Pills for each term (no "—" option - must select one)
                                        term_names = [term['name'] for term in var['terms']]
                                        selected_term = st.pills(
                                            f"Select term for {var['name']}",
                                            options=term_names,
                                            selection_mode="single",
                                            key=f"cons_pills_{var['name']}_{st.session_state.new_rule_counter}",
                                            label_visibility="collapsed"
                                        )
                                        # Pills always return a value when selected, or None if nothing selected
                                        if selected_term:
                                            consequents[var['name']] = selected_term

                        st.markdown("---")

                        # Validate rule: ALL antecedents and ALL consequents must be selected
                        is_valid = True
                        validation_messages = []

                        # Check if ALL input variables have a term selected
                        for var in st.session_state.input_variables:
                            if var['name'] not in antecedents:
                                is_valid = False
                                validation_messages.append(f"Select a term for input '{var['name']}'")

                        # Check if all output variables have a term selected
                        for var in st.session_state.output_variables:
                            if var['name'] not in consequents:
                                is_valid = False
                                validation_messages.append(f"Select a term for output '{var['name']}'")

                        # Show validation messages
                        if not is_valid:
                            for msg in validation_messages:
                                st.info(f"ℹ️ {msg}")

                        # Debug: Show current rule being built
                        if antecedents or consequents:
                            with st.expander("🔍 Debug: Current rule", expanded=False):
                                st.write("**Antecedents:**", antecedents)
                                st.write("**Consequents:**", consequents)
                                st.write("**Existing rules:**")
                                for idx, r in enumerate(st.session_state.fuzzy_rules):
                                    st.write(f"Rule {idx+1}:", r)

                        # Check for duplicate and conflicting rules before showing button
                        if is_valid and antecedents and consequents:
                            rule_duplicate = False
                            rule_conflict = False
                            conflict_rule_num = None
                            conflict_rule = None

                            for idx, r in enumerate(st.session_state.fuzzy_rules):
                                # Check if antecedents are the same
                                if r['antecedents'] == antecedents:
                                    # Check if consequents are also the same (duplicate)
                                    if r['consequents'] == consequents:
                                        rule_duplicate = True
                                        conflict_rule_num = idx + 1
                                        conflict_rule = r
                                        break
                                    else:
                                        # Same antecedents but different consequents (conflict!)
                                        rule_conflict = True
                                        conflict_rule_num = idx + 1
                                        conflict_rule = r
                                        break

                            if rule_duplicate:
                                st.warning(f"⚠️ Rule already exists (R{conflict_rule_num})")
                                is_valid = False
                            elif rule_conflict:
                                # Format the conflicting rule
                                ant_parts = [f"{var}={term}" for var, term in conflict_rule['antecedents'].items()]
                                cons_parts = [f"{var}={term}" for var, term in conflict_rule['consequents'].items()]
                                rule_str = f"IF {' AND '.join(ant_parts)} THEN {', '.join(cons_parts)}"

                                st.error(f"🚫 Conflicting with R{conflict_rule_num}: {rule_str}")
                                is_valid = False

                        # Show button only if valid and not duplicate/conflicting
                        if is_valid and antecedents and consequents:
                            if st.button("✓ Add Rule", type="primary", width="stretch"):
                                # Double-check before adding (for safety)
                                has_duplicate = False
                                has_conflict = False

                                for r in st.session_state.fuzzy_rules:
                                    if r['antecedents'] == antecedents:
                                        if r['consequents'] == consequents:
                                            has_duplicate = True
                                            break
                                        else:
                                            has_conflict = True
                                            break

                                if has_duplicate:
                                    st.error("⚠️ Rule already exists")
                                elif has_conflict:
                                    st.error("🚫 Rule conflicts with existing rule")
                                else:
                                    st.session_state.fuzzy_rules.append({
                                        'antecedents': antecedents,
                                        'consequents': consequents
                                    })
                                    # Increment counter to reset form
                                    st.session_state.new_rule_counter += 1
                                    st.success(f"✓ Rule added! Total rules: {len(st.session_state.fuzzy_rules)}")
                                    st.rerun()

                    elif add_method == "text":
                        # Text-based method: paste comma-separated values
                        st.markdown("**Paste Rules (comma-separated)**")

                        # Show format instructions
                        n_inputs = len(st.session_state.input_variables)
                        n_outputs = len(st.session_state.output_variables)

                        # Get input and output variable names
                        input_names = [var['name'] for var in st.session_state.input_variables]
                        output_names = [var['name'] for var in st.session_state.output_variables]

                        # Create example format
                        example_terms = []
                        for var in st.session_state.input_variables:
                            if var['terms']:
                                example_terms.append(var['terms'][0]['name'])
                        for var in st.session_state.output_variables:
                            if var['terms']:
                                example_terms.append(var['terms'][0]['name'])

                        st.info(f"""
**Format**: Each line = one rule with {n_inputs + n_outputs} values (comma-separated)

**Order**: {', '.join(input_names + output_names)}

**Values**: Use term names or indices (1, 2, 3...)

**Example**:
```
{','.join(example_terms)}
```
                        """)

                        # Text area for pasting rules
                        rules_text = st.text_area(
                            "Paste rules here (one per line)",
                            height=150,
                            key=f"rules_text_{st.session_state.new_rule_counter}",
                            placeholder="B,B,MB_p,MB_n\nMB,MA,B_p,MA_n"
                        )

                        if rules_text:
                            # Parse the text
                            lines = [line.strip() for line in rules_text.split('\n') if line.strip()]
                            parsed_rules = []
                            errors = []

                            for line_num, line in enumerate(lines, 1):
                                values = [v.strip() for v in line.split(',')]

                                # Check if we have the correct number of values
                                if len(values) != n_inputs + n_outputs:
                                    errors.append(f"Line {line_num}: Expected {n_inputs + n_outputs} values, got {len(values)}")
                                    continue

                                # Parse antecedents
                                antecedents = {}
                                for i, var in enumerate(st.session_state.input_variables):
                                    value = values[i]

                                    # Check if it's an index or term name
                                    if value.isdigit():
                                        idx = int(value)
                                        if 1 <= idx <= len(var['terms']):
                                            antecedents[var['name']] = var['terms'][idx - 1]['name']
                                        else:
                                            errors.append(f"Line {line_num}: Invalid index {idx} for '{var['name']}' (range 1-{len(var['terms'])})")
                                            break
                                    else:
                                        # Check if term name exists
                                        term_names = [term['name'] for term in var['terms']]
                                        if value in term_names:
                                            antecedents[var['name']] = value
                                        else:
                                            errors.append(f"Line {line_num}: Unknown term '{value}' for '{var['name']}'")
                                            break

                                if len(antecedents) != n_inputs:
                                    continue

                                # Parse consequents
                                consequents = {}
                                for i, var in enumerate(st.session_state.output_variables):
                                    value = values[n_inputs + i]

                                    # Check if it's an index or term name
                                    if value.isdigit():
                                        idx = int(value)
                                        if 1 <= idx <= len(var['terms']):
                                            consequents[var['name']] = var['terms'][idx - 1]['name']
                                        else:
                                            errors.append(f"Line {line_num}: Invalid index {idx} for '{var['name']}' (range 1-{len(var['terms'])})")
                                            break
                                    else:
                                        # Check if term name exists
                                        term_names = [term['name'] for term in var['terms']]
                                        if value in term_names:
                                            consequents[var['name']] = value
                                        else:
                                            errors.append(f"Line {line_num}: Unknown term '{value}' for '{var['name']}'")
                                            break

                                if len(consequents) != n_outputs:
                                    continue

                                parsed_rules.append({
                                    'antecedents': antecedents,
                                    'consequents': consequents,
                                    'line': line_num
                                })

                            # Show errors if any
                            if errors:
                                for error in errors:
                                    st.error(f"❌ {error}")

                            # Show parsed rules
                            if parsed_rules:
                                st.success(f"✓ Parsed {len(parsed_rules)} rule(s)")

                                # Check for duplicates and conflicts
                                rules_to_add = []
                                for rule in parsed_rules:
                                    is_duplicate = False
                                    is_conflict = False

                                    for existing_rule in st.session_state.fuzzy_rules:
                                        if existing_rule['antecedents'] == rule['antecedents']:
                                            if existing_rule['consequents'] == rule['consequents']:
                                                is_duplicate = True
                                                st.warning(f"⚠️ Line {rule['line']}: Rule already exists (skipped)")
                                                break
                                            else:
                                                is_conflict = True
                                                st.error(f"🚫 Line {rule['line']}: Conflicts with existing rule (skipped)")
                                                break

                                    if not is_duplicate and not is_conflict:
                                        rules_to_add.append(rule)

                                # Show add button if there are valid rules
                                if rules_to_add:
                                    st.markdown("---")
                                    if st.button(f"✓ Add {len(rules_to_add)} Rule(s)", type="primary", width="stretch"):
                                        for rule in rules_to_add:
                                            st.session_state.fuzzy_rules.append({
                                                'antecedents': rule['antecedents'],
                                                'consequents': rule['consequents']
                                            })
                                        st.session_state.new_rule_counter += 1
                                        st.success(f"✓ Added {len(rules_to_add)} rule(s)! Total rules: {len(st.session_state.fuzzy_rules)}")
                                        st.rerun()

                    elif add_method == "csv":
                        # CSV file upload method
                        st.markdown("**Upload CSV File**")

                        # Show format instructions
                        n_inputs = len(st.session_state.input_variables)
                        n_outputs = len(st.session_state.output_variables)

                        # Get input and output variable names
                        input_names = [var['name'] for var in st.session_state.input_variables]
                        output_names = [var['name'] for var in st.session_state.output_variables]

                        # Create example format
                        example_terms = []
                        for var in st.session_state.input_variables:
                            if var['terms']:
                                example_terms.append(var['terms'][0]['name'])
                        for var in st.session_state.output_variables:
                            if var['terms']:
                                example_terms.append(var['terms'][0]['name'])

                        st.info(f"""
**Format**: CSV file with {n_inputs + n_outputs} columns (no header)

**Column order**: {', '.join(input_names + output_names)}

**Values**: Use term names or indices (1, 2, 3...)

**Separator**: Comma (,) or semicolon (;) - auto-detected

**Example**:
```
{','.join(example_terms)}
{';'.join(example_terms[:2] + example_terms[2:])}
```
                        """)

                        # File uploader
                        uploaded_file = st.file_uploader(
                            "Choose a CSV file",
                            type=['csv', 'txt'],
                            key=f"csv_upload_{st.session_state.new_rule_counter}",
                            accept_multiple_files=False
                        )

                        if uploaded_file is not None:
                            # Read file content
                            content = uploaded_file.read().decode('utf-8')

                            # Detect separator automatically
                            import csv
                            try:
                                # Try to detect the delimiter
                                sample = content[:1024]  # Use first 1KB to detect
                                sniffer = csv.Sniffer()
                                detected_delimiter = sniffer.sniff(sample).delimiter
                                st.success(f"✓ Detected separator: `{detected_delimiter}`")
                            except:
                                # If detection fails, try both , and ;
                                if ',' in content and ';' not in content:
                                    detected_delimiter = ','
                                elif ';' in content and ',' not in content:
                                    detected_delimiter = ';'
                                else:
                                    # Count occurrences
                                    comma_count = content.count(',')
                                    semicolon_count = content.count(';')
                                    detected_delimiter = ',' if comma_count > semicolon_count else ';'
                                st.info(f"ℹ️ Using separator: `{detected_delimiter}`")

                            # Parse the CSV
                            lines = [line.strip() for line in content.split('\n') if line.strip()]
                            parsed_rules = []
                            errors = []

                            for line_num, line in enumerate(lines, 1):
                                values = [v.strip() for v in line.split(detected_delimiter)]

                                # Check if we have the correct number of values
                                if len(values) != n_inputs + n_outputs:
                                    errors.append(f"Line {line_num}: Expected {n_inputs + n_outputs} values, got {len(values)}")
                                    continue

                                # Parse antecedents
                                antecedents = {}
                                for i, var in enumerate(st.session_state.input_variables):
                                    value = values[i]

                                    # Check if it's an index or term name
                                    if value.isdigit():
                                        idx = int(value)
                                        if 1 <= idx <= len(var['terms']):
                                            antecedents[var['name']] = var['terms'][idx - 1]['name']
                                        else:
                                            errors.append(f"Line {line_num}: Invalid index {idx} for '{var['name']}' (range 1-{len(var['terms'])})")
                                            break
                                    else:
                                        # Check if term name exists
                                        term_names = [term['name'] for term in var['terms']]
                                        if value in term_names:
                                            antecedents[var['name']] = value
                                        else:
                                            errors.append(f"Line {line_num}: Unknown term '{value}' for '{var['name']}'")
                                            break

                                if len(antecedents) != n_inputs:
                                    continue

                                # Parse consequents
                                consequents = {}
                                for i, var in enumerate(st.session_state.output_variables):
                                    value = values[n_inputs + i]

                                    # Check if it's an index or term name
                                    if value.isdigit():
                                        idx = int(value)
                                        if 1 <= idx <= len(var['terms']):
                                            consequents[var['name']] = var['terms'][idx - 1]['name']
                                        else:
                                            errors.append(f"Line {line_num}: Invalid index {idx} for '{var['name']}' (range 1-{len(var['terms'])})")
                                            break
                                    else:
                                        # Check if term name exists
                                        term_names = [term['name'] for term in var['terms']]
                                        if value in term_names:
                                            consequents[var['name']] = value
                                        else:
                                            errors.append(f"Line {line_num}: Unknown term '{value}' for '{var['name']}'")
                                            break

                                if len(consequents) != n_outputs:
                                    continue

                                parsed_rules.append({
                                    'antecedents': antecedents,
                                    'consequents': consequents,
                                    'line': line_num
                                })

                            # Show errors if any
                            if errors:
                                for error in errors[:5]:  # Show first 5 errors
                                    st.error(f"❌ {error}")
                                if len(errors) > 5:
                                    st.error(f"❌ ... and {len(errors) - 5} more error(s)")

                            # Show parsed rules
                            if parsed_rules:
                                st.success(f"✓ Parsed {len(parsed_rules)} rule(s) from {uploaded_file.name}")

                                # Check for duplicates and conflicts
                                rules_to_add = []
                                for rule in parsed_rules:
                                    is_duplicate = False
                                    is_conflict = False

                                    for existing_rule in st.session_state.fuzzy_rules:
                                        if existing_rule['antecedents'] == rule['antecedents']:
                                            if existing_rule['consequents'] == rule['consequents']:
                                                is_duplicate = True
                                                st.warning(f"⚠️ Line {rule['line']}: Rule already exists (skipped)")
                                                break
                                            else:
                                                is_conflict = True
                                                st.error(f"🚫 Line {rule['line']}: Conflicts with existing rule (skipped)")
                                                break

                                    if not is_duplicate and not is_conflict:
                                        rules_to_add.append(rule)

                                # Show add button if there are valid rules
                                if rules_to_add:
                                    st.markdown("---")
                                    if st.button(f"✓ Add {len(rules_to_add)} Rule(s)", type="primary", width="stretch"):
                                        for rule in rules_to_add:
                                            st.session_state.fuzzy_rules.append({
                                                'antecedents': rule['antecedents'],
                                                'consequents': rule['consequents']
                                            })
                                        st.session_state.new_rule_counter += 1
                                        st.success(f"✓ Added {len(rules_to_add)} rule(s)! Total rules: {len(st.session_state.fuzzy_rules)}")
                                        st.rerun()

                st.markdown("<br>", unsafe_allow_html=True)
    
                # Display existing rules
                if st.session_state.fuzzy_rules:
                    col_header1, col_header2 = st.columns([3, 2])
                    with col_header1:
                        st.markdown(f"**Fuzzy Rules ({len(st.session_state.fuzzy_rules)})**")
                    with col_header2:
                        # Toggle between view modes
                        view_icons = {
                            "compact": "Rules",
                            "table": "Table",
                            "matrix": "Matrix"
                        }
                        view_mode = st.segmented_control(
                            "View mode",
                            options=["compact", "table", "matrix"],
                            format_func=lambda x: view_icons[x],
                            default="compact",
                            selection_mode="single",
                            key="rule_view_mode",
                            label_visibility="collapsed",
                            width = 'stretch'
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
                            width="stretch",
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
                                if st.button("✏️ Edit Rule", width="stretch"):
                                    edit_rule_table_dialog()
    
                            with col2:
                                if st.button("🗑️ Delete Rules", width="stretch"):
                                    delete_rules_table_dialog()

                    elif view_mode == "matrix":
                        # Matrix view - Visual colored matrix
                        render_rules_matrix_view()

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
                                    "edit": "Edit Rule",
                                    "delete": "Delete Rule"
                                }
    
                                action = st.segmented_control(
                                    f"Actions for rule {idx}",
                                    options=list(action_icons.keys()),
                                    format_func=lambda x: action_icons[x],
                                    selection_mode="single",
                                    key=f"rule_actions_{idx}",
                                    label_visibility="collapsed",
                                    width = 'stretch'
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
                engine = get_cached_engine(active_fis)
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
                    if st.button("⚡ Run Inference", type="primary", width="stretch"):
                        try:
                            # Compute output
                            result = engine.evaluate(input_values)

                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown("##### Output Results")

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
                            st.markdown("##### Fuzzification Analysis")

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

                                    st.plotly_chart(fig, width="stretch")

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

                            # Aggregated output visualization (only for Mamdani)
                            if 'Mamdani' in active_fis['type']:
                                st.markdown("<br>", unsafe_allow_html=True)
                                st.markdown("##### Aggregated Output Sets")

                                for var in active_fis['output_variables']:
                                    var_name = var['name']

                                    with st.expander(f"📤 {var_name} (defuzzified: {result[var_name]:.3f})"):
                                        try:
                                            # Get aggregated output
                                            x, aggregated = engine.get_aggregated_output(var_name, input_values)

                                            fig = go.Figure()

                                            # Plot individual membership functions (lighter)
                                            for term in var['terms']:
                                                term_x, term_y = engine.get_term_membership_curve(var_name, term['name'])
                                                fig.add_trace(go.Scatter(
                                                    x=term_x, y=term_y,
                                                    mode='lines',
                                                    name=term['name'],
                                                    line=dict(dash='dot', width=1),
                                                    opacity=0.4,
                                                    hovertemplate=f"{term['name']}<br>x=%{{x:.2f}}<br>μ=%{{y:.3f}}<extra></extra>"
                                                ))

                                            # Plot aggregated output (highlighted)
                                            fig.add_trace(go.Scatter(
                                                x=x, y=aggregated,
                                                mode='lines',
                                                name='Aggregated',
                                                line=dict(color='red', width=3),
                                                fill='tozeroy',
                                                fillcolor='rgba(255, 0, 0, 0.2)',
                                                hovertemplate="Aggregated<br>x=%{x:.2f}<br>μ=%{y:.3f}<extra></extra>"
                                            ))

                                            # Add vertical line for defuzzified output
                                            fig.add_vline(
                                                x=result[var_name],
                                                line_dash="dash",
                                                line_color="blue",
                                                line_width=2,
                                                annotation_text=f"Output: {result[var_name]:.3f}",
                                                annotation_position="top"
                                            )

                                            fig.update_layout(
                                                title=f"Aggregated Fuzzy Set for '{var_name}'",
                                                xaxis_title=var_name,
                                                yaxis_title="Membership Degree (μ)",
                                                hovermode='closest',
                                                height=400,
                                                showlegend=True
                                            )

                                            st.plotly_chart(fig, use_container_width=True)

                                            st.markdown(f"""
                                            **Interpretation:**
                                            - The **red filled area** shows the aggregated fuzzy output from all activated rules
                                            - The **blue vertical line** at {result[var_name]:.3f} shows the crisp defuzzified value
                                            - Dotted lines show the individual output membership functions
                                            """)

                                        except Exception as e:
                                            st.error(f"Error computing aggregated output: {str(e)}")

                            # Rule activation analysis
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown("##### Rule Activation Analysis")

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
