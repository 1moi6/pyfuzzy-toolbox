"""
Mamdani-specific output variable dialogs for Inference Systems
"""

import streamlit as st


def close_output_dialog(variable_idx):
    """Callback to reset output action selection"""
    st.session_state[f"output_actions_{variable_idx}"] = None


def close_output_term_dialog(variable_idx, term_idx):
    """Callback to reset output term action selection"""
    st.session_state[f"output_term_actions_{variable_idx}_{term_idx}"] = None


def reset_all_segment_controls():
    """Reset all segment control states efficiently (for on_dismiss callback)"""
    # Import here to avoid circular import
    from modules.inference import reset_all_segment_controls as _reset_fn
    _reset_fn()


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


@st.dialog("Edit Output Variable", on_dismiss=lambda: reset_all_segment_controls())
def edit_output_variable_dialog(variable_idx, reset_fn=None):
    """Dialog for editing an output variable (Mamdani)"""
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

    cols = st.columns([0.1,0.8,0.1])
    with cols[1]:
        if st.button("✓ Save Changes", width="stretch", type="primary"):
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


@st.dialog("View Output Variable Details", on_dismiss=lambda: reset_all_segment_controls())
def view_output_variable_dialog(variable_idx, reset_fn=None):
    """Dialog for viewing output variable details (Mamdani)"""
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

    if st.button("Close", width="stretch"):
        close_output_dialog(variable_idx)
        st.rerun()


@st.dialog("Delete Output Variable", on_dismiss=lambda: reset_all_segment_controls())
def delete_output_variable_dialog(variable_idx, reset_fn=None):
    """Dialog for confirming output variable deletion (Mamdani)"""
    variable = st.session_state.output_variables[variable_idx]

    st.warning(f"Are you sure you want to delete variable **'{variable['name']}'**?")
    if variable['terms']:
        st.error(f"This will also delete {len(variable['terms'])} term(s)!")

    cols = st.columns([0.1,0.8,0.1])
    with cols[1]:
        if st.button("Yes, Delete!", width="stretch", type="primary"):
            st.session_state.output_variables.pop(variable_idx)
            close_output_dialog(variable_idx)
            st.rerun()


@st.dialog("Add Output Term", on_dismiss=lambda: reset_all_segment_controls())
def add_output_term_dialog(variable_idx, variable, reset_fn=None):
    """Dialog for adding a new fuzzy term to output variable (Mamdani)"""

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

    cols = st.columns([0.1,0.8,0.1])
    with cols[1]:
        if st.button("Add Linguistic Term", width="stretch", type="primary"):
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


@st.dialog("Edit Output Term", on_dismiss=lambda: reset_all_segment_controls())
def edit_output_term_dialog(variable_idx, term_idx, reset_fn=None):
    """Dialog for editing an output fuzzy term (Mamdani)"""
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

    cols = st.columns([0.1,0.8,0.1])
    with cols[1]:
        if st.button("Save Changes", width="stretch", type="primary"):
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


@st.dialog("Delete Output Term", on_dismiss=lambda: reset_all_segment_controls())
def delete_output_term_dialog(variable_idx, term_idx, reset_fn=None):
    """Dialog for confirming output term deletion (Mamdani)"""
    variable = st.session_state.output_variables[variable_idx]
    term = variable['terms'][term_idx]

    st.warning(f"Are you sure you want to delete term **'{term['name']}'**?")
    st.caption(f"Type: {term['mf_type']} | Parameters: {term['params']}")

    cols = st.columns([0.1,0.8,0.1])
    with cols[1]:
        if st.button("Yes, Delete!", width="stretch", type="primary"):
            st.session_state.output_variables[variable_idx]['terms'].pop(term_idx)
            close_output_term_dialog(variable_idx, term_idx)
            st.rerun()
