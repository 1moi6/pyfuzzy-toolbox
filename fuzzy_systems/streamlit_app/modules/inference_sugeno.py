"""
Sugeno-specific output variable dialogs for Inference Systems
Sugeno systems use constant or linear functions for output instead of fuzzy sets
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


@st.dialog("Edit Output Variable", on_dismiss=lambda: reset_all_segment_controls())
def edit_output_variable_dialog(variable_idx, reset_fn=None):
    """Dialog for editing an output variable (Sugeno)"""
    variable = st.session_state.output_variables[variable_idx]

    st.markdown(f"**Editing Output Variable (Sugeno)**")
    st.caption("Sugeno output variables use constant or linear functions")

    new_name = st.text_input("Variable Name", value=variable['name'])

    # For Sugeno, min/max are mainly for visualization purposes
    col1, col2 = st.columns(2)
    with col1:
        new_min = st.number_input("Min (for display)", value=float(variable['min']))
    with col2:
        new_max = st.number_input("Max (for display)", value=float(variable['max']))

    cols = st.columns([0.1,0.8,0.1])
    with cols[1]:
        if st.button("✓ Save Changes", width="stretch", type="primary"):
            if new_name and (new_name == variable['name'] or new_name not in [v['name'] for v in st.session_state.output_variables]):
                st.session_state.output_variables[variable_idx]['name'] = new_name
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
    """Dialog for viewing output variable details (Sugeno)"""
    variable = st.session_state.output_variables[variable_idx]

    st.markdown(f"### {variable['name']}")
    st.markdown(f"**Range (display):** [{variable['min']}, {variable['max']}]")
    st.markdown(f"**Number of Terms:** {len(variable['terms'])}")
    st.caption("Sugeno system - terms are constant or linear functions")

    if variable['terms']:
        st.markdown("---")
        st.markdown("**Output Functions:**")
        for term in variable['terms']:
            with st.container():
                st.markdown(f"**{term['name']}**")
                if term['mf_type'] == 'constant':
                    st.caption(f"Type: Constant | Value: {term['params'][0]}")
                elif term['mf_type'] == 'linear':
                    st.caption(f"Type: Linear | Coefficients: {term['params']}")
    else:
        st.info("No terms defined yet")

    if st.button("Close", width="stretch"):
        close_output_dialog(variable_idx)
        st.rerun()


@st.dialog("Delete Output Variable", on_dismiss=lambda: reset_all_segment_controls())
def delete_output_variable_dialog(variable_idx, reset_fn=None):
    """Dialog for confirming output variable deletion (Sugeno)"""
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
    """Dialog for adding a new output term to Sugeno variable"""

    st.markdown(f"**Adding term to variable: `{variable['name']}`**")
    st.caption("Sugeno terms use constant or linear functions")

    term_name = st.text_input("Term Name", placeholder="e.g., slow, medium, fast")

    # Sugeno-specific function types
    function_icons = {
        "constant": "C",
        "linear": "Σ"
    }

    function_type = st.segmented_control(
        "Function Type",
        options=["constant", "linear"],
        format_func=lambda x: f"{function_icons[x]} {x.title()}",
        default="constant",
        selection_mode="single"
    )

    st.markdown("---")
    st.markdown("**Parameters**")

    if function_type == "constant":
        st.caption("Constant function: z = c")
        constant_value = st.number_input(
            "Constant value (c)",
            value=(variable['min'] + variable['max'])/2,
            help="The constant output value for this term"
        )
        params = (constant_value,)
        mf_type = "constant"

    else:  # linear
        st.caption("Linear function: z = p₁·x₁ + p₂·x₂ + ... + c")
        st.info("Enter coefficients for each input variable, followed by the constant term")

        # Get input variables to create coefficients
        input_vars = st.session_state.input_variables

        if not input_vars:
            st.error("No input variables defined! Add input variables first.")
            return

        # Create coefficient inputs for each input variable
        coefficients = []
        cols = st.columns(min(3, len(input_vars)))
        for i, input_var in enumerate(input_vars):
            with cols[i % len(cols)]:
                coef = st.number_input(
                    f"p_{i+1} ({input_var['name']})",
                    value=0.0,
                    help=f"Coefficient for {input_var['name']}"
                )
                coefficients.append(coef)

        # Constant term
        constant = st.number_input(
            "c (constant)",
            value=0.0,
            help="Constant term in the linear equation"
        )
        coefficients.append(constant)

        params = tuple(coefficients)
        mf_type = "linear"

    cols = st.columns([0.1,0.8,0.1])
    with cols[1]:
        if st.button("Add Output Function", width="stretch", type="primary"):
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
    """Dialog for editing a Sugeno output term"""
    variable = st.session_state.output_variables[variable_idx]
    term = variable['terms'][term_idx]

    st.markdown(f"**Editing term in variable: `{variable['name']}`**")
    st.caption("Sugeno terms use constant or linear functions")

    new_term_name = st.text_input("Term Name", value=term['name'])

    # Sugeno-specific function types
    function_icons = {
        "constant": "C",
        "linear": "Σ"
    }

    function_type = st.segmented_control(
        "Function Type",
        options=["constant", "linear"],
        format_func=lambda x: f"{function_icons[x]} {x.title()}",
        default=term['mf_type'],
        selection_mode="single"
    )

    st.markdown("---")
    st.markdown("**Parameters**")

    current_params = term['params'] if term['mf_type'] == function_type else None

    if function_type == "constant":
        st.caption("Constant function: z = c")
        constant_value = st.number_input(
            "Constant value (c)",
            value=float(current_params[0]) if current_params else (variable['min'] + variable['max'])/2,
            help="The constant output value for this term"
        )
        params = (constant_value,)
        mf_type = "constant"

    else:  # linear
        st.caption("Linear function: z = p₁·x₁ + p₂·x₂ + ... + c")
        st.info("Enter coefficients for each input variable, followed by the constant term")

        # Get input variables to create coefficients
        input_vars = st.session_state.input_variables

        if not input_vars:
            st.error("No input variables defined! Add input variables first.")
            return

        # Create coefficient inputs for each input variable
        coefficients = []
        cols = st.columns(min(3, len(input_vars)))
        for i, input_var in enumerate(input_vars):
            with cols[i % len(cols)]:
                default_val = float(current_params[i]) if current_params and i < len(current_params) else 0.0
                coef = st.number_input(
                    f"p_{i+1} ({input_var['name']})",
                    value=default_val,
                    help=f"Coefficient for {input_var['name']}"
                )
                coefficients.append(coef)

        # Constant term
        default_const = float(current_params[-1]) if current_params and len(current_params) > len(input_vars) else 0.0
        constant = st.number_input(
            "c (constant)",
            value=default_const,
            help="Constant term in the linear equation"
        )
        coefficients.append(constant)

        params = tuple(coefficients)
        mf_type = "linear"

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
    """Dialog for confirming Sugeno output term deletion"""
    variable = st.session_state.output_variables[variable_idx]
    term = variable['terms'][term_idx]

    st.warning(f"Are you sure you want to delete term **'{term['name']}'**?")
    if term['mf_type'] == 'constant':
        st.caption(f"Type: Constant | Value: {term['params'][0]}")
    else:
        st.caption(f"Type: Linear | Coefficients: {term['params']}")

    cols = st.columns([0.1,0.8,0.1])
    with cols[1]:
        if st.button("Yes, Delete!", width="stretch", type="primary"):
            st.session_state.output_variables[variable_idx]['terms'].pop(term_idx)
            close_output_term_dialog(variable_idx, term_idx)
            st.rerun()
