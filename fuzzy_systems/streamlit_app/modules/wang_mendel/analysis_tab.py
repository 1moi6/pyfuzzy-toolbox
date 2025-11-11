"""
Analysis Tab for Wang-Mendel Module
Visualizes rules, membership functions, and system behavior
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render():
    """Render analysis tab"""

    if not st.session_state.get('wm_trained', False) or st.session_state.get('wm_model', None) is None:
        st.warning("⚠️ Train a model first to analyze the system (Training tab)")
        return

    # Rule base analysis
    with st.expander("**Rule Base** - Generated Rules & Structure", expanded=False):
        render_rule_analysis()

    st.markdown("")

    # Membership functions
    with st.expander("**Membership Functions** - Variable Visualization", expanded=False):
        render_membership_functions()

    st.markdown("")

    # System behavior
    with st.expander("**System Behavior** - Decision Surface", expanded=False):
        render_system_behavior()

    # Add space at the end
    st.markdown("")
    st.markdown("")


def render_rule_analysis():
    """Display and analyze the generated rule base"""

    system = st.session_state.wm_system
    stats = st.session_state.get('wm_training_stats', {})

    # Overview
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Rules", stats.get('final_rules', 0))

    with col2:
        st.metric("Inputs", len(system.input_variables))

    with col3:
        st.metric("Outputs", len(system.output_variables))

    st.markdown("---")

    if len(system.rule_base) == 0:
        st.warning("⚠️ No rules found in the system")
        return

    # Radio button for different visualizations (similar to ANFIS)
    view_option = st.radio(
        "Select visualization:",
        options=["Table View", "Visual Matrix"],
        horizontal=True,
        key='wm_fuzzy_rules_view'
    )

    st.markdown("")

    # Render selected view
    if view_option == "Table View":
        render_rules_table_view(system)
    else:  # Visual Matrix
        render_rules_matrix_view(system)


def render_rules_table_view(system):
    """Render rules as interactive table with controls"""
    import pandas as pd

    # Get variable names in order
    input_names = list(system.input_variables.keys())
    output_names = list(system.output_variables.keys())

    # Create rule dataframe with separate columns for each variable
    rules_data = []

    for i, rule in enumerate(system.rule_base.rules, 1):
        row = {'Rule': f'R{i}'}

        # Add input columns (antecedents)
        for var_name in input_names:
            if var_name in rule.antecedents:
                term_value = rule.antecedents[var_name]
                # term_value is a tuple (var_name, term_name)
                if isinstance(term_value, tuple) and len(term_value) == 2:
                    _, term_name = term_value
                else:
                    term_name = str(term_value)
                row[var_name] = term_name
            else:
                row[var_name] = '-'

        # Add output columns (consequents)
        for var_name in output_names:
            if var_name in rule.consequent:
                term_value = rule.consequent[var_name]
                # term_value is a tuple (var_name, term_name)
                if isinstance(term_value, tuple) and len(term_value) == 2:
                    _, term_name = term_value
                else:
                    term_name = str(term_value)
                row[var_name] = term_name
            else:
                row[var_name] = '-'

        rules_data.append(row)

    rules_df = pd.DataFrame(rules_data)

    # Display options
    n_rules = len(rules_data)

    col1, col2 = st.columns([1, 3])

    with col1:
        show_all = st.checkbox("Show all rules", value=(n_rules <= 20), key='show_all_rules_wm')

    with col2:
        if not show_all:
            n_display = st.slider("Number of rules to display", 5, min(50, n_rules), 10, key='n_rules_display_wm')
        else:
            n_display = n_rules

    st.markdown("")

    # Add visual separation hint
    n_inputs = len(input_names)
    n_outputs = len(output_names)

    st.caption(f"📥 **Inputs (Antecedents):** {', '.join(input_names)} | "
              f"📤 **Output (Consequent):** {', '.join(output_names)}")

    st.markdown("")

    # Display dataframe with styling
    df_display = rules_df.head(n_display) if not show_all and n_rules > 20 else rules_df

    # Use column config to better format the display
    column_config = {
        'Rule': st.column_config.TextColumn('Rule', width='small'),
    }

    # Add config for input columns
    for var_name in input_names:
        column_config[var_name] = st.column_config.TextColumn(
            var_name,
            width='medium',
            help=f'Input: {var_name}'
        )

    # Add config for output columns
    for var_name in output_names:
        column_config[var_name] = st.column_config.TextColumn(
            var_name,
            width='medium',
            help=f'Output: {var_name}'
        )

    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        height=min(600, 35 * (len(df_display) + 1)),
        column_config=column_config
    )

    if not show_all and n_rules > 20:
        st.caption(f"Showing {n_display} of {n_rules} rules")

    # Download button
    st.markdown("")
    csv = rules_df.to_csv(index=False)
    st.download_button(
        label="Download Rules (CSV)",
        data=csv,
        file_name="wang_mendel_rules.csv",
        mime="text/csv",
        use_container_width=True
    )


def render_rules_matrix_view(system):
    """Render rules as visual colored matrix"""

    st.markdown("**Visual Rules Matrix**")
    st.caption("Color-coded matrix showing antecedents and consequents for each rule")

    # Initialize colormap and font size in session state
    if 'wm_matrix_colormap' not in st.session_state:
        st.session_state.wm_matrix_colormap = 'blues'
    if 'wm_matrix_fontsize' not in st.session_state:
        st.session_state.wm_matrix_fontsize = 9

    # Add color palette and font size selector in popover
    with st.popover("🎨 Appearance", use_container_width=False):
        st.markdown("**Select Color Scheme**")

        # Plotly colorscales organized by category (all valid colorscales)
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
            current_idx = palette_list.index(st.session_state.wm_matrix_colormap)
        except ValueError:
            current_idx = 0

        # Single selectbox with all options
        st.session_state.wm_matrix_colormap = st.selectbox(
            "Color palette",
            options=palette_list,
            index=current_idx,
            key='wm_colormap_selector',
            help="Choose a color scheme for the rule matrix"
        )

        st.markdown("")
        st.caption("💡 **Tip:** Sequential palettes (like blues) work best for rules visualization")

        st.markdown("---")

        # Font size selector
        st.markdown("**Text Size**")
        st.session_state.wm_matrix_fontsize = st.slider(
            "Font size",
            min_value=6,
            max_value=14,
            value=st.session_state.wm_matrix_fontsize,
            step=1,
            key='wm_fontsize_slider',
            help="Adjust the size of text in the matrix cells"
        )

    st.markdown("")

    # Check if suitable for matrix visualization
    n_inputs = len(system.input_variables)

    if n_inputs > 5:
        st.info(f"Visual matrix works best with ≤5 inputs. Your system has {n_inputs} inputs.\n\n"
                "Consider using Table View for better readability.")

    try:
        render_rules_visual_matrix(
            system,
            colormap=st.session_state.wm_matrix_colormap,
            fontsize=st.session_state.wm_matrix_fontsize
        )
    except Exception as e:
        st.error(f"❌ Error generating visual matrix: {str(e)}")


def render_membership_functions():
    """Visualize membership functions for all variables"""

    system = st.session_state.wm_system

    # Interactive membership visualization with Plotly
    st.markdown("**Interactive Membership Functions:**")

    # Variable selector
    all_vars = list(system.input_variables.keys()) + list(system.output_variables.keys())

    selected_var = st.selectbox(
        "Select variable to visualize",
        all_vars,
        key='wm_analysis_var_selector'
    )

    if selected_var:
        render_interactive_membership(system, selected_var)


def render_interactive_membership(system, var_name):
    """Render interactive membership function plot"""

    # Get variable
    if var_name in system.input_variables:
        var = system.input_variables[var_name]
    else:
        var = system.output_variables[var_name]

    # Create points for plotting
    x_range = var.universe
    x_points = np.linspace(x_range[0], x_range[1], 1000)

    # Create figure
    fig = go.Figure()

    # Plot each term
    for term_name, mf in var.terms.items():
        y_points = mf.membership(x_points)

        fig.add_trace(go.Scatter(
            x=x_points,
            y=y_points,
            mode='lines',
            name=term_name,
            line=dict(width=2),
            hovertemplate=f'<b>{term_name}</b><br>x=%{{x:.3f}}<br>μ=%{{y:.3f}}<extra></extra>'
        ))

    fig.update_layout(
        title=f'Membership Functions - {var_name}',
        xaxis_title=var_name,
        yaxis_title='Membership Degree (μ)',
        height=400,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)


def render_system_behavior():
    """Visualize system behavior (decision surface for 2D inputs)"""

    system = st.session_state.wm_system
    model = st.session_state.wm_model

    n_inputs = len(system.input_variables)

    if n_inputs == 1:
        render_1d_behavior(system, model)
    elif n_inputs == 2:
        render_2d_behavior(system, model)
    else:
        render_nd_behavior(system, model, n_inputs)


def render_1d_behavior(system, model):
    """Render 1D input-output curve"""

    st.markdown("**Input-Output Curve:**")

    # Get input variable
    input_name = list(system.input_variables.keys())[0]
    input_var = system.input_variables[input_name]

    # Create input range
    x_min, x_max = input_var.universe
    x_points = np.linspace(x_min, x_max, 200).reshape(-1, 1)

    # Scale if needed
    scaler_X = st.session_state.get('wm_scaler_X', None)
    if scaler_X is not None:
        x_points_scaled = scaler_X.transform(x_points)
    else:
        x_points_scaled = x_points

    # Predict
    try:
        y_pred = model.predict(x_points_scaled)

        # Inverse scale if needed
        scaler_y = st.session_state.get('wm_scaler_y', None)
        task = st.session_state.get('wm_task', 'regression')

        if scaler_y is not None and task == 'regression':
            y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        # Plot
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x_points.flatten(),
            y=y_pred.flatten() if hasattr(y_pred, 'flatten') else y_pred,
            mode='lines',
            name='System Output',
            line=dict(color='blue', width=2)
        ))

        # Add training points if available
        X_train = st.session_state.wm_X_train
        y_train = st.session_state.wm_y_train

        if scaler_X is not None:
            X_train = scaler_X.inverse_transform(X_train)
        if scaler_y is not None and task == 'regression':
            y_train = scaler_y.inverse_transform(y_train)

        fig.add_trace(go.Scatter(
            x=X_train.flatten(),
            y=y_train.flatten(),
            mode='markers',
            name='Training Data',
            marker=dict(color='red', size=6)
        ))

        target_name = st.session_state.get('wm_target_name', 'Y')

        fig.update_layout(
            title='System Input-Output Behavior',
            xaxis_title=input_name,
            yaxis_title=target_name,
            height=500,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error generating 1D behavior plot: {str(e)}")


def render_2d_behavior(system, model):
    """Render 2D decision surface"""

    st.markdown("**Decision Surface:**")

    # Get input variables
    input_names = list(system.input_variables.keys())
    input1_name = input_names[0]
    input2_name = input_names[1]

    input1_var = system.input_variables[input1_name]
    input2_var = system.input_variables[input2_name]

    # Create grid
    resolution = st.slider("Surface resolution", 20, 100, 50, 10, key='wm_surface_resolution')

    x1_min, x1_max = input1_var.universe
    x2_min, x2_max = input2_var.universe

    x1 = np.linspace(x1_min, x1_max, resolution)
    x2 = np.linspace(x2_min, x2_max, resolution)

    X1, X2 = np.meshgrid(x1, x2)

    # Flatten for prediction
    X_grid = np.c_[X1.ravel(), X2.ravel()]

    # Scale if needed
    scaler_X = st.session_state.get('wm_scaler_X', None)
    if scaler_X is not None:
        X_grid_scaled = scaler_X.transform(X_grid)
    else:
        X_grid_scaled = X_grid

    # Predict
    try:
        with st.spinner("Generating decision surface..."):
            y_pred = model.predict(X_grid_scaled)

            # Inverse scale if needed
            scaler_y = st.session_state.get('wm_scaler_y', None)
            task = st.session_state.get('wm_task', 'regression')

            if scaler_y is not None and task == 'regression':
                y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

            # Reshape
            Z = y_pred.reshape(X1.shape)

            # Plot
            fig = go.Figure(data=[go.Surface(
                x=X1,
                y=X2,
                z=Z,
                colorscale='Viridis',
                name='System Output'
            )])

            target_name = st.session_state.get('wm_target_name', 'Y')

            fig.update_layout(
                title='Decision Surface',
                scene=dict(
                    xaxis_title=input1_name,
                    yaxis_title=input2_name,
                    zaxis_title=target_name
                ),
                height=600
            )

            st.plotly_chart(fig, use_container_width=True)

            # Also show contour plot
            st.markdown("**Contour Plot:**")

            fig_contour = go.Figure(data=[go.Contour(
                x=x1,
                y=x2,
                z=Z,
                colorscale='Viridis',
                contours=dict(showlabels=True),
                name='System Output'
            )])

            # Add training points
            X_train = st.session_state.wm_X_train

            if scaler_X is not None:
                X_train = scaler_X.inverse_transform(X_train)

            fig_contour.add_trace(go.Scatter(
                x=X_train[:, 0],
                y=X_train[:, 1],
                mode='markers',
                name='Training Data',
                marker=dict(color='red', size=6, symbol='circle-open', line=dict(width=2))
            ))

            fig_contour.update_layout(
                title='Contour Plot with Training Data',
                xaxis_title=input1_name,
                yaxis_title=input2_name,
                height=500
            )

            st.plotly_chart(fig_contour, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error generating decision surface: {str(e)}")


def render_nd_behavior(system, model, n_inputs):
    """Render behavior for N-dimensional inputs (N > 2)"""

    st.info(f"Decision surface visualization is only available for 1-2 inputs. Your system has {n_inputs} inputs.")

    st.markdown("**Feature Importance Analysis (Coming Soon)**")
    st.caption("For high-dimensional systems, feature importance and sensitivity analysis will be available in future versions.")

    # Show input ranges
    st.markdown("---")
    st.markdown("**Input Variable Ranges:**")

    for input_name, input_var in system.input_variables.items():
        x_min, x_max = input_var.universe
        n_terms = len(input_var.terms)
        st.caption(f"• **{input_name}**: [{x_min:.2f}, {x_max:.2f}] with {n_terms} linguistic terms")


def render_rules_visual_matrix(system, colormap='blues', fontsize=9):
    """Render rules as visual colored matrix using Plotly

    Args:
        system: FuzzySystem object
        colormap: Plotly colorscale name (e.g., 'blues', 'greens', 'rdylbu')
        fontsize: Font size for cell text (default: 9)
    """

    # Get input and output names
    input_names = list(system.input_variables.keys())
    output_names = list(system.output_variables.keys())

    # For classification with multiple outputs, simplify display
    if len(output_names) > 3:
        output_names = [f"{len(output_names)} outputs"]

    # Get rules
    rules = system.rule_base.rules
    n_rules = len(rules)

    # Limit display for large rule bases
    max_display = 50
    if n_rules > max_display:
        st.warning(f"Showing first {max_display} rules (total: {n_rules})")
        n_display = max_display
    else:
        n_display = n_rules

    # Build column headers
    all_columns = input_names + output_names

    # Create mapping of term names to indices for coloring
    term_to_idx = {}
    idx_counter = 1

    # Map input terms
    for var_name in input_names:
        var = system.input_variables[var_name]
        for term_name in var.terms.keys():
            key = f"{var_name}:{term_name}"
            if key not in term_to_idx:
                term_to_idx[key] = idx_counter
                idx_counter += 1

    # Store the range of input colors for reuse in outputs
    max_input_color = idx_counter - 1  # Highest color used by inputs

    # Map output terms with consistent colors for yes/no across all outputs
    # YES/NO use fixed values for better contrast (not too extreme)
    YES_COLOR = 5  # Dark color (but not the darkest)
    NO_COLOR = 3   # Light-medium color (visible but not too light)

    for var_name in output_names:
        var = system.output_variables[var_name]
        for term_name in var.terms.keys():
            # Use term name without variable prefix for yes/no consistency
            if term_name.lower() == 'yes':
                # YES uses a dark color (value 5)
                key = f"{var_name}:{term_name}"
                term_to_idx[key] = YES_COLOR
            elif term_name.lower() == 'no':
                # NO uses a lighter color (value 3)
                key = f"{var_name}:{term_name}"
                term_to_idx[key] = NO_COLOR
            else:
                # For non-yes/no terms, keep per-variable coloring
                key = f"{var_name}:{term_name}"
                if key not in term_to_idx:
                    term_to_idx[key] = idx_counter
                    idx_counter += 1

    # Build matrix data
    matrix_data = []
    annotations = []

    for rule_idx, rule in enumerate(rules[:n_display]):
        row_data = []

        # Process antecedents (inputs)
        for var_name in input_names:
            if var_name in rule.antecedents:
                term_value = rule.antecedents[var_name]

                # Extract term name from tuple or string
                if isinstance(term_value, tuple) and len(term_value) == 2:
                    _, term_name = term_value
                else:
                    term_name = str(term_value)

                key = f"{var_name}:{term_name}"
                color_val = term_to_idx.get(key, 0)
                row_data.append(color_val)

                # Choose text color based on background darkness
                # Blues colorscale: higher values = darker blue
                # Use threshold of 4 (between NO=3 and YES=5)
                text_color = 'white' if color_val >= 4 else 'black'

                annotations.append(
                    dict(
                        x=input_names.index(var_name),
                        y=rule_idx,
                        text=f'<b>{term_name}</b>',
                        showarrow=False,
                        font=dict(size=fontsize, color=text_color)
                    )
                )
            else:
                row_data.append(0)
                annotations.append(
                    dict(
                        x=input_names.index(var_name),
                        y=rule_idx,
                        text='<b>-</b>',
                        showarrow=False,
                        font=dict(size=fontsize, color='gray')
                    )
                )

        # Process consequents (outputs)
        # Check if this is a multi-output classification case
        if len(system.output_variables) > 3:
            # Classification with multiple outputs - find which class this rule predicts
            consequent_terms = []
            for out_var in system.output_variables.keys():
                if out_var in rule.consequent:
                    term_value = rule.consequent[out_var]
                    if isinstance(term_value, tuple) and len(term_value) == 2:
                        _, term_name = term_value
                    else:
                        term_name = str(term_value)

                    if term_name == 'yes':
                        # Extract class number from output name like "Y_class_0"
                        class_idx = out_var.split('_')[-1]
                        consequent_terms.append(f"C{class_idx}")

            row_data.append(0)
            annotations.append(
                dict(
                    x=len(input_names),
                    y=rule_idx,
                    text=','.join(consequent_terms) if consequent_terms else '-',
                    showarrow=False,
                    font=dict(size=max(6, fontsize-1), color='darkgreen', weight='bold')
                )
            )
        else:
            # Regression or simple classification
            for var_name in output_names:
                if var_name in rule.consequent:
                    term_value = rule.consequent[var_name]

                    # Extract term name from tuple or string
                    if isinstance(term_value, tuple) and len(term_value) == 2:
                        _, term_name = term_value
                    else:
                        term_name = str(term_value)

                    # Get color value for this output term (same as inputs)
                    key = f"{var_name}:{term_name}"
                    color_val = term_to_idx.get(key, 0)
                    row_data.append(color_val)

                    # Choose text color based on background darkness
                    text_color = 'white' if color_val >= 4 else 'black'

                    annotations.append(
                        dict(
                            x=len(input_names) + output_names.index(var_name),
                            y=rule_idx,
                            text=term_name,
                            showarrow=False,
                            font=dict(size=fontsize, color=text_color, weight='bold')
                        )
                    )

        matrix_data.append(row_data)

    # Create heatmap with selected colormap and gridlines
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=matrix_data,
        x=all_columns,
        y=[f'R{i+1}' for i in range(n_display)],
        colorscale=colormap,
        showscale=False,
        hoverinfo='skip',
        xgap=1,  # Gap between columns (gridlines)
        ygap=1   # Gap between rows (gridlines)
    ))

    # Add text annotations
    fig.update_layout(
        annotations=annotations,
        xaxis=dict(
            title='',
            tickmode='array',
            tickvals=list(range(len(all_columns))),
            ticktext=all_columns,
            side='top'
        ),
        yaxis=dict(
            title='Rules',
            autorange='reversed'
        ),
        height=min(800, 25 * n_display + 100),
        template='plotly_white',
        margin=dict(l=60, r=20, t=80, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)
