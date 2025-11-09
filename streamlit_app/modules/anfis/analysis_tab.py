"""
Analysis Tab for ANFIS Module
Model structure analysis, membership functions, rules, and surface visualization
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render():
    """Render analysis tab"""

    if not st.session_state.get('anfis_trained', False) or st.session_state.get('anfis_model', None) is None:
        st.info("🎯 Train a model first to see analysis here (Training tab)")
        return

    # Get model
    model = st.session_state.anfis_model

    # Render sections
    render_model_architecture(model)

    st.markdown("")

    render_membership_functions(model)

    st.markdown("")

    render_fuzzy_rules(model)

    st.markdown("")

    if model.n_inputs == 2:
        render_surface_plot(model)
        st.markdown("")

    render_sensitivity_analysis(model)

    # Add space at the end
    st.markdown("")
    st.markdown("")


def render_model_architecture(model):
    """Display ANFIS model architecture information"""

    with st.expander("**Model Architecture**", expanded=True):

        # Get feature and target names
        feature_names = st.session_state.get('anfis_feature_names',
                                             [f'X{i+1}' for i in range(model.n_inputs)])
        target_name = st.session_state.get('anfis_target_name', 'Y')

        # Architecture overview
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Inputs", model.n_inputs, help="Number of input features")

        with col2:
            if isinstance(model.n_mfs, int):
                mf_display = f"{model.n_mfs} per input"
            else:
                mf_display = f"{min(model.n_mfs)}-{max(model.n_mfs)}"
            st.metric("MFs", mf_display, help="Membership functions per input")

        with col3:
            st.metric("Rules", model.n_rules, help="Total fuzzy rules")

        with col4:
            # Count total parameters
            if hasattr(model, 'premise_parameters') and hasattr(model, 'consequent_parameters'):
                n_premise = len(model.premise_parameters)
                n_consequent = len(model.consequent_parameters)
                total_params = n_premise + n_consequent
            else:
                total_params = "N/A"
            st.metric("Parameters", total_params, help="Total trainable parameters")

        st.markdown("---")

        # ANFIS layer information
        st.markdown("**ANFIS Layers Structure**")

        layers_info = [
            {
                'Layer': 'Layer 1',
                'Name': 'Fuzzification',
                'Function': 'Membership Functions',
                'Description': f'{model.mf_type} MFs compute input membership degrees'
            },
            {
                'Layer': 'Layer 2',
                'Name': 'Rule Firing',
                'Function': 'Product (T-norm)',
                'Description': f'{model.n_rules} rules compute firing strengths'
            },
            {
                'Layer': 'Layer 3',
                'Name': 'Normalization',
                'Function': 'Division',
                'Description': 'Normalize firing strengths to sum=1'
            },
            {
                'Layer': 'Layer 4',
                'Name': 'Consequent',
                'Function': 'Linear/Constant',
                'Description': 'Apply consequent parameters to normalized strengths'
            },
            {
                'Layer': 'Layer 5',
                'Name': 'Defuzzification',
                'Function': 'Weighted Sum',
                'Description': 'Aggregate outputs to produce final result'
            }
        ]

        df_layers = pd.DataFrame(layers_info)
        st.dataframe(df_layers, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Input/Output information
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Input Features**")
            for i, name in enumerate(feature_names):
                if st.session_state.get('anfis_X_train', None) is not None:
                    x_min = st.session_state.anfis_X_train[:, i].min()
                    x_max = st.session_state.anfis_X_train[:, i].max()
                    st.caption(f"• {name}: [{x_min:.3f}, {x_max:.3f}]")
                else:
                    st.caption(f"• {name}")

        with col2:
            st.markdown("**Output**")
            st.caption(f"• {target_name}")
            if hasattr(model, 'classification') and model.classification:
                st.caption("  Type: Classification")
            else:
                st.caption("  Type: Regression")


def render_membership_functions(model):
    """Visualize membership functions for each input"""

    with st.expander("**Membership Functions**", expanded=True):

        # Get feature names
        feature_names = st.session_state.get('anfis_feature_names',
                                             [f'X{i+1}' for i in range(model.n_inputs)])

        # Get input ranges from training data
        if st.session_state.get('anfis_X_train', None) is not None:
            X_train = st.session_state.anfis_X_train
            input_ranges = [(X_train[:, i].min(), X_train[:, i].max())
                          for i in range(model.n_inputs)]
        else:
            input_ranges = [(-3, 3) for _ in range(model.n_inputs)]

        # Select input to visualize
        if model.n_inputs > 1:
            selected_input = st.selectbox(
                "Select input feature",
                range(model.n_inputs),
                format_func=lambda i: feature_names[i],
                key='mf_input_selector'
            )
        else:
            selected_input = 0

        st.markdown("")

        # Get MF parameters for selected input
        n_mfs_for_input = model.n_mfs if isinstance(model.n_mfs, int) else model.n_mfs[selected_input]

        # Create plot
        fig = go.Figure()

        # Generate x values for plotting
        x_min, x_max = input_ranges[selected_input]
        x_range = x_max - x_min
        x_vals = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 200)

        # Color palette for MFs
        colors = ['#667eea', '#f093fb', '#4ade80', '#fbbf24', '#f87171', '#a78bfa', '#fb923c', '#2dd4bf']

        # Get MF type
        mf_type = model.mf_type

        # Extract and plot each MF
        for mf_idx in range(n_mfs_for_input):
            # Get MF parameters
            if hasattr(model, 'mf_params') and model.mf_params is not None:
                params = model.mf_params[selected_input][mf_idx]

                # Compute membership values
                y_vals = compute_membership(x_vals, params, mf_type)

                # Plot MF
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines',
                    name=f'MF{mf_idx + 1}',
                    line=dict(color=colors[mf_idx % len(colors)], width=2),
                    hovertemplate=f'MF{mf_idx + 1}<br>x: %{{x:.3f}}<br>μ: %{{y:.3f}}<extra></extra>'
                ))

        # Add training data distribution
        if st.session_state.get('anfis_X_train', None) is not None:
            x_data = st.session_state.anfis_X_train[:, selected_input]

            # Create histogram for distribution
            hist, bin_edges = np.histogram(x_data, bins=30, density=True)
            # Normalize to [0, 0.3] for better visualization
            hist_normalized = hist / hist.max() * 0.3

            fig.add_trace(go.Bar(
                x=(bin_edges[:-1] + bin_edges[1:]) / 2,
                y=hist_normalized,
                name='Data Distribution',
                marker_color='rgba(200, 200, 200, 0.3)',
                showlegend=True,
                hovertemplate='Bin: %{x:.3f}<br>Density: %{y:.3f}<extra></extra>'
            ))

        fig.update_layout(
            title=f'Membership Functions - {feature_names[selected_input]}',
            xaxis_title=feature_names[selected_input],
            yaxis_title='Membership Degree (μ)',
            yaxis_range=[0, 1.05],
            template='plotly_white',
            height=400,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show MF parameters
        with st.popover("MF Parameters Details",width='stretch',type='tertiary'):
            if hasattr(model, 'mf_params') and model.mf_params is not None:
                params_data = []

                for mf_idx in range(n_mfs_for_input):
                    params = model.mf_params[selected_input][mf_idx]

                    if mf_type == 'gaussmf':
                        params_data.append({
                            'MF': f'MF{mf_idx + 1}',
                            'Type': 'Gaussian',
                            'Center (c)': f"{params[0]:.4f}",
                            'Sigma (σ)': f"{params[1]:.4f}",
                            'Formula': 'exp(-(x-c)²/(2σ²))'
                        })
                    elif mf_type == 'gbellmf':
                        params_data.append({
                            'MF': f'MF{mf_idx + 1}',
                            'Type': 'Generalized Bell',
                            'a': f"{params[0]:.4f}",
                            'b': f"{params[1]:.4f}",
                            'c': f"{params[2]:.4f}",
                            'Formula': '1/(1+|(x-c)/a|^(2b))'
                        })
                    elif mf_type == 'sigmf':
                        params_data.append({
                            'MF': f'MF{mf_idx + 1}',
                            'Type': 'Sigmoid',
                            'a': f"{params[0]:.4f}",
                            'c': f"{params[1]:.4f}",
                            'Formula': '1/(1+exp(-a(x-c)))'
                        })

                df_params = pd.DataFrame(params_data)
                st.dataframe(df_params, use_container_width=True, hide_index=True)
            else:
                st.info("MF parameters not available for this model")


def compute_membership(x, params, mf_type):
    """Compute membership function values"""

    if mf_type == 'gaussmf':
        # Gaussian: exp(-(x-c)²/(2σ²))
        c, sigma = params
        return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))

    elif mf_type == 'gbellmf':
        # Generalized Bell: 1/(1+|(x-c)/a|^(2b))
        a, b, c = params
        return 1 / (1 + np.abs((x - c) / a) ** (2 * b))

    elif mf_type == 'sigmf':
        # Sigmoid: 1/(1+exp(-a(x-c)))
        a, c = params
        return 1 / (1 + np.exp(-a * (x - c)))

    else:
        return np.zeros_like(x)


def render_fuzzy_rules(model):
    """Display fuzzy rules information"""

    with st.expander("**Fuzzy Rules**", expanded=False):

        st.markdown(f"**Total Rules:** {model.n_rules}")
        st.markdown("")

        # Get feature names
        feature_names = st.session_state.get('anfis_feature_names',
                                             [f'X{i+1}' for i in range(model.n_inputs)])
        target_name = st.session_state.get('anfis_target_name', 'Y')

        # Check if model has the necessary methods
        if not hasattr(model, 'rules_to_dataframe'):
            st.warning("Model does not support rule visualization")
            return

        # Radio button for different visualizations
        view_option = st.radio(
            "Select visualization:",
            options=["Table View", "Visual Matrix", "Activation Analysis"],
            horizontal=True,
            key='fuzzy_rules_view'
        )

        st.markdown("")

        # Render selected view
        if view_option == "Table View":
            render_rules_table(model, feature_names, target_name)
        elif view_option == "Visual Matrix":
            render_rules_visual_matrix(model, feature_names, target_name)
        else:  # Activation Analysis
            render_rule_activation_analysis(model)


def render_rules_table(model, feature_names, target_name):
    """Render rules as interactive table using ANFIS rules_to_dataframe"""

    # st.markdown("**Fuzzy Rules Table**")
    # st.caption("Interactive table showing all fuzzy rules with their antecedents and consequents")

    # st.markdown("")

    try:
        # Use ANFIS native method to generate rules DataFrame
        df_rules = model.rules_to_dataframe(
            input_names=feature_names,
            output_name=target_name
        )

        # Display options
        col1, col2 = st.columns([1, 3])

        with col1:
            show_all = st.checkbox("Show all rules", value=(model.n_rules <= 20), key='show_all_rules_table')

        with col2:
            if not show_all:
                n_display = st.slider("Number of rules to display", 5, min(50, model.n_rules), 10, key='n_rules_display')
            else:
                n_display = model.n_rules

        st.markdown("")

        # Display dataframe
        if show_all or model.n_rules <= 20:
            st.dataframe(df_rules, use_container_width=True, height=min(600, 35 * (len(df_rules) + 1)))
        else:
            st.dataframe(df_rules.head(n_display), use_container_width=True, height=min(600, 35 * (n_display + 1)))
            st.caption(f"Showing {n_display} of {model.n_rules} rules")

        # Download button
        st.markdown("")
        csv = df_rules.to_csv(index=False)
        st.download_button(
            label="Download Rules (CSV)",
            data=csv,
            file_name="anfis_fuzzy_rules.csv",
            mime="text/csv",
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Error generating rules table: {str(e)}")
        st.info("Using fallback rule generation...")
        render_fallback_rules_table(model, feature_names, target_name)


def render_fallback_rules_table(model, feature_names, target_name):
    """Fallback rule table if native method fails"""

    if isinstance(model.n_mfs, int):
        n_mfs_per_input = [model.n_mfs] * model.n_inputs
    else:
        n_mfs_per_input = model.n_mfs

    rules_data = []
    rule_idx = 0

    for combination in generate_rule_combinations(n_mfs_per_input):
        antecedent_parts = []
        for i, mf_idx in enumerate(combination):
            antecedent_parts.append(f"{feature_names[i]} is MF{mf_idx + 1}")

        antecedent = " AND ".join(antecedent_parts)

        rules_data.append({
            'Rule': f'R{rule_idx + 1}',
            'IF (Antecedent)': antecedent,
            'THEN (Consequent)': f'{target_name} = f(inputs)'
        })
        rule_idx += 1

    df_rules = pd.DataFrame(rules_data)
    st.dataframe(df_rules, use_container_width=True, height=400)


def render_rules_visual_matrix(model, feature_names, target_name):
    """Render rules as visual colored matrix using Plotly"""

    st.markdown("**Visual Rules Matrix**")
    st.caption("Color-coded matrix showing antecedents and consequent parameters")

    st.markdown("")

    try:
        # Get consequent parameters directly from model
        consequent_params = model.consequent_params  # Shape: (n_rules, n_inputs + 1)

        # Get number of rules
        n_rules = model.n_rules
        n_inputs = model.n_inputs

        # Limit display for large rule bases
        max_display = 50
        if n_rules > max_display:
            st.warning(f"Showing first {max_display} rules (total: {n_rules})")
            n_display = max_display
        else:
            n_display = n_rules

        # Build column headers: feature names + consequent parameters
        all_columns = feature_names + ['Consequent']

        # Build matrix data
        matrix_data = []
        annotations = []

        for rule_idx in range(n_display):
            row_data = []

            # Get MF indices for this rule
            mf_indices = model._rule_indices_cache[rule_idx]

            # Add antecedent MF values
            for input_idx, mf_idx in enumerate(mf_indices):
                row_data.append(mf_idx + 1)
                annotations.append(
                    dict(
                        x=input_idx,
                        y=rule_idx,
                        text=f'MF{mf_idx + 1}',
                        showarrow=False,
                        font=dict(size=9, color='white')
                    )
                )

            # Add consequent parameters as tuple
            params = consequent_params[rule_idx]
            params_str = '(' + ', '.join([f'{p:.3f}' for p in params]) + ')'
            row_data.append(0)  # Neutral color for parameters column
            annotations.append(
                dict(
                    x=n_inputs,
                    y=rule_idx,
                    text=params_str,
                    showarrow=False,
                    font=dict(size=7, color='black')
                )
            )

            matrix_data.append(row_data)

        # Create heatmap
        fig = go.Figure()

        fig.add_trace(go.Heatmap(
            z=matrix_data,
            x=all_columns,
            y=[f'R{i+1}' for i in range(n_display)],
            colorscale='Blues',
            showscale=False,
            hoverinfo='skip'
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

    except Exception as e:
        st.error(f"Error generating visual matrix: {str(e)}")


def render_rule_activation_analysis(model):
    """Render rule activation statistics and analysis"""

    st.markdown("**Rule Activation Analysis**")
    st.caption("Statistical analysis of how often each rule is activated during training")

    st.markdown("")

    if st.session_state.get('anfis_X_train', None) is None:
        st.info("Training data needed for activation analysis")
        return

    X_train = st.session_state.anfis_X_train

    try:
        # Get firing strengths for training data
        firing_strengths = compute_firing_strengths(model, X_train)

        # Calculate average activation per rule
        avg_activation = firing_strengths.mean(axis=0)
        max_activation = firing_strengths.max(axis=0)
        min_activation = firing_strengths.min(axis=0)

        # Find most/least active rules
        most_active_idx = np.argmax(avg_activation)
        least_active_idx = np.argmin(avg_activation)

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Most Active Rule", f"R{most_active_idx + 1}",
                     help=f"Average activation: {avg_activation[most_active_idx]:.4f}")

        with col2:
            st.metric("Least Active Rule", f"R{least_active_idx + 1}",
                     help=f"Average activation: {avg_activation[least_active_idx]:.4f}")

        with col3:
            active_rules = np.sum(avg_activation > 0.01)
            st.metric("Active Rules", f"{active_rules}/{model.n_rules}",
                     help="Rules with avg activation > 0.01")

        with col4:
            dormant_rules = np.sum(avg_activation < 0.001)
            st.metric("Dormant Rules", dormant_rules,
                     help="Rules with avg activation < 0.001")

        st.markdown("")

        # Activation distribution plot
        st.markdown("**Activation Distribution**")

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Average Activation per Rule', 'Activation Range (Min-Max)')
        )

        # Average activation bar chart
        fig.add_trace(
            go.Bar(
                x=list(range(1, model.n_rules + 1)),
                y=avg_activation,
                name='Avg Activation',
                marker_color='#667eea',
                hovertemplate='Rule %{x}<br>Avg: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Min-Max range
        fig.add_trace(
            go.Scatter(
                x=list(range(1, model.n_rules + 1)),
                y=max_activation,
                mode='lines',
                name='Max',
                line=dict(color='#4ade80', width=2),
                hovertemplate='Rule %{x}<br>Max: %{y:.4f}<extra></extra>'
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=list(range(1, model.n_rules + 1)),
                y=min_activation,
                mode='lines',
                name='Min',
                fill='tonexty',
                line=dict(color='#f87171', width=2),
                hovertemplate='Rule %{x}<br>Min: %{y:.4f}<extra></extra>'
            ),
            row=1, col=2
        )

        fig.update_xaxes(title_text="Rule Number", row=1, col=1)
        fig.update_xaxes(title_text="Rule Number", row=1, col=2)
        fig.update_yaxes(title_text="Activation", row=1, col=1)
        fig.update_yaxes(title_text="Activation", row=1, col=2)

        fig.update_layout(
            template='plotly_white',
            height=400,
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

        # Rule importance table
        st.markdown("**Top 10 Most Active Rules**")

        top_indices = np.argsort(avg_activation)[-10:][::-1]

        top_rules_data = []
        for rank, idx in enumerate(top_indices, 1):
            top_rules_data.append({
                'Rank': rank,
                'Rule': f'R{idx + 1}',
                'Avg Activation': f'{avg_activation[idx]:.4f}',
                'Max Activation': f'{max_activation[idx]:.4f}',
                'Min Activation': f'{min_activation[idx]:.4f}'
            })

        df_top = pd.DataFrame(top_rules_data)
        st.dataframe(df_top, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error computing rule activations: {str(e)}")


def render_all_rules(model, feature_names, target_name, n_mfs_per_input):
    """Render all fuzzy rules"""

    rules_data = []

    # Generate all rule combinations
    rule_idx = 0
    for combination in generate_rule_combinations(n_mfs_per_input):
        # Build rule antecedent
        antecedent_parts = []
        for i, mf_idx in enumerate(combination):
            antecedent_parts.append(f"{feature_names[i]} is MF{mf_idx + 1}")

        antecedent = " AND ".join(antecedent_parts)

        # Get consequent parameters if available
        if hasattr(model, 'consequent_parameters'):
            # For Sugeno, show linear equation
            consequent = f"Linear function of inputs"
        else:
            consequent = "Computed"

        rules_data.append({
            'Rule': f'R{rule_idx + 1}',
            'IF (Antecedent)': antecedent,
            'THEN (Consequent)': f'{target_name} = {consequent}'
        })

        rule_idx += 1

    df_rules = pd.DataFrame(rules_data)
    st.dataframe(df_rules, use_container_width=True, hide_index=True, height=400)


def render_sample_rules(model, feature_names, target_name, n_mfs_per_input):
    """Render sample of fuzzy rules for large rule bases"""

    # Show first 10 and last 10 rules
    rules_data = []

    rule_idx = 0
    for combination in generate_rule_combinations(n_mfs_per_input):
        if rule_idx < 10 or rule_idx >= model.n_rules - 10:
            # Build rule antecedent
            antecedent_parts = []
            for i, mf_idx in enumerate(combination):
                antecedent_parts.append(f"{feature_names[i]} is MF{mf_idx + 1}")

            antecedent = " AND ".join(antecedent_parts)

            rules_data.append({
                'Rule': f'R{rule_idx + 1}',
                'IF (Antecedent)': antecedent,
                'THEN': f'{target_name} = f(inputs)'
            })

        elif rule_idx == 10:
            # Add separator
            rules_data.append({
                'Rule': '...',
                'IF (Antecedent)': '...',
                'THEN': '...'
            })

        rule_idx += 1

    df_rules = pd.DataFrame(rules_data)
    st.dataframe(df_rules, use_container_width=True, hide_index=True, height=400)


def generate_rule_combinations(n_mfs_per_input):
    """Generate all combinations of MF indices for rules"""

    if len(n_mfs_per_input) == 1:
        for i in range(n_mfs_per_input[0]):
            yield (i,)
    else:
        for i in range(n_mfs_per_input[0]):
            for rest in generate_rule_combinations(n_mfs_per_input[1:]):
                yield (i,) + rest


def compute_firing_strengths(model, X):
    """Compute firing strengths for input data"""

    n_samples = X.shape[0]
    firing_strengths = np.zeros((n_samples, model.n_rules))

    # Get MF parameters
    if not hasattr(model, 'mf_params'):
        raise ValueError("Model does not have mf_params attribute")

    # For each sample
    for sample_idx in range(n_samples):
        x = X[sample_idx]

        # For each rule
        rule_idx = 0
        n_mfs_per_input = [model.n_mfs] * model.n_inputs if isinstance(model.n_mfs, int) else model.n_mfs

        for combination in generate_rule_combinations(n_mfs_per_input):
            # Compute firing strength (product of memberships)
            firing = 1.0

            for input_idx, mf_idx in enumerate(combination):
                params = model.mf_params[input_idx][mf_idx]
                membership = compute_membership(np.array([x[input_idx]]), params, model.mf_type)[0]
                firing *= membership

            firing_strengths[sample_idx, rule_idx] = firing
            rule_idx += 1

    return firing_strengths


def render_surface_plot(model):
    """Render 3D surface plot for 2-input models"""

    with st.expander("**Decision Surface**", expanded=False):

        st.markdown("**3D Surface Plot** - Model output over input space")

        # Get feature names
        feature_names = st.session_state.get('anfis_feature_names', ['X1', 'X2'])
        target_name = st.session_state.get('anfis_target_name', 'Y')

        # Get input ranges
        if st.session_state.get('anfis_X_train', None) is not None:
            X_train = st.session_state.anfis_X_train
            x1_range = (X_train[:, 0].min(), X_train[:, 0].max())
            x2_range = (X_train[:, 1].min(), X_train[:, 1].max())
        else:
            x1_range = (-3, 3)
            x2_range = (-3, 3)

        # Resolution slider
        resolution = st.slider("Surface resolution", 20, 100, 50, step=10,
                              help="Higher resolution = smoother surface but slower",
                              key='surface_resolution')

        st.markdown("")

        # Generate grid
        x1_vals = np.linspace(x1_range[0], x1_range[1], resolution)
        x2_vals = np.linspace(x2_range[0], x2_range[1], resolution)
        X1_grid, X2_grid = np.meshgrid(x1_vals, x2_vals)

        # Prepare input
        X_grid = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

        # Predict (use forward_batch to get continuous values for surface)
        with st.spinner("Generating surface..."):
            Z_pred = model.forward_batch(X_grid)  # Always use continuous values for surface
            Z_pred = np.atleast_1d(Z_pred)  # Ensure array format
            Z_grid = Z_pred.reshape(X1_grid.shape)

        # Create 3D surface plot
        fig = go.Figure()

        fig.add_trace(go.Surface(
            x=X1_grid,
            y=X2_grid,
            z=Z_grid,
            colorscale='Viridis',
            name='ANFIS Output',
            hovertemplate=f'{feature_names[0]}: %{{x:.3f}}<br>{feature_names[1]}: %{{y:.3f}}<br>{target_name}: %{{z:.3f}}<extra></extra>'
        ))

        # Add training data points if available
        if st.session_state.get('anfis_y_train', None) is not None:
            X_train = st.session_state.anfis_X_train
            y_train = st.session_state.anfis_y_train.ravel()

            fig.add_trace(go.Scatter3d(
                x=X_train[:, 0],
                y=X_train[:, 1],
                z=y_train,
                mode='markers',
                marker=dict(
                    size=3,
                    color='red',
                    opacity=0.6
                ),
                name='Training Data',
                hovertemplate=f'{feature_names[0]}: %{{x:.3f}}<br>{feature_names[1]}: %{{y:.3f}}<br>{target_name}: %{{z:.3f}}<extra></extra>'
            ))

        fig.update_layout(
            scene=dict(
                xaxis_title=feature_names[0],
                yaxis_title=feature_names[1],
                zaxis_title=target_name,
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            title='ANFIS Decision Surface',
            template='plotly_white',
            height=600,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)


def render_sensitivity_analysis(model):
    """Perform and display sensitivity analysis"""

    with st.expander("**Sensitivity Analysis** - Feature Importance", expanded=False):

        if st.session_state.get('anfis_X_train', None) is None:
            st.info("Training data needed for sensitivity analysis")
            return

        st.markdown("**Feature Importance based on Permutation**")
        st.caption("Measures output variance increase when feature values are randomly shuffled")

        st.markdown("")

        # Method selection
        col_method, col_reps = st.columns([2, 1])

        with col_method:
            method = st.selectbox(
                "Perturbation method:",
                options=["Permutation (recommended)", "Mean substitution", "Random noise"],
                key='sensitivity_method',
                help="**Permutation**: Shuffle feature values randomly (breaks dependencies)\n\n"
                     "**Mean substitution**: Replace all values with mean (can cause artifacts)\n\n"
                     "**Random noise**: Add random noise to feature values"
            )

        with col_reps:
            if method == "Permutation (recommended)":
                n_repeats = st.number_input("Repetitions", 1, 20, 5, key='sensitivity_reps',
                                          help="Number of random permutations to average")
            else:
                n_repeats = 1

        st.markdown("")

        # Get data
        X_train = st.session_state.anfis_X_train
        feature_names = st.session_state.get('anfis_feature_names',
                                             [f'X{i+1}' for i in range(model.n_inputs)])

        # Compute sensitivity for each feature
        sensitivities = []

        with st.spinner("Computing feature importance..."):
            # Get baseline prediction (use forward_batch for continuous values)
            y_baseline = model.forward_batch(X_train)
            y_baseline = np.atleast_1d(y_baseline)  # Ensure array format
            baseline_variance = np.var(y_baseline)

            for feature_idx in range(model.n_inputs):

                if method == "Permutation (recommended)":
                    # Permutation importance (average over multiple random shuffles)
                    importances = []
                    for _ in range(n_repeats):
                        X_perturbed = X_train.copy()
                        # Randomly shuffle the feature values
                        np.random.shuffle(X_perturbed[:, feature_idx])

                        y_perturbed = model.forward_batch(X_perturbed)
                        y_perturbed = np.atleast_1d(y_perturbed)

                        # Calculate variance reduction (same logic as original)
                        perturbed_variance = np.var(y_perturbed)
                        variance_reduction = baseline_variance - perturbed_variance
                        importance = variance_reduction / baseline_variance if baseline_variance != 0 else 0
                        importances.append(importance)

                    sensitivity = np.mean(importances)

                elif method == "Mean substitution":
                    # Original method: set feature to mean
                    X_perturbed = X_train.copy()
                    X_perturbed[:, feature_idx] = X_train[:, feature_idx].mean()

                    y_perturbed = model.forward_batch(X_perturbed)
                    y_perturbed = np.atleast_1d(y_perturbed)

                    # Calculate variance reduction (original logic)
                    perturbed_variance = np.var(y_perturbed)
                    variance_reduction = baseline_variance - perturbed_variance
                    sensitivity = variance_reduction / baseline_variance if baseline_variance != 0 else 0

                else:  # Random noise
                    # Add random noise to feature
                    importances = []
                    feature_std = X_train[:, feature_idx].std()
                    for _ in range(max(3, n_repeats)):
                        X_perturbed = X_train.copy()
                        X_perturbed[:, feature_idx] += np.random.normal(0, feature_std, X_train.shape[0])

                        y_perturbed = model.forward_batch(X_perturbed)
                        y_perturbed = np.atleast_1d(y_perturbed)

                        # Calculate variance reduction
                        perturbed_variance = np.var(y_perturbed)
                        variance_reduction = baseline_variance - perturbed_variance
                        importance = variance_reduction / baseline_variance if baseline_variance != 0 else 0
                        importances.append(importance)

                    sensitivity = np.mean(importances)

                sensitivities.append({
                    'Feature': feature_names[feature_idx],
                    'Sensitivity': sensitivity,
                    'Importance': f"{sensitivity * 100:.2f}%"
                })

        # Sort by sensitivity
        sensitivities.sort(key=lambda x: x['Sensitivity'], reverse=True)

        # Display results
        col1, col2 = st.columns([1, 2])

        with col1:
            # Table
            df_sensitivity = pd.DataFrame(sensitivities)
            st.dataframe(df_sensitivity[['Feature', 'Importance']],
                        use_container_width=True, hide_index=True)

        with col2:
            # Bar chart
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=[s['Feature'] for s in sensitivities],
                y=[s['Sensitivity'] * 100 for s in sensitivities],
                marker_color='#667eea',
                text=[f"{s['Sensitivity'] * 100:.1f}%" for s in sensitivities],
                textposition='outside',
                hovertemplate='%{x}<br>Importance: %{y:.2f}%<extra></extra>'
            ))

            fig.update_layout(
                title='Feature Importance',
                xaxis_title='Feature',
                yaxis_title='Importance (%)',
                template='plotly_white',
                height=300,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

        st.markdown("")

        # Method-specific interpretation
        if method == "Permutation (recommended)":
            st.caption(
                "💡 **Interpretation:** Positive values indicate that shuffling the feature **reduces** output variance "
                "(feature is important for variability). Values near zero suggest the feature is **redundant**. "
                "Negative values mean shuffling increased variance (rare, feature may have stabilizing role)."
            )
        elif method == "Mean substitution":
            st.caption(
                "💡 **Interpretation:** Positive values mean fixing the feature to its mean **reduces** variance "
                "(feature contributes to output variability). Negative values occur if the feature has a stabilizing/normalizing role. "
                "This is the original method used in the code."
            )
        else:  # Random noise
            st.caption(
                "💡 **Interpretation:** Positive values indicate that adding noise **reduces** output variance "
                "(counterintuitive but can happen). Negative values mean noise increases variance (feature is sensitive). "
                "Useful for testing model robustness."
            )
