"""
Prediction Tab for Wang-Mendel Module
Handles manual predictions, batch predictions, and test set evaluation
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def render():
    """Render prediction tab"""

    if not st.session_state.get('wm_trained', False) or st.session_state.get('wm_model', None) is None:
        st.info("🎯 Train a model first to make predictions (Training tab)")
        return

    # Get model
    model = st.session_state.wm_model
    system = st.session_state.wm_system
    n_inputs = len(system.input_variables)

    # Get feature information
    feature_names = st.session_state.get('wm_feature_names', [f'X{i+1}' for i in range(n_inputs)])
    target_name = st.session_state.get('wm_target_name', 'Y')
    task = st.session_state.get('wm_task', 'regression')
    scaler_X = st.session_state.get('wm_scaler_X', None)
    scaler_y = st.session_state.get('wm_scaler_y', None)

    # Render sections
    render_manual_prediction(model, n_inputs, feature_names, target_name, task, scaler_X, scaler_y)

    st.markdown("")

    render_dataset_predictions(model, task, scaler_y)

    st.markdown("")

    render_batch_prediction(model, feature_names, target_name, task, scaler_X, scaler_y)

    # Add space at the end
    st.markdown("")
    st.markdown("")


def render_manual_prediction(model, n_inputs, feature_names, target_name, task, scaler_X, scaler_y):
    """Render manual prediction input"""

    with st.expander("**Prediction** - Input Custom Values", expanded=False):

        st.markdown(f"Enter values for **{n_inputs}** input features:")

        # Calculate feature ranges and default values from training data
        feature_ranges = []
        default_values = []

        X_train = st.session_state.get('wm_X_train', None)

        if X_train is not None:
            # Get original scale data for calculating ranges
            if scaler_X is not None:
                try:
                    X_train_original = scaler_X.inverse_transform(X_train)
                except:
                    X_train_original = X_train
            else:
                X_train_original = X_train

            # Calculate ranges and midpoints for each feature
            for i in range(n_inputs):
                min_val = X_train_original[:, i].min()
                max_val = X_train_original[:, i].max()
                midpoint = (min_val + max_val) / 2

                feature_ranges.append((min_val, max_val))
                default_values.append(midpoint)
        else:
            # Fallback to 0.0 if no training data available
            feature_ranges = [(0.0, 1.0)] * n_inputs
            default_values = [0.0] * n_inputs

        # Display feature ranges as caption
        if feature_ranges:
            range_text = " | ".join([
                f"**{feature_names[i]}**: [{feature_ranges[i][0]:.2f}, {feature_ranges[i][1]:.2f}]"
                for i in range(n_inputs)
            ])
            st.caption(f"📊 Feature ranges from training data: {range_text}")

        st.markdown("")

        # Create input fields based on number of inputs
        if n_inputs <= 3:
            cols = st.columns(n_inputs)
        else:
            # For more inputs, create rows of 3 columns
            cols_per_row = 3
            n_rows = (n_inputs + cols_per_row - 1) // cols_per_row

            input_values = []
            for row in range(n_rows):
                cols = st.columns(cols_per_row)
                for col_idx in range(cols_per_row):
                    input_idx = row * cols_per_row + col_idx
                    if input_idx < n_inputs:
                        with cols[col_idx]:
                            value = st.number_input(
                                feature_names[input_idx],
                                value=float(default_values[input_idx]),
                                format="%.4f",
                                key=f'wm_manual_input_{input_idx}'
                            )
                            input_values.append(value)

        # For n_inputs <= 3, collect values differently
        if n_inputs <= 3:
            input_values = []
            for i, col in enumerate(cols):
                with col:
                    value = st.number_input(
                        feature_names[i],
                        value=float(default_values[i]),
                        format="%.4f",
                        key=f'wm_manual_input_{i}'
                    )
                    input_values.append(value)

        st.markdown("")

        # Predict button
        col_button, col_result = st.columns([1, 1], vertical_alignment='center')

        with col_button:
            if st.button("Predict", type="primary", use_container_width=True, key='wm_manual_predict_btn'):
                # Create input array
                X_input = np.array(input_values).reshape(1, -1)

                # Apply feature scaling if used during training
                if scaler_X is not None:
                    X_input_scaled = scaler_X.transform(X_input)
                else:
                    X_input_scaled = X_input

                if task == 'classification':
                    # For classification, get class prediction
                    y_pred = model.predict(X_input_scaled)
                    predicted_class = int(y_pred[0]) if hasattr(y_pred, '__getitem__') else int(y_pred)

                    # Store in session state
                    st.session_state.wm_manual_prediction_class = predicted_class
                else:
                    # For regression, just predict
                    y_pred = model.predict(X_input_scaled)
                    y_pred = np.atleast_1d(y_pred)

                    # Inverse transform if target was scaled
                    if scaler_y is not None:
                        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()

                    prediction_value = float(y_pred[0]) if len(y_pred) > 0 else float(y_pred)
                    st.session_state.wm_manual_prediction_result = prediction_value

                st.session_state.wm_manual_prediction_inputs = input_values
                st.rerun()

        with col_result:
            if task == 'classification' and 'wm_manual_prediction_class' in st.session_state:
                # Classification result display
                predicted_class = st.session_state.wm_manual_prediction_class

                st.markdown(
                    f"""
                    <div style="border-left: 4px solid #1f77b4;
                                padding: 1rem 1.25rem;
                                background-color: #f8f9fa;
                                border-radius: 4px;">
                        <div style="font-size: 0.75rem;
                                    color: #6c757d;
                                    text-transform: uppercase;
                                    letter-spacing: 0.5px;
                                    margin-bottom: 0.5rem;">
                            Predicted Class
                        </div>
                        <div style="font-size: 1.5rem;
                                    font-weight: 600;
                                    color: #212529;">
                            Class {predicted_class}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif task == 'regression' and 'wm_manual_prediction_result' in st.session_state:
                # Regression result display
                prediction = st.session_state.wm_manual_prediction_result

                st.markdown(
                    f"""
                    <div style="border-left: 4px solid #1f77b4;
                                padding: 1rem 1.25rem;
                                background-color: #f8f9fa;
                                border-radius: 4px;">
                        <div style="font-size: 0.75rem;
                                    color: #6c757d;
                                    text-transform: uppercase;
                                    letter-spacing: 0.5px;
                                    margin-bottom: 0.5rem;">
                            Predicted {target_name}
                        </div>
                        <div style="font-size: 1.5rem;
                                    font-weight: 600;
                                    color: #212529;
                                    font-family: 'Courier New', monospace;">
                            {prediction:.4f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Show membership degrees
        if 'wm_manual_prediction_inputs' in st.session_state:
            st.markdown("")
            st.markdown("---")
            st.markdown("**Membership Degrees** - Linguistic Term Activations")

            # Get input values
            input_values = st.session_state.wm_manual_prediction_inputs
            X_input = np.array([input_values])

            # Scale if needed
            if scaler_X is not None:
                X_input_scaled = scaler_X.transform(X_input)
            else:
                X_input_scaled = X_input

            # Get membership degrees
            try:
                memberships = model.predict_membership(X_input_scaled)

                # Display in columns
                system = st.session_state.wm_system
                output_names = list(memberships.keys())
                n_outputs = len(output_names)

                if n_outputs <= 3:
                    cols = st.columns(n_outputs)
                else:
                    cols = st.columns(3)

                for idx, output_name in enumerate(output_names):
                    col_idx = idx % len(cols)
                    with cols[col_idx]:
                        st.markdown(f"**{output_name}**")

                        membership_values = memberships[output_name][0]  # First sample
                        output_var = system.output_variables[output_name]
                        term_names = list(output_var.terms.keys())

                        # Create bar chart for memberships
                        fig = go.Figure()

                        colors = ['#667eea' if membership_values[i] == max(membership_values)
                                 else '#a0aec0' for i in range(len(term_names))]

                        fig.add_trace(go.Bar(
                            x=term_names,
                            y=membership_values,
                            marker_color=colors,
                            text=[f'{val:.3f}' for val in membership_values],
                            textposition='outside',
                            hovertemplate='%{x}<br>μ = %{y:.3f}<extra></extra>'
                        ))

                        fig.update_layout(
                            yaxis_title='Membership (μ)',
                            yaxis_range=[0, 1.1],
                            height=250,
                            template='plotly_white',
                            showlegend=False,
                            margin=dict(l=20, r=20, t=20, b=20)
                        )

                        st.plotly_chart(fig, use_container_width=True, key=f'membership_{output_name}')

                        # Show dominant term
                        dominant_idx = np.argmax(membership_values)
                        dominant_term = term_names[dominant_idx]
                        dominant_value = membership_values[dominant_idx]
                        st.caption(f"🎯 Dominant: **{dominant_term}** (μ = {dominant_value:.3f})")

            except Exception as e:
                st.warning(f"Could not compute membership degrees: {str(e)}")

        # Show normalization info if used
        if scaler_X is not None or scaler_y is not None:
            st.markdown("")
            info_parts = []
            if scaler_X is not None:
                info_parts.append(f"Features scaled with {scaler_X.__class__.__name__}")
            if scaler_y is not None:
                info_parts.append(f"Target scaled with {scaler_y.__class__.__name__}")
            st.caption("ℹ️ " + " | ".join(info_parts))


def render_dataset_predictions(model, task, scaler_y):
    """Evaluate model on datasets and export predictions"""

    with st.expander("**Dataset Predictions** - Export & Evaluate", expanded=False):

        # Dataset selection for download and visualization
        available_datasets = []
        if st.session_state.get('wm_X_train', None) is not None:
            available_datasets.append('Train')
        if st.session_state.get('wm_X_test', None) is not None:
            available_datasets.append('Test')

        if not available_datasets:
            st.info("No datasets available. Please split your dataset first.")
            return

        # Radio button to select dataset
        selected_dataset = st.radio(
            "Select dataset:",
            available_datasets,
            horizontal=True,
            key='wm_prediction_export_dataset'
        )

        st.markdown("")

        # Get data based on selection
        if selected_dataset == 'Train':
            X_data = st.session_state.wm_X_train
            y_data = st.session_state.wm_y_train
            y_pred_key = 'wm_y_pred_train'
        else:  # Test
            X_data = st.session_state.wm_X_test
            y_data = st.session_state.wm_y_test
            y_pred_key = 'wm_y_pred_test'

        # Use cached predictions or calculate
        if y_pred_key in st.session_state:
            y_pred = st.session_state[y_pred_key]
        else:
            y_pred = model.predict(X_data)
            st.session_state[y_pred_key] = y_pred

        st.markdown(f"**{selected_dataset} Set Size:** {len(y_data)} samples")

        st.markdown("")

        # Calculate and display metrics based on problem type
        if task == 'classification':
            # Classification metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            # Convert one-hot to labels if needed
            if y_data.ndim == 2:
                y_data_labels = np.argmax(y_data, axis=1)
            else:
                y_data_labels = y_data.ravel().astype(int)

            if y_pred.ndim == 1:
                y_pred_labels = y_pred.astype(int)
            else:
                y_pred_labels = np.argmax(y_pred, axis=1)

            # Calculate metrics
            accuracy = accuracy_score(y_data_labels, y_pred_labels)

            n_classes = len(np.unique(y_data_labels))
            avg_method = 'binary' if n_classes == 2 else 'weighted'

            precision = precision_score(y_data_labels, y_pred_labels, average=avg_method, zero_division=0)
            recall = recall_score(y_data_labels, y_pred_labels, average=avg_method, zero_division=0)
            f1 = f1_score(y_data_labels, y_pred_labels, average=avg_method, zero_division=0)

            # Display classification metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Accuracy", f"{accuracy:.4f}")

            with col2:
                st.metric("Precision", f"{precision:.4f}")

            with col3:
                st.metric("Recall", f"{recall:.4f}")

            with col4:
                st.metric("F1-Score", f"{f1:.4f}")
        else:
            # Regression metrics
            y_data_flat = y_data.ravel()
            y_pred_flat = y_pred.ravel()

            rmse = np.sqrt(np.mean((y_data_flat - y_pred_flat) ** 2))
            mae = np.mean(np.abs(y_data_flat - y_pred_flat))

            ss_res = np.sum((y_data_flat - y_pred_flat) ** 2)
            ss_tot = np.sum((y_data_flat - np.mean(y_data_flat)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            mask = y_data_flat != 0
            mape = np.mean(np.abs((y_data_flat[mask] - y_pred_flat[mask]) / y_data_flat[mask])) * 100 if mask.any() else 0

            # Display regression metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("RMSE", f"{rmse:.4f}")

            with col2:
                st.metric("MAE", f"{mae:.4f}")

            with col3:
                st.metric("R²", f"{r2:.4f}")

            with col4:
                st.metric("MAPE", f"{mape:.2f}%")

        st.markdown("")

        # Scatter plot
        st.markdown(f"**Predicted vs Actual ({selected_dataset} Set)**")

        if task == 'classification':
            y_plot_true = y_data_labels
            y_plot_pred = y_pred_labels
            plot_title = f'{selected_dataset} Set - Class Predictions (Accuracy = {accuracy:.4f})'
        else:
            y_plot_true = y_data_flat
            y_plot_pred = y_pred_flat
            plot_title = f'{selected_dataset} Set - Predicted vs Actual (R² = {r2:.4f})'

        fig = go.Figure()

        # Scatter points
        fig.add_trace(go.Scatter(
            x=y_plot_true,
            y=y_plot_pred,
            mode='markers',
            name='Predictions',
            marker=dict(
                color=np.abs(y_plot_true - y_plot_pred),
                colorscale='Viridis',
                size=8,
                opacity=0.6,
                colorbar=dict(title='|Error|')
            ),
            hovertemplate='Actual: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>'
        ))

        # Perfect prediction line
        min_val = min(y_plot_true.min(), y_plot_pred.min())
        max_val = max(y_plot_true.max(), y_plot_pred.max())

        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash', width=2)
        ))

        fig.update_layout(
            title=plot_title,
            xaxis_title='Actual Values',
            yaxis_title='Predicted Values',
            template='plotly_white',
            height=400,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("")

        # Export results
        if st.button(f"💾 Export {selected_dataset} Results", use_container_width=True, key=f'wm_export_{selected_dataset.lower()}_results'):
            # Inverse transform if scalers were used
            if scaler_y is not None and task == 'regression':
                y_data_original = scaler_y.inverse_transform(y_data).ravel()
                y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
            else:
                y_data_original = y_plot_true
                y_pred_original = y_plot_pred

            # Create results dataframe
            df_results = pd.DataFrame({
                'Actual': y_data_original,
                'Predicted': y_pred_original,
                'Error': y_data_original - y_pred_original,
                'Absolute_Error': np.abs(y_data_original - y_pred_original)
            })

            csv = df_results.to_csv(index=False)

            st.download_button(
                label=f"📥 Download {selected_dataset} Results (CSV)",
                data=csv,
                file_name=f"wang_mendel_{selected_dataset.lower()}_results.csv",
                mime="text/csv",
                use_container_width=True,
                key='wm_download_dataset_results'
            )


def render_batch_prediction(model, feature_names, target_name, task, scaler_X, scaler_y):
    """Render batch prediction from CSV upload"""

    with st.expander("**Batch Prediction** - Upload CSV File", expanded=False):

        st.markdown("Upload a CSV file with the same features used during training:")

        # Show expected format
        with st.expander("Expected CSV Format"):
            example_data = {name: [0.0, 1.0, 2.0] for name in feature_names}
            example_df = pd.DataFrame(example_data)
            st.dataframe(example_df, use_container_width=True)
            st.caption("CSV should have columns: " + ", ".join(feature_names))

        st.markdown("")

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            key='wm_batch_prediction_upload'
        )

        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)

                # Validate columns
                missing_cols = set(feature_names) - set(df.columns)
                if missing_cols:
                    st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
                    return

                # Extract features
                X_batch = df[feature_names].values

                st.success(f"✅ Loaded {len(X_batch)} samples from CSV")

                st.markdown("")

                # Show preview
                with st.expander("Data Preview", expanded=False):
                    st.dataframe(df.head(10), use_container_width=True)

                st.markdown("")

                # Predict button
                if st.button("Predict Batch", type="primary", use_container_width=True, key='wm_batch_predict_btn'):
                    with st.spinner("Making predictions..."):
                        # Apply feature scaling if used during training
                        if scaler_X is not None:
                            X_batch_scaled = scaler_X.transform(X_batch)
                        else:
                            X_batch_scaled = X_batch

                        # Make predictions
                        y_pred = model.predict(X_batch_scaled)

                        # Create results dataframe
                        df_results = df.copy()

                        if task == 'classification':
                            # For classification
                            if y_pred.ndim == 1:
                                predicted_classes = y_pred.astype(int)
                            else:
                                predicted_classes = np.argmax(y_pred, axis=1)

                            df_results['Predicted_Class'] = predicted_classes
                        else:
                            # For regression
                            y_pred_flat = y_pred.ravel()

                            # Inverse transform if target was scaled
                            if scaler_y is not None:
                                y_pred_flat = scaler_y.inverse_transform(y_pred_flat.reshape(-1, 1)).ravel()

                            df_results[f'Predicted_{target_name}'] = y_pred_flat

                        # Store in session state
                        st.session_state.wm_batch_prediction_results = df_results
                        st.session_state.wm_batch_is_classification = (task == 'classification')

                        st.success(f"✅ Predictions completed for {len(y_pred)} samples")
                        st.rerun()

                # Show results if available
                if 'wm_batch_prediction_results' in st.session_state:
                    st.markdown("---")
                    st.markdown("**Prediction Results**")

                    df_results = st.session_state.wm_batch_prediction_results

                    # Display results
                    st.dataframe(df_results, use_container_width=True, height=300)

                    # Statistics
                    st.markdown("**Prediction Statistics**")

                    is_classification = st.session_state.get('wm_batch_is_classification', False)

                    if is_classification:
                        # Classification statistics
                        col1, col2 = st.columns(2)

                        class_counts = df_results['Predicted_Class'].value_counts()
                        total_samples = len(df_results)

                        with col1:
                            st.metric("Total Samples", total_samples)
                            st.metric("Classes Found", len(class_counts))

                        with col2:
                            # Bar chart of class distribution
                            fig_dist = go.Figure()
                            fig_dist.add_trace(go.Bar(
                                x=[f"Class {c}" for c in sorted(class_counts.index)],
                                y=[class_counts[c] for c in sorted(class_counts.index)],
                                marker_color='#667eea'
                            ))
                            fig_dist.update_layout(
                                title='Class Distribution',
                                xaxis_title='Class',
                                yaxis_title='Count',
                                template='plotly_white',
                                height=250,
                                showlegend=False
                            )
                            st.plotly_chart(fig_dist, use_container_width=True)

                    else:
                        # Regression statistics
                        col1, col2, col3, col4 = st.columns(4)

                        pred_col = f'Predicted_{target_name}'

                        with col1:
                            st.metric("Mean", f"{df_results[pred_col].mean():.4f}")

                        with col2:
                            st.metric("Std Dev", f"{df_results[pred_col].std():.4f}")

                        with col3:
                            st.metric("Min", f"{df_results[pred_col].min():.4f}")

                        with col4:
                            st.metric("Max", f"{df_results[pred_col].max():.4f}")

                    st.markdown("")

                    # Download button
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Results (CSV)",
                        data=csv,
                        file_name="wang_mendel_batch_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                    # Visualization
                    if not is_classification:
                        st.markdown("**Prediction Distribution**")

                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=df_results[pred_col],
                            nbinsx=30,
                            marker_color='#667eea',
                            opacity=0.7,
                            name='Predictions'
                        ))

                        fig.update_layout(
                            title='Distribution of Predictions',
                            xaxis_title=f'Predicted {target_name}',
                            yaxis_title='Frequency',
                            template='plotly_white',
                            height=350,
                            showlegend=False
                        )

                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
