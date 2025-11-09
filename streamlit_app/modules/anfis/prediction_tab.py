"""
Prediction Tab for ANFIS Module
Handles manual predictions, batch predictions, and test set evaluation
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from io import StringIO


def render():
    """Render prediction tab"""

    if not st.session_state.get('anfis_trained', False) or st.session_state.get('anfis_model', None) is None:
        st.info("🎯 Train a model first to make predictions (Training tab)")
        return

    # Get model
    model = st.session_state.anfis_model
    n_inputs = model.n_inputs

    # Get feature information
    feature_names = st.session_state.get('anfis_feature_names', [f'X{i+1}' for i in range(n_inputs)])
    target_name = st.session_state.get('anfis_target_name', 'Y')
    scaler_X = st.session_state.get('anfis_scaler_X', None)
    scaler_y = st.session_state.get('anfis_scaler_y', None)

    # Render sections
    render_manual_prediction(model, n_inputs, feature_names, target_name, scaler_X, scaler_y)

    st.markdown("")

    render_test_set_evaluation(model, scaler_y)

    st.markdown("")

    render_batch_prediction(model, feature_names, target_name, scaler_X, scaler_y)

    # Add space at the end
    st.markdown("")
    st.markdown("")


def render_manual_prediction(model, n_inputs, feature_names, target_name, scaler_X, scaler_y):
    """Render manual prediction input"""

    with st.expander("**Prediction** - Input Custom Values", expanded=True):

        st.markdown(f"Enter values for **{n_inputs}** input features:")

        # Calculate feature ranges and default values from training data
        feature_ranges = []
        default_values = []

        X_train = st.session_state.get('anfis_X_train', None)

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
                                key=f'manual_input_{input_idx}'
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
                        key=f'manual_input_{i}'
                    )
                    input_values.append(value)

        st.markdown("")

        # Predict button
        col_button, col_result = st.columns([1, 1],vertical_alignment = 'center')

        # Determine if this is a classification problem from session state
        is_classification = st.session_state.get('anfis_problem_type', 'Regression') == 'Classification'

        with col_button:
            if st.button("Predict", type="primary", use_container_width=True, key='manual_predict_btn'):
                # Create input array
                X_input = np.array(input_values).reshape(1, -1)

                # Apply feature scaling if used during training
                if scaler_X is not None:
                    X_input_scaled = scaler_X.transform(X_input)
                else:
                    X_input_scaled = X_input

                if is_classification:
                    # For classification, get raw output and calculate probabilities
                    # Raw output (continuous value representing class membership ~[0,1])
                    raw_output = model.forward_batch(X_input_scaled)[0]

                    # Clip raw output to valid probability range [0, 1]
                    # ANFIS trained with MSE already outputs ~[0,1], we just ensure bounds
                    proba_class1 = float(np.clip(raw_output, 0, 1))
                    proba_class0 = 1.0 - proba_class1
                    proba = np.array([proba_class0, proba_class1])

                    # Predicted class using predict
                    predicted_class = model.predict(X_input_scaled)
                    predicted_class = int(predicted_class) if np.isscalar(predicted_class) else int(predicted_class[0])

                    # Probability of the predicted class
                    prob = proba[predicted_class]

                    # Store in session state
                    st.session_state.manual_prediction_result = raw_output
                    st.session_state.manual_prediction_class = predicted_class
                    st.session_state.manual_prediction_prob = prob
                    st.session_state.manual_prediction_proba = proba  # Both probabilities
                else:
                    # For regression, just predict
                    y_pred = model.predict(X_input_scaled)
                    y_pred = np.atleast_1d(y_pred)

                    # Inverse transform if target was scaled
                    if scaler_y is not None:
                        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()

                    prediction_value = float(y_pred[0]) if len(y_pred) > 0 else float(y_pred)
                    st.session_state.manual_prediction_result = prediction_value

                st.session_state.manual_prediction_inputs = input_values

        with col_result:
            if 'manual_prediction_result' in st.session_state:
                prediction = st.session_state.manual_prediction_result

                if is_classification:
                    # Classification result display - clean and academic
                    predicted_class = st.session_state.get('manual_prediction_class', 0)

                    # Get class names if available (from OvR or dataset)
                    class_names = st.session_state.get('anfis_ovr_class_names', None)
                    if class_names and predicted_class < len(class_names):
                        class_label = class_names[predicted_class]
                    else:
                        class_label = f"Class {predicted_class}"

                    # Modern, clean display
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
                                        color: #212529;
                                        margin-bottom: 0.25rem;">
                                {class_label}
                            </div>
                            <div style="font-size: 0.875rem;
                                        color: #6c757d;
                                        font-family: 'Courier New', monospace;">
                                Output: {prediction:.4f}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    # Regression result display - clean and academic
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

        # Show normalization info if used
        if scaler_X is not None or scaler_y is not None:
            st.markdown("")
            info_parts = []
            if scaler_X is not None:
                info_parts.append(f"Features scaled with {scaler_X.__class__.__name__}")
            if scaler_y is not None:
                info_parts.append(f"Target scaled with {scaler_y.__class__.__name__}")
            st.caption("ℹ️ " + " | ".join(info_parts))


def render_batch_prediction(model, feature_names, target_name, scaler_X, scaler_y):
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
            key='batch_prediction_upload'
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
                if st.button("🔮 Predict Batch", type="primary", use_container_width=True, key='batch_predict_btn'):
                    with st.spinner("Making predictions..."):
                        # Apply feature scaling if used during training
                        if scaler_X is not None:
                            X_batch_scaled = scaler_X.transform(X_batch)
                        else:
                            X_batch_scaled = X_batch

                        # Determine if this is a classification problem from session state
                        is_classification = st.session_state.get('anfis_problem_type', 'Regression') == 'Classification'

                        # Create results dataframe
                        df_results = df.copy()

                        if is_classification:
                            # For classification, get raw output and calculate probabilities
                            # Raw output (continuous values representing class membership ~[0,1])
                            raw_output = model.forward_batch(X_batch_scaled)

                            # Clip raw output to valid probability range [0, 1]
                            # ANFIS trained with MSE already outputs ~[0,1], we just ensure bounds
                            proba_class1 = np.clip(raw_output, 0, 1)
                            proba_class0 = 1.0 - proba_class1

                            # Predicted classes using predict
                            predicted_classes = model.predict(X_batch_scaled)
                            predicted_classes = np.atleast_1d(predicted_classes).astype(int)

                            # Add to dataframe
                            df_results['Raw_Output'] = raw_output
                            df_results['Predicted_Class'] = predicted_classes
                            df_results['Prob_Class_0'] = proba_class0
                            df_results['Prob_Class_1'] = proba_class1

                            # Confidence = probability of predicted class
                            # Use numpy advanced indexing to select probability of predicted class
                            proba = np.column_stack([proba_class0, proba_class1])
                            df_results['Confidence'] = proba[np.arange(len(predicted_classes)), predicted_classes]

                            # Add class labels if available
                            class_names = st.session_state.get('anfis_ovr_class_names', None)
                            if class_names:
                                df_results['Predicted_Label'] = df_results['Predicted_Class'].map(
                                    lambda x: class_names[x] if x < len(class_names) else f"Class {x}"
                                )
                        else:
                            # For regression, just predict
                            y_pred = model.predict(X_batch_scaled)
                            y_pred = np.atleast_1d(y_pred)

                            # Inverse transform if target was scaled
                            if scaler_y is not None:
                                y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()

                            # Add predicted values
                            df_results[f'Predicted_{target_name}'] = y_pred

                        # Store in session state
                        st.session_state.batch_prediction_results = df_results
                        st.session_state.batch_is_classification = is_classification

                        st.success(f"✅ Predictions completed for {len(y_pred)} samples")
                        st.rerun()

                # Show results if available
                if 'batch_prediction_results' in st.session_state:
                    st.markdown("---")
                    st.markdown("**Prediction Results**")

                    df_results = st.session_state.batch_prediction_results

                    # Display results
                    st.dataframe(df_results, use_container_width=True, height=300)

                    # Statistics
                    st.markdown("**Prediction Statistics**")

                    is_classification = st.session_state.get('batch_is_classification', False)

                    if is_classification:
                        # Classification statistics
                        col1, col2, col3, col4 = st.columns(4)

                        class_counts = df_results['Predicted_Class'].value_counts()
                        total_samples = len(df_results)

                        with col1:
                            st.metric("Total Samples", total_samples)

                        with col2:
                            st.metric("Classes Found", len(class_counts))

                        with col3:
                            avg_confidence = df_results['Confidence'].mean()
                            st.metric("Avg Confidence", f"{avg_confidence:.2%}")

                        with col4:
                            min_confidence = df_results['Confidence'].min()
                            st.metric("Min Confidence", f"{min_confidence:.2%}")

                        st.markdown("")

                        # Class distribution
                        st.markdown("**Class Distribution**")
                        col1, col2 = st.columns(2)

                        with col1:
                            for class_val in sorted(class_counts.index):
                                count = class_counts[class_val]
                                pct = (count / total_samples) * 100
                                st.write(f"Class {class_val}: {count} samples ({pct:.1f}%)")

                        with col2:
                            # Bar chart of class distribution
                            fig_dist = go.Figure()
                            fig_dist.add_trace(go.Bar(
                                x=[f"Class {c}" for c in sorted(class_counts.index)],
                                y=[class_counts[c] for c in sorted(class_counts.index)],
                                marker_color='#667eea'
                            ))
                            fig_dist.update_layout(
                                title='',
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
                        file_name="anfis_batch_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                    # Visualization
                    if is_classification:
                        st.markdown("**Confidence Distribution**")

                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=df_results['Confidence'],
                            nbinsx=20,
                            marker_color='#667eea',
                            opacity=0.7,
                            name='Confidence'
                        ))

                        fig.update_layout(
                            title='Distribution of Prediction Confidence',
                            xaxis_title='Confidence',
                            yaxis_title='Frequency',
                            template='plotly_white',
                            height=350,
                            showlegend=False
                        )

                        st.plotly_chart(fig, use_container_width=True)
                    else:
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


def render_test_set_evaluation(model, scaler_y):
    """Evaluate model on all datasets and export predictions"""

    with st.expander("**Dataset Predictions** - Export & Evaluate", expanded=False):

        # Dataset selection for download and visualization
        available_datasets = []
        if st.session_state.get('anfis_X_train', None) is not None:
            available_datasets.append('Train')
        if st.session_state.get('anfis_X_val', None) is not None:
            available_datasets.append('Validation')
        if st.session_state.get('anfis_X_test', None) is not None:
            available_datasets.append('Test')

        if not available_datasets:
            st.info("No datasets available. Please split your dataset first.")
            return

        # Radio button to select dataset
        selected_dataset = st.radio(
            "Select dataset:",
            available_datasets,
            horizontal=True,
            key='prediction_export_dataset'
        )

        st.markdown("")

        # Get data based on selection
        if selected_dataset == 'Train':
            X_data = st.session_state.anfis_X_train
            y_data = st.session_state.anfis_y_train.ravel()
            y_pred_key = 'anfis_y_pred_train'
        elif selected_dataset == 'Validation':
            X_data = st.session_state.anfis_X_val
            y_data = st.session_state.anfis_y_val.ravel()
            y_pred_key = 'anfis_y_pred_val'
        else:  # Test
            X_data = st.session_state.anfis_X_test
            y_data = st.session_state.anfis_y_test.ravel()
            y_pred_key = 'anfis_y_pred_test'

        # Check if predictions already calculated
        is_classification = st.session_state.get('anfis_problem_type', 'Regression') == 'Classification'

        if y_pred_key not in st.session_state:
            if is_classification:
                # For classification, use predict
                y_pred = model.predict(X_data)
                st.session_state[y_pred_key] = np.atleast_1d(y_pred).astype(int)
            else:
                # For regression, use predict to get values
                y_pred = model.predict(X_data)
                st.session_state[y_pred_key] = np.atleast_1d(y_pred)

        y_pred = st.session_state[y_pred_key]

        st.markdown(f"**{selected_dataset} Set Size:** {len(y_data)} samples")

        st.markdown("")

        # Calculate and display metrics based on problem type
        if is_classification:
            # Classification metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            # y_pred already contains predicted classes (from model.predict)
            # y_data should also be class labels
            y_data_class = y_data.astype(int)
            y_pred_class = y_pred.astype(int)

            # Calculate metrics
            accuracy = accuracy_score(y_data_class, y_pred_class)

            # Handle binary vs multiclass
            n_classes = len(np.unique(y_data_class))
            avg_method = 'binary' if n_classes == 2 else 'weighted'

            precision = precision_score(y_data_class, y_pred_class, average=avg_method, zero_division=0)
            recall = recall_score(y_data_class, y_pred_class, average=avg_method, zero_division=0)
            f1 = f1_score(y_data_class, y_pred_class, average=avg_method, zero_division=0)

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
            rmse = np.sqrt(np.mean((y_data - y_pred) ** 2))
            mae = np.mean(np.abs(y_data - y_pred))

            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            mask = y_data != 0
            mape = np.mean(np.abs((y_data[mask] - y_pred[mask]) / y_data[mask])) * 100 if mask.any() else 0

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

        # Scatter plot - use RAW outputs for visualization
        st.markdown(f"**Predicted vs Actual ({selected_dataset} Set)**")

        # Get raw continuous outputs from forward_batch
        y_pred_raw = model.forward_batch(X_data)
        y_pred_raw = np.atleast_1d(y_pred_raw).ravel()

        fig = go.Figure()

        # Scatter points using RAW outputs
        fig.add_trace(go.Scatter(
            x=y_data,
            y=y_pred_raw,
            mode='markers',
            name='Predictions (raw)',
            marker=dict(
                color=np.abs(y_data - y_pred_raw),
                colorscale='Viridis',
                size=8,
                opacity=0.6,
                colorbar=dict(title='|Error|')
            ),
            hovertemplate='Actual: %{x:.3f}<br>Predicted (raw): %{y:.3f}<extra></extra>'
        ))

        # Perfect prediction line
        min_val = min(y_data.min(), y_pred_raw.min())
        max_val = max(y_data.max(), y_pred_raw.max())

        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash', width=2)
        ))

        # Title based on problem type
        if is_classification:
            # For classification, show accuracy in title (using discrete classes)
            plot_title = f'{selected_dataset} Set - Raw Output vs Actual (Accuracy = {accuracy:.4f})'
        else:
            # For regression, show R² in title (using raw values)
            from sklearn.metrics import r2_score
            r2_plot = r2_score(y_data, y_pred_raw)
            plot_title = f'{selected_dataset} Set - Predicted vs Actual (R² = {r2_plot:.4f})'

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
        if st.button(f"💾 Export {selected_dataset} Results", use_container_width=True, key=f'export_{selected_dataset.lower()}_results'):
            # Inverse transform if scalers were used
            if scaler_y is not None:
                y_data_original = scaler_y.inverse_transform(y_data.reshape(-1, 1)).ravel()
                y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
            else:
                y_data_original = y_data
                y_pred_original = y_pred

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
                file_name=f"anfis_{selected_dataset.lower()}_results.csv",
                mime="text/csv",
                use_container_width=True,
                key='download_test_results'
            )
