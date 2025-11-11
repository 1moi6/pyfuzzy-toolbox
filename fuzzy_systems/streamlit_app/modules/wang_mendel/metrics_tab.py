"""
Metrics Tab for Wang-Mendel Module
Displays training metrics and error analysis
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score


def render():
    """Render metrics tab"""

    if not st.session_state.get('wm_trained', False) or st.session_state.get('wm_model', None) is None:
        st.warning("⚠️ Train a model first to see metrics here (Training tab)")
        return

    # Render sections
    render_overview_metrics()

    st.markdown("")

    render_performance_comparison()

    st.markdown("")

    # Task-specific metrics
    task = st.session_state.get('wm_task', 'regression')

    if task == 'classification':
        render_classification_metrics()
        st.markdown("")

    render_scatter_plots()

    st.markdown("")

    if task == 'regression':
        render_residual_analysis()

    # Add space at the end
    st.markdown("")
    st.markdown("")


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""

    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # MAE
    mae = np.mean(np.abs(y_true - y_pred))

    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else 0

    return {'RMSE': rmse, 'MAE': mae, 'R²': r2, 'MAPE': mape}


def render_overview_metrics():
    """Display overview metrics cards"""

    with st.expander("**Performance Overview** - Quick Metrics", expanded=True):

        task = st.session_state.get('wm_task', 'regression')

        # Get predictions
        y_train = st.session_state.wm_y_train
        y_pred_train = st.session_state.wm_y_pred_train

        # Inverse transform if scaled (only for regression)
        scaler_y = st.session_state.get('wm_scaler_y', None)
        if scaler_y is not None and task == 'regression':
            y_train = scaler_y.inverse_transform(y_train)
            y_pred_train = scaler_y.inverse_transform(y_pred_train)

        if task == 'classification':
            # Convert one-hot encoded data back to class labels
            y_train_labels = np.argmax(y_train, axis=1)

            # Check if predictions are already labels (1D) or one-hot (2D)
            if y_pred_train.ndim == 1:
                y_pred_train_labels = y_pred_train.astype(int)
            else:
                y_pred_train_labels = np.argmax(y_pred_train, axis=1)

            n_classes = y_train.shape[1]
            average_method = 'binary' if n_classes == 2 else 'weighted'

            train_accuracy = accuracy_score(y_train_labels, y_pred_train_labels)
            train_precision = precision_score(y_train_labels, y_pred_train_labels, average=average_method, zero_division=0)
            train_recall = recall_score(y_train_labels, y_pred_train_labels, average=average_method, zero_division=0)
            train_f1 = f1_score(y_train_labels, y_pred_train_labels, average=average_method, zero_division=0)

            # Display classification metrics
            cols = st.columns(4)

            with cols[0]:
                st.metric("Train Accuracy", f"{train_accuracy:.4f}", help="Proportion of correct predictions")

            with cols[1]:
                st.metric("Train Precision", f"{train_precision:.4f}", help="Precision score")

            with cols[2]:
                st.metric("Train Recall", f"{train_recall:.4f}", help="Recall score")

            with cols[3]:
                st.metric("Train F1", f"{train_f1:.4f}", help="F1 score")

            # Test metrics if available
            if st.session_state.get('wm_X_test', None) is not None:
                st.markdown("---")

                y_test = st.session_state.wm_y_test
                y_pred_test = st.session_state.wm_y_pred_test

                # Convert one-hot to labels
                y_test_labels = np.argmax(y_test, axis=1)
                if y_pred_test.ndim == 1:
                    y_pred_test_labels = y_pred_test.astype(int)
                else:
                    y_pred_test_labels = np.argmax(y_pred_test, axis=1)

                test_accuracy = accuracy_score(y_test_labels, y_pred_test_labels)
                test_precision = precision_score(y_test_labels, y_pred_test_labels, average=average_method, zero_division=0)
                test_recall = recall_score(y_test_labels, y_pred_test_labels, average=average_method, zero_division=0)
                test_f1 = f1_score(y_test_labels, y_pred_test_labels, average=average_method, zero_division=0)

                cols = st.columns(4)

                with cols[0]:
                    delta = test_accuracy - train_accuracy
                    st.metric("Test Accuracy", f"{test_accuracy:.4f}", delta=f"{delta:.4f}", delta_color="normal")

                with cols[1]:
                    delta = test_precision - train_precision
                    st.metric("Test Precision", f"{test_precision:.4f}", delta=f"{delta:.4f}", delta_color="normal")

                with cols[2]:
                    delta = test_recall - train_recall
                    st.metric("Test Recall", f"{test_recall:.4f}", delta=f"{delta:.4f}", delta_color="normal")

                with cols[3]:
                    delta = test_f1 - train_f1
                    st.metric("Test F1", f"{test_f1:.4f}", delta=f"{delta:.4f}", delta_color="normal")

        else:
            # Regression metrics
            train_metrics = calculate_metrics(y_train, y_pred_train)

            # Create columns for metrics
            cols = st.columns(4)

            with cols[0]:
                st.metric(
                    "Train RMSE",
                    f"{train_metrics['RMSE']:.4f}",
                    help="Root Mean Squared Error on training set"
                )

            with cols[1]:
                st.metric(
                    "Train MAE",
                    f"{train_metrics['MAE']:.4f}",
                    help="Mean Absolute Error on training set"
                )

            with cols[2]:
                st.metric(
                    "Train R²",
                    f"{train_metrics['R²']:.4f}",
                    help="R-squared (coefficient of determination)"
                )

            with cols[3]:
                st.metric(
                    "Train MAPE",
                    f"{train_metrics['MAPE']:.2f}%",
                    help="Mean Absolute Percentage Error"
                )

            # Test metrics if available
            if st.session_state.get('wm_X_test', None) is not None:
                st.markdown("---")

                y_test = st.session_state.wm_y_test
                y_pred_test = st.session_state.wm_y_pred_test

                # Inverse transform if scaled
                if scaler_y is not None:
                    y_test = scaler_y.inverse_transform(y_test)
                    y_pred_test = scaler_y.inverse_transform(y_pred_test)

                test_metrics = calculate_metrics(y_test, y_pred_test)

                cols = st.columns(4)

                with cols[0]:
                    delta = test_metrics['RMSE'] - train_metrics['RMSE']
                    st.metric(
                        "Test RMSE",
                        f"{test_metrics['RMSE']:.4f}",
                        delta=f"{delta:.4f}",
                        delta_color="inverse"
                    )

                with cols[1]:
                    delta = test_metrics['MAE'] - train_metrics['MAE']
                    st.metric(
                        "Test MAE",
                        f"{test_metrics['MAE']:.4f}",
                        delta=f"{delta:.4f}",
                        delta_color="inverse"
                    )

                with cols[2]:
                    delta = test_metrics['R²'] - train_metrics['R²']
                    st.metric(
                        "Test R²",
                        f"{test_metrics['R²']:.4f}",
                        delta=f"{delta:.4f}",
                        delta_color="normal"
                    )

                with cols[3]:
                    delta = test_metrics['MAPE'] - train_metrics['MAPE']
                    st.metric(
                        "Test MAPE",
                        f"{test_metrics['MAPE']:.2f}%",
                        delta=f"{delta:.2f}%",
                        delta_color="inverse"
                    )


def render_performance_comparison():
    """Compare performance across train/test sets"""

    with st.expander("**Dataset Comparison** - Performance Across Sets", expanded=False):

        task = st.session_state.get('wm_task', 'regression')

        if task == 'classification':
            # Classification comparison
            render_classification_comparison()
        else:
            # Regression comparison
            render_regression_comparison()


def render_classification_comparison():
    """Compare classification metrics across datasets"""

    # Collect metrics for all available datasets
    datasets = []

    # Train
    y_train = st.session_state.wm_y_train
    y_pred_train = st.session_state.wm_y_pred_train

    y_train_labels = np.argmax(y_train, axis=1)
    if y_pred_train.ndim == 1:
        y_pred_train_labels = y_pred_train.astype(int)
    else:
        y_pred_train_labels = np.argmax(y_pred_train, axis=1)

    n_classes = y_train.shape[1]
    average_method = 'binary' if n_classes == 2 else 'weighted'

    train_acc = accuracy_score(y_train_labels, y_pred_train_labels)
    train_prec = precision_score(y_train_labels, y_pred_train_labels, average=average_method, zero_division=0)
    train_rec = recall_score(y_train_labels, y_pred_train_labels, average=average_method, zero_division=0)
    train_f1 = f1_score(y_train_labels, y_pred_train_labels, average=average_method, zero_division=0)

    datasets.append(('Train', {'accuracy': train_acc, 'precision': train_prec, 'recall': train_rec, 'f1': train_f1}, len(y_train)))

    # Test
    if st.session_state.get('wm_X_test', None) is not None:
        y_test = st.session_state.wm_y_test
        y_pred_test = st.session_state.wm_y_pred_test

        y_test_labels = np.argmax(y_test, axis=1)
        if y_pred_test.ndim == 1:
            y_pred_test_labels = y_pred_test.astype(int)
        else:
            y_pred_test_labels = np.argmax(y_pred_test, axis=1)

        test_acc = accuracy_score(y_test_labels, y_pred_test_labels)
        test_prec = precision_score(y_test_labels, y_pred_test_labels, average=average_method, zero_division=0)
        test_rec = recall_score(y_test_labels, y_pred_test_labels, average=average_method, zero_division=0)
        test_f1 = f1_score(y_test_labels, y_pred_test_labels, average=average_method, zero_division=0)

        datasets.append(('Test', {'accuracy': test_acc, 'precision': test_prec, 'recall': test_rec, 'f1': test_f1}, len(y_test)))

    # Create comparison table
    comparison_data = []
    for name, metrics, n_samples in datasets:
        comparison_data.append({
            'Dataset': name,
            'Samples': n_samples,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1']:.4f}"
        })

    df_comparison = pd.DataFrame(comparison_data)

    st.dataframe(df_comparison, use_container_width=True, hide_index=True)

    # Visual comparison
    st.markdown("**Visual Comparison**")

    fig = go.Figure()

    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_keys = ['accuracy', 'precision', 'recall', 'f1']
    colors = ['#667eea', '#f093fb', '#4ade80', '#fbbf24']

    for idx, (metric_name, metric_key) in enumerate(zip(metrics_names, metric_keys)):
        fig.add_trace(go.Bar(
            name=metric_name,
            x=[d[0] for d in datasets],
            y=[d[1][metric_key] for d in datasets],
            marker_color=colors[idx]
        ))

    fig.update_layout(
        title='Classification Metrics Comparison',
        yaxis_title='Score',
        yaxis_range=[0, 1],
        barmode='group',
        template='plotly_white',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def render_regression_comparison():
    """Compare regression metrics across datasets"""

    # Collect metrics for all available datasets
    datasets = []

    scaler_y = st.session_state.get('wm_scaler_y', None)

    # Train
    y_train = st.session_state.wm_y_train
    y_pred_train = st.session_state.wm_y_pred_train

    if scaler_y is not None:
        y_train = scaler_y.inverse_transform(y_train)
        y_pred_train = scaler_y.inverse_transform(y_pred_train)

    train_metrics = calculate_metrics(y_train, y_pred_train)
    datasets.append(('Train', train_metrics, len(y_train)))

    # Test
    if st.session_state.get('wm_X_test', None) is not None:
        y_test = st.session_state.wm_y_test
        y_pred_test = st.session_state.wm_y_pred_test

        if scaler_y is not None:
            y_test = scaler_y.inverse_transform(y_test)
            y_pred_test = scaler_y.inverse_transform(y_pred_test)

        test_metrics = calculate_metrics(y_test, y_pred_test)
        datasets.append(('Test', test_metrics, len(y_test)))

    # Create comparison table
    comparison_data = []
    for name, metrics, n_samples in datasets:
        comparison_data.append({
            'Dataset': name,
            'Samples': n_samples,
            'RMSE': f"{metrics['RMSE']:.4f}",
            'MAE': f"{metrics['MAE']:.4f}",
            'R²': f"{metrics['R²']:.4f}",
            'MAPE': f"{metrics['MAPE']:.2f}%"
        })

    df_comparison = pd.DataFrame(comparison_data)

    st.dataframe(
        df_comparison,
        use_container_width=True,
        hide_index=True
    )

    # Visual comparison with bar charts
    st.markdown("**Visual Comparison**")

    col1, col2 = st.columns(2)

    with col1:
        # RMSE and MAE comparison
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='RMSE',
            x=[d[0] for d in datasets],
            y=[d[1]['RMSE'] for d in datasets],
            marker_color='#667eea'
        ))

        fig.add_trace(go.Bar(
            name='MAE',
            x=[d[0] for d in datasets],
            y=[d[1]['MAE'] for d in datasets],
            marker_color='#f093fb'
        ))

        fig.update_layout(
            title='Error Metrics',
            yaxis_title='Error',
            barmode='group',
            template='plotly_white',
            height=300
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # R² comparison
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=[d[0] for d in datasets],
            y=[d[1]['R²'] for d in datasets],
            marker_color='#4ade80',
            text=[f"{d[1]['R²']:.4f}" for d in datasets],
            textposition='outside'
        ))

        fig.update_layout(
            title='R² Score',
            yaxis_title='R²',
            yaxis_range=[0, 1],
            template='plotly_white',
            height=300,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)


def render_classification_metrics():
    """Display classification-specific metrics"""

    with st.expander("**Classification Details** - Confusion Matrix & Reports", expanded=False):

        # Get data
        y_train = st.session_state.wm_y_train
        y_pred_train = st.session_state.wm_y_pred_train

        # Convert one-hot to labels
        y_train_labels = np.argmax(y_train, axis=1)
        if y_pred_train.ndim == 1:
            y_pred_train_labels = y_pred_train.astype(int)
        else:
            y_pred_train_labels = np.argmax(y_pred_train, axis=1)

        # Get unique classes
        unique_classes = np.unique(y_train_labels)
        class_names = [f"Class {c}" for c in unique_classes]

        # Check if test set available
        has_test = st.session_state.get('wm_X_test', None) is not None

        # Use tabs within the expander for different views
        tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Per-Class Metrics", "Misclassification Analysis"])

        with tab1:
            render_confusion_matrix(y_train_labels, y_pred_train_labels, class_names, unique_classes, "Training")

            if has_test:
                st.markdown("")
                y_test = st.session_state.wm_y_test
                y_pred_test = st.session_state.wm_y_pred_test
                y_test_labels = np.argmax(y_test, axis=1)
                if y_pred_test.ndim == 1:
                    y_pred_test_labels = y_pred_test.astype(int)
                else:
                    y_pred_test_labels = np.argmax(y_pred_test, axis=1)

                render_confusion_matrix(y_test_labels, y_pred_test_labels, class_names, unique_classes, "Test")

        with tab2:
            render_per_class_metrics(y_train_labels, y_pred_train_labels, unique_classes, class_names, "Training")

            if has_test:
                st.markdown("")
                st.markdown("---")
                st.markdown("")
                y_test = st.session_state.wm_y_test
                y_pred_test = st.session_state.wm_y_pred_test
                y_test_labels = np.argmax(y_test, axis=1)
                if y_pred_test.ndim == 1:
                    y_pred_test_labels = y_pred_test.astype(int)
                else:
                    y_pred_test_labels = np.argmax(y_pred_test, axis=1)

                render_per_class_metrics(y_test_labels, y_pred_test_labels, unique_classes, class_names, "Test")

        with tab3:
            render_misclassification_analysis(y_train_labels, y_pred_train_labels, unique_classes, "Training")

            if has_test:
                st.markdown("")
                st.markdown("---")
                st.markdown("")
                y_test = st.session_state.wm_y_test
                y_pred_test = st.session_state.wm_y_pred_test
                y_test_labels = np.argmax(y_test, axis=1)
                if y_pred_test.ndim == 1:
                    y_pred_test_labels = y_pred_test.astype(int)
                else:
                    y_pred_test_labels = np.argmax(y_pred_test, axis=1)

                render_misclassification_analysis(y_test_labels, y_pred_test_labels, unique_classes, "Test")


def render_confusion_matrix(y_true, y_pred, class_names, classes, dataset_name):
    """Render confusion matrix heatmap"""

    st.markdown(f"#### {dataset_name} Set")

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # Normalize for percentage display
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        text=[[f'{cm[i, j]}<br>({cm_normalized[i, j]:.1f}%)' for j in range(len(class_names))] for i in range(len(class_names))],
        texttemplate='%{text}',
        textfont={"size": 12},
        colorscale='Blues',
        showscale=True,
        hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
    ))

    fig.update_layout(
        title=f'Confusion Matrix - {dataset_name}',
        xaxis_title='Predicted Class',
        yaxis_title='True Class',
        template='plotly_white',
        height=400,
        xaxis={'side': 'bottom'},
        yaxis={'autorange': 'reversed'}
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption("Diagonal values show correct predictions. Off-diagonal values show misclassifications.")


def render_per_class_metrics(y_true, y_pred, unique_classes, class_names, dataset_name):
    """Render per-class precision, recall, and F1 score"""

    st.markdown(f"#### {dataset_name} Set - Per-Class Performance")

    # Calculate metrics for each class
    report = classification_report(y_true, y_pred, labels=unique_classes, output_dict=True, zero_division=0)

    # Create dataframe
    metrics_data = []
    for i, cls in enumerate(unique_classes):
        cls_str = str(int(cls))
        if cls_str in report:
            metrics_data.append({
                'Class': class_names[i],
                'Precision': report[cls_str]['precision'],
                'Recall': report[cls_str]['recall'],
                'F1-Score': report[cls_str]['f1-score'],
                'Support': report[cls_str]['support']
            })

    df = pd.DataFrame(metrics_data)

    # Display as dataframe
    st.dataframe(
        df.style.format({
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'F1-Score': '{:.4f}',
            'Support': '{:.0f}'
        }).background_gradient(subset=['Precision', 'Recall', 'F1-Score'], cmap='RdYlGn', vmin=0, vmax=1),
        use_container_width=True
    )

    # Bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(name='Precision', x=df['Class'], y=df['Precision'], marker_color='lightblue'))
    fig.add_trace(go.Bar(name='Recall', x=df['Class'], y=df['Recall'], marker_color='lightgreen'))
    fig.add_trace(go.Bar(name='F1-Score', x=df['Class'], y=df['F1-Score'], marker_color='lightsalmon'))

    fig.update_layout(
        title=f'Per-Class Metrics - {dataset_name}',
        xaxis_title='Class',
        yaxis_title='Score',
        barmode='group',
        template='plotly_white',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def render_misclassification_analysis(y_true, y_pred, unique_classes, dataset_name):
    """Analyze misclassifications"""

    st.markdown(f"#### {dataset_name} Set - Misclassification Analysis")

    # Find misclassified samples
    misclassified = y_true != y_pred
    n_misclassified = misclassified.sum()
    n_total = len(y_true)
    accuracy = (n_total - n_misclassified) / n_total * 100

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Correctly Classified", f"{n_total - n_misclassified} / {n_total}", f"{accuracy:.2f}%")

    with col2:
        error_rate = 100 - accuracy
        st.metric("Misclassified", f"{n_misclassified} / {n_total}", f"{error_rate:.2f}%")

    if n_misclassified > 0:
        st.markdown("")
        st.markdown("**Misclassification Patterns:**")

        # Create misclassification matrix (True class -> Predicted class)
        misclass_pairs = []
        for true_cls in unique_classes:
            for pred_cls in unique_classes:
                if true_cls != pred_cls:
                    mask = (y_true == true_cls) & (y_pred == pred_cls)
                    count = mask.sum()
                    if count > 0:
                        misclass_pairs.append({
                            'True Class': f"Class {int(true_cls)}",
                            'Predicted As': f"Class {int(pred_cls)}",
                            'Count': count,
                            'Percentage': f"{count / n_misclassified * 100:.1f}%"
                        })

        if misclass_pairs:
            df_misclass = pd.DataFrame(misclass_pairs).sort_values('Count', ascending=False)
            st.dataframe(df_misclass, use_container_width=True)
    else:
        st.success("✅ Perfect classification! No misclassifications found.")


def render_scatter_plots():
    """Display predicted vs actual scatter plots"""

    with st.expander("**Prediction Accuracy** - Predicted vs Actual", expanded=False):

        task = st.session_state.get('wm_task', 'regression')
        scaler_y = st.session_state.get('wm_scaler_y', None)

        # Create tabs for different datasets
        available_sets = ['Train']
        if st.session_state.get('wm_X_test', None) is not None:
            available_sets.append('Test')

        tabs = st.tabs(available_sets)

        for i, dataset_name in enumerate(available_sets):
            with tabs[i]:
                if dataset_name == 'Train':
                    y_true = st.session_state.wm_y_train
                    y_pred = st.session_state.wm_y_pred_train
                else:  # Test
                    y_true = st.session_state.wm_y_test
                    y_pred = st.session_state.wm_y_pred_test

                if task == 'classification':
                    # Convert to labels for classification
                    y_true_labels = np.argmax(y_true, axis=1)
                    if y_pred.ndim == 1:
                        y_pred_labels = y_pred.astype(int)
                    else:
                        y_pred_labels = np.argmax(y_pred, axis=1)

                    # Scatter plot
                    fig = go.Figure()

                    # Scatter points
                    fig.add_trace(go.Scatter(
                        x=y_true_labels,
                        y=y_pred_labels,
                        mode='markers',
                        name='Predictions',
                        marker=dict(
                            color=np.abs(y_true_labels - y_pred_labels),
                            colorscale='Viridis',
                            size=8,
                            opacity=0.6,
                            colorbar=dict(title='|Error|')
                        ),
                        hovertemplate='Actual: %{x}<br>Predicted: %{y}<extra></extra>'
                    ))

                    # Perfect prediction line
                    min_val = min(y_true_labels.min(), y_pred_labels.min())
                    max_val = max(y_true_labels.max(), y_pred_labels.max())

                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='red', dash='dash', width=2)
                    ))

                    # Calculate accuracy
                    acc = accuracy_score(y_true_labels, y_pred_labels)
                    title = f'{dataset_name} Set - Predicted vs Actual (Accuracy = {acc:.4f})'

                else:  # Regression
                    # Inverse transform if scaled
                    if scaler_y is not None:
                        y_true = scaler_y.inverse_transform(y_true)
                        y_pred = scaler_y.inverse_transform(y_pred)

                    y_true = y_true.ravel()
                    y_pred = y_pred.ravel()

                    # Scatter plot
                    fig = go.Figure()

                    # Scatter points
                    fig.add_trace(go.Scatter(
                        x=y_true,
                        y=y_pred,
                        mode='markers',
                        name='Predictions',
                        marker=dict(
                            color=np.abs(y_true - y_pred),
                            colorscale='Viridis',
                            size=8,
                            opacity=0.6,
                            colorbar=dict(title='|Error|')
                        ),
                        hovertemplate='Actual: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>'
                    ))

                    # Perfect prediction line
                    min_val = min(y_true.min(), y_pred.min())
                    max_val = max(y_true.max(), y_pred.max())

                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='red', dash='dash', width=2)
                    ))

                    # Calculate R²
                    metrics = calculate_metrics(y_true.reshape(-1, 1), y_pred.reshape(-1, 1))
                    title = f'{dataset_name} Set - Predicted vs Actual (R² = {metrics["R²"]:.4f})'

                fig.update_layout(
                    title=title,
                    xaxis_title='Actual Values',
                    yaxis_title='Predicted Values',
                    template='plotly_white',
                    height=450,
                    showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)


def render_residual_analysis():
    """Display residual analysis plots"""

    with st.expander("**Residual Analysis** - Error Distribution", expanded=False):

        # Select dataset
        available_sets = ['Train']
        if st.session_state.get('wm_X_test', None) is not None:
            available_sets.append('Test')

        dataset_choice = st.radio(
            "Select dataset",
            available_sets,
            horizontal=True,
            key='wm_residual_dataset_choice'
        )

        # Get data based on selection
        scaler_y = st.session_state.get('wm_scaler_y', None)

        if dataset_choice == 'Train':
            y_true = st.session_state.wm_y_train
            y_pred = st.session_state.wm_y_pred_train
        else:  # Test
            if st.session_state.get('wm_X_test', None) is None:
                st.warning("⚠️ Test set not available")
                return
            y_true = st.session_state.wm_y_test
            y_pred = st.session_state.wm_y_pred_test

        # Inverse transform if scaled
        if scaler_y is not None:
            y_true = scaler_y.inverse_transform(y_true)
            y_pred = scaler_y.inverse_transform(y_pred)

        y_true = y_true.ravel()
        y_pred = y_pred.ravel()

        # Calculate residuals
        residuals = y_pred - y_true

        # Create subplot with residual plots
        col1, col2 = st.columns(2)

        with col1:
            # Residual plot
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=y_pred,
                y=residuals,
                mode='markers',
                marker=dict(
                    color='#667eea',
                    size=8,
                    opacity=0.6
                ),
                hovertemplate='Predicted: %{x:.3f}<br>Residual: %{y:.3f}<extra></extra>'
            ))

            # Zero line
            fig.add_hline(y=0, line_dash="dash", line_color="red")

            fig.update_layout(
                title='Residual Plot',
                xaxis_title='Predicted Values',
                yaxis_title='Residuals',
                template='plotly_white',
                height=350
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Residual histogram
            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=residuals,
                nbinsx=30,
                marker_color='#667eea',
                opacity=0.7,
                name='Residuals'
            ))

            # Add zero line
            fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero Error")

            # Add mean line
            mean_error = residuals.mean()
            fig.add_vline(x=mean_error, line_dash="dot", line_color="green", annotation_text=f"Mean: {mean_error:.4f}")

            fig.update_layout(
                title='Residual Distribution',
                xaxis_title='Residual Value',
                yaxis_title='Frequency',
                template='plotly_white',
                height=350,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

        # Residual statistics
        st.markdown("**Residual Statistics**")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Mean", f"{np.mean(residuals):.4f}")

        with col2:
            st.metric("Std Dev", f"{np.std(residuals):.4f}")

        with col3:
            st.metric("Min", f"{np.min(residuals):.4f}")

        with col4:
            st.metric("Max", f"{np.max(residuals):.4f}")
