"""
Metrics Tab for ANFIS Module
Displays training metrics, convergence plots, and error analysis
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

    if not st.session_state.get('anfis_trained', False) or st.session_state.get('anfis_model', None) is None:
        st.info("🎯 Train a model first to see metrics here (Training tab)")
        return

    # Get model and data
    model = st.session_state.anfis_model

    # Calculate predictions if not already done
    calculate_predictions()

    # Render sections
    render_overview_metrics()

    st.markdown("")

    render_convergence_plots()

    st.markdown("")

    render_performance_comparison()

    st.markdown("")

    # Classification metrics (only for classification problems)
    if is_classification_problem():
        render_classification_metrics()
        st.markdown("")

    render_scatter_plots()

    st.markdown("")

    render_residual_analysis()

    # Add space at the end
    st.markdown("")
    st.markdown("")


def calculate_predictions():
    """Calculate predictions for all datasets if not cached"""

    model = st.session_state.anfis_model

    # Determine problem type
    is_classification = st.session_state.get('anfis_problem_type', 'Regression') == 'Classification'

    # Train predictions
    if 'anfis_y_pred_train' not in st.session_state:
        y_pred = model.predict(st.session_state.anfis_X_train)
        if is_classification:
            st.session_state.anfis_y_pred_train = np.atleast_1d(y_pred).astype(int)
        else:
            st.session_state.anfis_y_pred_train = np.atleast_1d(y_pred)

    # Validation predictions
    if st.session_state.get('anfis_X_val', None) is not None and 'anfis_y_pred_val' not in st.session_state:
        y_pred = model.predict(st.session_state.anfis_X_val)
        if is_classification:
            st.session_state.anfis_y_pred_val = np.atleast_1d(y_pred).astype(int)
        else:
            st.session_state.anfis_y_pred_val = np.atleast_1d(y_pred)

    # Test predictions
    if st.session_state.get('anfis_X_test', None) is not None and 'anfis_y_pred_test' not in st.session_state:
        y_pred = model.predict(st.session_state.anfis_X_test)
        if is_classification:
            st.session_state.anfis_y_pred_test = np.atleast_1d(y_pred).astype(int)
        else:
            st.session_state.anfis_y_pred_test = np.atleast_1d(y_pred)


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

        # Check if classification problem
        is_classification = is_classification_problem()

        # Get predictions
        y_train = st.session_state.anfis_y_train.ravel()
        y_pred_train = st.session_state.anfis_y_pred_train

        if is_classification:
            # Classification metrics
            y_train_class = np.round(y_train).astype(int)
            y_pred_train_class = np.round(y_pred_train).astype(int)

            n_classes = len(np.unique(y_train_class))
            average_method = 'binary' if n_classes == 2 else 'weighted'

            train_accuracy = accuracy_score(y_train_class, y_pred_train_class)
            train_precision = precision_score(y_train_class, y_pred_train_class, average=average_method, zero_division=0)
            train_recall = recall_score(y_train_class, y_pred_train_class, average=average_method, zero_division=0)
            train_f1 = f1_score(y_train_class, y_pred_train_class, average=average_method, zero_division=0)

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

            # Validation metrics if available
            if st.session_state.get('anfis_X_val', None) is not None:
                st.markdown("---")

                y_val = st.session_state.anfis_y_val.ravel()
                y_pred_val = st.session_state.anfis_y_pred_val
                y_val_class = np.round(y_val).astype(int)
                y_pred_val_class = np.round(y_pred_val).astype(int)

                val_accuracy = accuracy_score(y_val_class, y_pred_val_class)
                val_precision = precision_score(y_val_class, y_pred_val_class, average=average_method, zero_division=0)
                val_recall = recall_score(y_val_class, y_pred_val_class, average=average_method, zero_division=0)
                val_f1 = f1_score(y_val_class, y_pred_val_class, average=average_method, zero_division=0)

                cols = st.columns(4)

                with cols[0]:
                    delta = val_accuracy - train_accuracy
                    st.metric("Val Accuracy", f"{val_accuracy:.4f}", delta=f"{delta:.4f}", delta_color="normal")

                with cols[1]:
                    delta = val_precision - train_precision
                    st.metric("Val Precision", f"{val_precision:.4f}", delta=f"{delta:.4f}", delta_color="normal")

                with cols[2]:
                    delta = val_recall - train_recall
                    st.metric("Val Recall", f"{val_recall:.4f}", delta=f"{delta:.4f}", delta_color="normal")

                with cols[3]:
                    delta = val_f1 - train_f1
                    st.metric("Val F1", f"{val_f1:.4f}", delta=f"{delta:.4f}", delta_color="normal")

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

            # Validation metrics if available
            if st.session_state.get('anfis_X_val', None) is not None:
                st.markdown("---")

                y_val = st.session_state.anfis_y_val.ravel()
                y_pred_val = st.session_state.anfis_y_pred_val

                val_metrics = calculate_metrics(y_val, y_pred_val)

                cols = st.columns(4)

                with cols[0]:
                    delta = val_metrics['RMSE'] - train_metrics['RMSE']
                    st.metric(
                        "Val RMSE",
                        f"{val_metrics['RMSE']:.4f}",
                        delta=f"{delta:.4f}",
                        delta_color="inverse"
                    )

                with cols[1]:
                    delta = val_metrics['MAE'] - train_metrics['MAE']
                    st.metric(
                        "Val MAE",
                        f"{val_metrics['MAE']:.4f}",
                        delta=f"{delta:.4f}",
                        delta_color="inverse"
                    )

                with cols[2]:
                    delta = val_metrics['R²'] - train_metrics['R²']
                    st.metric(
                        "Val R²",
                        f"{val_metrics['R²']:.4f}",
                        delta=f"{delta:.4f}",
                        delta_color="normal"
                    )

                with cols[3]:
                    delta = val_metrics['MAPE'] - train_metrics['MAPE']
                    st.metric(
                        "Val MAPE",
                        f"{val_metrics['MAPE']:.2f}%",
                        delta=f"{delta:.2f}%",
                        delta_color="inverse"
                    )


def render_convergence_plots():
    """Display training convergence plots"""

    with st.expander("**Training Convergence** - Training History", expanded=True):

        model = st.session_state.anfis_model

        # Check for available training history
        has_hybrid = hasattr(model, 'history') and model.history is not None
        has_metaheuristic = hasattr(model, 'metaheuristic_history') and model.metaheuristic_history is not None
        has_old_metrics = hasattr(model, 'convergence_metrics') and model.convergence_metrics

        if not has_hybrid and not has_metaheuristic and not has_old_metrics:
            st.info("No training history available for this model")
            return

        # Create tabs for different visualizations
        tabs_list = []

        if has_hybrid:
            tabs_list.append("📊 Metrics Evolution")
        if has_metaheuristic:
            tabs_list.append("🔬 Metaheuristic Convergence")
        if hasattr(model, 'total_cost_history') and len(model.total_cost_history) > 0:
            tabs_list.append("⚖️ Regularization")
        if has_old_metrics:
            tabs_list.append("📉 Error History")

        if not tabs_list:
            st.info("No training history available")
            return

        tabs = st.tabs(tabs_list)
        tab_idx = 0

        # Tab: Metrics Evolution (hybrid training)
        if has_hybrid:
            with tabs[tab_idx]:
                render_metrics_evolution(model)
            tab_idx += 1

        # Tab: Metaheuristic Convergence
        if has_metaheuristic:
            with tabs[tab_idx]:
                render_metaheuristic_convergence(model)
            tab_idx += 1

        # Tab: Regularization
        if hasattr(model, 'total_cost_history') and len(model.total_cost_history) > 0:
            with tabs[tab_idx]:
                render_regularization_plots(model)
            tab_idx += 1

        # Tab: Error History (old format)
        if has_old_metrics:
            with tabs[tab_idx]:
                render_error_history(model)


def render_metrics_evolution(model):
    """Render metrics evolution during hybrid training"""

    history = model.history

    if 'train' not in history or 'loss' not in history['train']:
        st.info("No hybrid training history available")
        return

    # Get available metrics
    train_metrics = history['train']
    val_metrics = history.get('val', {})
    has_val = len(val_metrics) > 0

    # Available metric names
    available_metrics = list(train_metrics.keys())

    # Classification vs regression from session state
    is_classification = st.session_state.get('anfis_problem_type', 'Regression') == 'Classification'

    st.markdown("**Training Metrics Evolution**")

    # Metric selector
    if is_classification:
        default_metrics = ['loss', 'accuracy', 'f1_score']
    else:
        default_metrics = ['loss', 'rmse', 'r2']

    # Filter to only available metrics
    default_metrics = [m for m in default_metrics if m in available_metrics]

    selected_metrics = st.multiselect(
        "Select metrics to display",
        available_metrics,
        default=default_metrics,
        key='metrics_selector'
    )

    if not selected_metrics:
        st.warning("Select at least one metric")
        return

    st.markdown("")

    # Create subplots
    n_metrics = len(selected_metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[m.upper().replace('_', ' ') for m in selected_metrics]
    )

    epochs = list(range(1, len(train_metrics['loss']) + 1))

    for idx, metric in enumerate(selected_metrics):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        # Train line
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=train_metrics[metric],
                mode='lines',
                name=f'Train {metric}',
                line=dict(color='#667eea', width=2),
                legendgroup=f'group{idx}',
                showlegend=(idx == 0),
                hovertemplate=f'Epoch: %{{x}}<br>Train: %{{y:.4f}}<extra></extra>'
            ),
            row=row, col=col
        )

        # Val line if available
        if has_val and metric in val_metrics and len(val_metrics[metric]) > 0:
            fig.add_trace(
                go.Scatter(
                    x=epochs[:len(val_metrics[metric])],
                    y=val_metrics[metric],
                    mode='lines',
                    name=f'Val {metric}',
                    line=dict(color='#f093fb', width=2, dash='dash'),
                    legendgroup=f'group{idx}',
                    showlegend=(idx == 0),
                    hovertemplate=f'Epoch: %{{x}}<br>Val: %{{y:.4f}}<extra></extra>'
                ),
                row=row, col=col
            )

    fig.update_xaxes(title_text="Epoch")
    fig.update_layout(
        template='plotly_white',
        height=300 * n_rows,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, width="stretch")

    # Summary statistics
    st.markdown("**Training Summary**")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Epochs", len(epochs))

    with col2:
        final_train_loss = train_metrics['loss'][-1]
        st.metric("Final Train Loss", f"{final_train_loss:.4f}")

    with col3:
        if has_val and 'loss' in val_metrics:
            final_val_loss = val_metrics['loss'][-1]
            st.metric("Final Val Loss", f"{final_val_loss:.4f}")


def render_metaheuristic_convergence(model):
    """Render metaheuristic optimization convergence"""

    meta_hist = model.metaheuristic_history

    if 'convergence' not in meta_hist:
        st.info("No metaheuristic convergence data available")
        return

    conv = meta_hist['convergence']
    iterations = [c['iteration'] for c in conv]
    train_loss = [c['train_loss'] for c in conv]
    val_loss = [c['val_loss'] for c in conv if c['val_loss'] is not None]
    has_val = len(val_loss) > 0

    st.markdown("**Metaheuristic Optimization Convergence**")

    # Info box
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Optimizer", meta_hist['optimizer'].upper())

    with col2:
        st.metric("Population", meta_hist['n_particles'])

    with col3:
        st.metric("Iterations", f"{len(iterations)}/{meta_hist['n_iterations']}")

    with col4:
        st.metric("Best Fitness", f"{meta_hist['best_fitness']:.6f}")

    st.markdown("")

    # Convergence plots (linear and log scale)
    col1, col2 = st.columns(2)

    with col1:
        # Linear scale
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=iterations,
            y=train_loss,
            mode='lines',
            name='Train Loss',
            line=dict(color='#667eea', width=2.5)
        ))

        if has_val:
            fig.add_trace(go.Scatter(
                x=iterations[:len(val_loss)],
                y=val_loss,
                mode='lines',
                name='Val Loss',
                line=dict(color='#f093fb', width=2.5, dash='dash')
            ))

        if meta_hist.get('early_stopped', False):
            best_iter = meta_hist['best_iteration']
            fig.add_vline(
                x=best_iter,
                line_dash="dot",
                line_color="green",
                line_width=2,
                annotation_text=f"Best (iter {best_iter})"
            )

        fig.update_layout(
            title='Convergence (Linear Scale)',
            xaxis_title='Iteration',
            yaxis_title='Loss (MSE)',
            template='plotly_white',
            height=350
        )

        st.plotly_chart(fig, width="stretch")

    with col2:
        # Log scale
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=iterations,
            y=train_loss,
            mode='lines',
            name='Train Loss',
            line=dict(color='#667eea', width=2.5)
        ))

        if has_val:
            fig.add_trace(go.Scatter(
                x=iterations[:len(val_loss)],
                y=val_loss,
                mode='lines',
                name='Val Loss',
                line=dict(color='#f093fb', width=2.5, dash='dash')
            ))

        if meta_hist.get('early_stopped', False):
            best_iter = meta_hist['best_iteration']
            fig.add_vline(
                x=best_iter,
                line_dash="dot",
                line_color="green",
                line_width=2,
                annotation_text=f"Best (iter {best_iter})"
            )

        fig.update_layout(
            title='Convergence (Log Scale)',
            xaxis_title='Iteration',
            yaxis_title='Loss (MSE)',
            yaxis_type="log",
            template='plotly_white',
            height=350
        )

        st.plotly_chart(fig, width="stretch")

    # Early stopping info
    if meta_hist.get('early_stopped', False):
        st.warning(f"⚠️ Early stopping triggered at iteration {meta_hist.get('best_iteration', 'N/A')}")


def render_regularization_plots(model):
    """Render regularization penalty evolution"""

    st.markdown("**Regularization Penalties Evolution**")
    st.caption("Evolution of L1 and L2 penalties during training")

    st.markdown("")

    epochs = list(range(1, len(model.total_cost_history) + 1))

    # Create 3-column layout
    col1, col2, col3 = st.columns(3)

    with col1:
        # Total cost
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=epochs,
            y=model.total_cost_history,
            mode='lines',
            line=dict(color='#667eea', width=2),
            hovertemplate='Epoch: %{x}<br>Total Cost: %{y:.4f}<extra></extra>'
        ))

        fig.update_layout(
            title='Total Cost<br>J = MSE + λ₁L1 + λ₂L2',
            xaxis_title='Epoch',
            yaxis_title='Total Cost',
            template='plotly_white',
            height=300
        )

        st.plotly_chart(fig, width="stretch")

    with col2:
        # L1 penalty
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=epochs,
            y=model.l1_history,
            mode='lines',
            line=dict(color='#f87171', width=2),
            hovertemplate='Epoch: %{x}<br>L1: %{y:.4f}<extra></extra>'
        ))

        lambda_l1 = model.lambda_l1 if hasattr(model, 'lambda_l1') else 0
        fig.update_layout(
            title=f'L1 Penalty<br>λ₁ = {lambda_l1}',
            xaxis_title='Epoch',
            yaxis_title='L1 Penalty',
            template='plotly_white',
            height=300
        )

        st.plotly_chart(fig, width="stretch")

    with col3:
        # L2 penalty
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=epochs,
            y=model.l2_history,
            mode='lines',
            line=dict(color='#4ade80', width=2),
            hovertemplate='Epoch: %{x}<br>L2: %{y:.4f}<extra></extra>'
        ))

        lambda_l2 = model.lambda_l2 if hasattr(model, 'lambda_l2') else 0
        fig.update_layout(
            title=f'L2 Penalty<br>λ₂ = {lambda_l2}',
            xaxis_title='Epoch',
            yaxis_title='L2 Penalty',
            template='plotly_white',
            height=300
        )

        st.plotly_chart(fig, width="stretch")

    # Regularization type info
    st.markdown("")
    lambda_l1 = model.lambda_l1 if hasattr(model, 'lambda_l1') else 0
    lambda_l2 = model.lambda_l2 if hasattr(model, 'lambda_l2') else 0

    if lambda_l1 > 0 and lambda_l2 > 0:
        reg_type = "Elastic Net (L1 + L2)"
    elif lambda_l1 > 0:
        reg_type = "Lasso (L1)"
    elif lambda_l2 > 0:
        reg_type = "Ridge (L2)"
    else:
        reg_type = "No regularization"

    st.info(f"**Regularization Type:** {reg_type}")


def render_error_history(model):
    """Render error history (old format for backwards compatibility)"""

    metrics = model.convergence_metrics

    if 'train_errors' not in metrics or len(metrics['train_errors']) == 0:
        st.info("No error history available")
        return

    st.markdown("**Error History**")

    # Create convergence plot
    fig = go.Figure()

    epochs = list(range(1, len(metrics['train_errors']) + 1))

    # Training error
    fig.add_trace(go.Scatter(
        x=epochs,
        y=metrics['train_errors'],
        mode='lines',
        name='Train Error',
        line=dict(color='#667eea', width=2),
        hovertemplate='Epoch: %{x}<br>Train Error: %{y:.4f}<extra></extra>'
    ))

    # Validation error if available
    if 'val_errors' in metrics and len(metrics['val_errors']) > 0:
        fig.add_trace(go.Scatter(
            x=epochs,
            y=metrics['val_errors'],
            mode='lines',
            name='Val Error',
            line=dict(color='#f093fb', width=2, dash='dash'),
            hovertemplate='Epoch: %{x}<br>Val Error: %{y:.4f}<extra></extra>'
        ))

    # Mark best epoch if available
    if 'best_epoch' in metrics:
        best_epoch = metrics['best_epoch']
        best_val_error = metrics['val_errors'][best_epoch - 1] if 'val_errors' in metrics else None

        if best_val_error:
            fig.add_trace(go.Scatter(
                x=[best_epoch],
                y=[best_val_error],
                mode='markers',
                name='Best Model',
                marker=dict(color='#4ade80', size=12, symbol='star'),
                hovertemplate=f'Best Epoch: {best_epoch}<br>Val Error: {best_val_error:.4f}<extra></extra>'
            ))

    fig.update_layout(
        title='Training Error Convergence',
        xaxis_title='Epoch',
        yaxis_title='Error (RMSE)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )

    st.plotly_chart(fig, width="stretch")

    # Show convergence statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Epochs", len(metrics['train_errors']))

    with col2:
        final_train_error = metrics['train_errors'][-1]
        st.metric("Final Train Error", f"{final_train_error:.4f}")

    with col3:
        if 'val_errors' in metrics and len(metrics['val_errors']) > 0:
            final_val_error = metrics['val_errors'][-1]
            st.metric("Final Val Error", f"{final_val_error:.4f}")
        else:
            st.metric("Final Val Error", "N/A")

    # Early stopping info
    if 'early_stopped' in metrics and metrics['early_stopped']:
        st.info(f"✋ Early stopping triggered at epoch {len(metrics['train_errors'])}")


def render_performance_comparison():
    """Compare performance across train/val/test sets"""

    with st.expander("**Dataset Comparison** - Performance Across Sets", expanded=False):

        # Check if classification
        is_classification = is_classification_problem()

        if is_classification:
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
    y_train = st.session_state.anfis_y_train.ravel()
    y_pred_train = st.session_state.anfis_y_pred_train
    y_train_class = np.round(y_train).astype(int)
    y_pred_train_class = np.round(y_pred_train).astype(int)

    n_classes = len(np.unique(y_train_class))
    average_method = 'binary' if n_classes == 2 else 'weighted'

    train_acc = accuracy_score(y_train_class, y_pred_train_class)
    train_prec = precision_score(y_train_class, y_pred_train_class, average=average_method, zero_division=0)
    train_rec = recall_score(y_train_class, y_pred_train_class, average=average_method, zero_division=0)
    train_f1 = f1_score(y_train_class, y_pred_train_class, average=average_method, zero_division=0)

    datasets.append(('Train', {'accuracy': train_acc, 'precision': train_prec, 'recall': train_rec, 'f1': train_f1}, len(y_train)))

    # Validation
    if st.session_state.get('anfis_X_val', None) is not None:
        y_val = st.session_state.anfis_y_val.ravel()
        y_pred_val = st.session_state.anfis_y_pred_val
        y_val_class = np.round(y_val).astype(int)
        y_pred_val_class = np.round(y_pred_val).astype(int)

        val_acc = accuracy_score(y_val_class, y_pred_val_class)
        val_prec = precision_score(y_val_class, y_pred_val_class, average=average_method, zero_division=0)
        val_rec = recall_score(y_val_class, y_pred_val_class, average=average_method, zero_division=0)
        val_f1 = f1_score(y_val_class, y_pred_val_class, average=average_method, zero_division=0)

        datasets.append(('Validation', {'accuracy': val_acc, 'precision': val_prec, 'recall': val_rec, 'f1': val_f1}, len(y_val)))

    # Test
    if st.session_state.get('anfis_X_test', None) is not None:
        y_test = st.session_state.anfis_y_test.ravel()
        y_pred_test = st.session_state.anfis_y_pred_test
        y_test_class = np.round(y_test).astype(int)
        y_pred_test_class = np.round(y_pred_test).astype(int)

        test_acc = accuracy_score(y_test_class, y_pred_test_class)
        test_prec = precision_score(y_test_class, y_pred_test_class, average=average_method, zero_division=0)
        test_rec = recall_score(y_test_class, y_pred_test_class, average=average_method, zero_division=0)
        test_f1 = f1_score(y_test_class, y_pred_test_class, average=average_method, zero_division=0)

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

    st.dataframe(df_comparison, width="stretch", hide_index=True)

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

    st.plotly_chart(fig, width="stretch")


def render_regression_comparison():
    """Compare regression metrics across datasets"""

    # Collect metrics for all available datasets
    datasets = []

    # Train
    y_train = st.session_state.anfis_y_train.ravel()
    y_pred_train = st.session_state.anfis_y_pred_train
    train_metrics = calculate_metrics(y_train, y_pred_train)
    datasets.append(('Train', train_metrics, len(y_train)))

    # Validation
    if st.session_state.get('anfis_X_val', None) is not None:
        y_val = st.session_state.anfis_y_val.ravel()
        y_pred_val = st.session_state.anfis_y_pred_val
        val_metrics = calculate_metrics(y_val, y_pred_val)
        datasets.append(('Validation', val_metrics, len(y_val)))

    # Test
    if st.session_state.get('anfis_X_test', None) is not None:
        y_test = st.session_state.anfis_y_test.ravel()
        y_pred_test = st.session_state.anfis_y_pred_test
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
        width="stretch",
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

        st.plotly_chart(fig, width="stretch")

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

        st.plotly_chart(fig, width="stretch")


def render_scatter_plots():
    """Display predicted vs actual scatter plots"""

    with st.expander("**Prediction Accuracy** - Predicted vs Actual", expanded=False):

        # Check if classification
        is_classification = is_classification_problem()
        model = st.session_state.anfis_model

        # Create tabs for different datasets
        available_sets = ['Train']
        if st.session_state.get('anfis_X_val', None) is not None:
            available_sets.append('Validation')
        if st.session_state.get('anfis_X_test', None) is not None:
            available_sets.append('Test')

        tabs = st.tabs(available_sets)

        for i, dataset_name in enumerate(available_sets):
            with tabs[i]:
                if dataset_name == 'Train':
                    y_true = st.session_state.anfis_y_train.ravel()
                    y_pred = st.session_state.anfis_y_pred_train
                    X_data = st.session_state.anfis_X_train
                elif dataset_name == 'Validation':
                    y_true = st.session_state.anfis_y_val.ravel()
                    y_pred = st.session_state.anfis_y_pred_val
                    X_data = st.session_state.anfis_X_val
                else:  # Test
                    y_true = st.session_state.anfis_y_test.ravel()
                    y_pred = st.session_state.anfis_y_pred_test
                    X_data = st.session_state.anfis_X_test

                # For visualization, always use raw continuous outputs from forward_batch
                y_pred_raw = model.forward_batch(X_data)
                y_pred_raw = np.atleast_1d(y_pred_raw).ravel()

                # Create scatter plot using RAW outputs
                fig = go.Figure()

                # Scatter points
                fig.add_trace(go.Scatter(
                    x=y_true,
                    y=y_pred_raw,
                    mode='markers',
                    name='Predictions (raw)',
                    marker=dict(
                        color=np.abs(y_true - y_pred_raw),
                        colorscale='Viridis',
                        size=8,
                        opacity=0.6,
                        colorbar=dict(title='|Error|')
                    ),
                    hovertemplate='Actual: %{x:.3f}<br>Predicted (raw): %{y:.3f}<extra></extra>'
                ))

                # Perfect prediction line
                min_val = min(y_true.min(), y_pred_raw.min())
                max_val = max(y_true.max(), y_pred_raw.max())

                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash', width=2)
                ))

                # Title based on problem type
                if is_classification:
                    # For classification, show accuracy using discrete classes
                    y_pred_class = y_pred  # Already contains discrete classes
                    acc = accuracy_score(y_true, y_pred_class)
                    title = f'{dataset_name} Set - Raw Output vs Actual (Accuracy = {acc:.4f})'
                else:
                    # For regression, show R² using raw values
                    from sklearn.metrics import r2_score
                    r2 = r2_score(y_true, y_pred_raw)
                    title = f'{dataset_name} Set - Predicted vs Actual (R² = {r2:.4f})'

                fig.update_layout(
                    title=title,
                    xaxis_title='Actual Values',
                    yaxis_title='Predicted Values',
                    template='plotly_white',
                    height=450,
                    showlegend=True
                )

                st.plotly_chart(fig, width="stretch")


def render_residual_analysis():
    """Display residual analysis plots"""

    with st.expander("**Residual Analysis** - Error Distribution", expanded=False):

        # Select dataset
        dataset_choice = st.radio(
            "Select dataset",
            ['Train', 'Validation', 'Test'],
            horizontal=True,
            key='residual_dataset_choice'
        )

        # Get data based on selection
        if dataset_choice == 'Train':
            y_true = st.session_state.anfis_y_train.ravel()
            y_pred = st.session_state.anfis_y_pred_train
        elif dataset_choice == 'Validation':
            if st.session_state.get('anfis_X_val', None) is None:
                st.warning("Validation set not available")
                return
            y_true = st.session_state.anfis_y_val.ravel()
            y_pred = st.session_state.anfis_y_pred_val
        else:  # Test
            if st.session_state.get('anfis_X_test', None) is None:
                st.warning("Test set not available")
                return
            y_true = st.session_state.anfis_y_test.ravel()
            y_pred = st.session_state.anfis_y_pred_test

        # Calculate residuals
        residuals = y_true - y_pred

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

            st.plotly_chart(fig, width="stretch")

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

            fig.update_layout(
                title='Residual Distribution',
                xaxis_title='Residual Value',
                yaxis_title='Frequency',
                template='plotly_white',
                height=350,
                showlegend=False
            )

            st.plotly_chart(fig, width="stretch")

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


def is_classification_problem():
    """Check if the problem is classification or regression based on user selection"""
    return st.session_state.get('anfis_problem_type', 'Regression') == 'Classification'


def render_classification_metrics():
    """Display classification-specific metrics"""

    with st.expander("**Classification Metrics** - Model Performance", expanded=True):

        st.markdown("**Classification Performance Metrics**")
        st.caption("Metrics specific to classification tasks")

        st.markdown("")

        # Get true labels and predictions
        y_train = st.session_state.anfis_y_train.ravel()
        y_pred_train = st.session_state.anfis_y_pred_train

        # Convert predictions to class labels (round to nearest integer)
        y_pred_train_class = np.round(y_pred_train).astype(int)
        y_train_class = np.round(y_train).astype(int)

        # Get unique classes
        classes = np.unique(y_train_class)
        n_classes = len(classes)

        # Determine if binary or multiclass
        is_binary = n_classes == 2
        average_method = 'binary' if is_binary else 'weighted'

        # Calculate metrics for train set
        train_accuracy = accuracy_score(y_train_class, y_pred_train_class)
        train_precision = precision_score(y_train_class, y_pred_train_class, average=average_method, zero_division=0)
        train_recall = recall_score(y_train_class, y_pred_train_class, average=average_method, zero_division=0)
        train_f1 = f1_score(y_train_class, y_pred_train_class, average=average_method, zero_division=0)

        # Display train metrics
        st.markdown("**Training Set**")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Accuracy", f"{train_accuracy:.4f}", help="Proportion of correct predictions")

        with col2:
            st.metric("Precision", f"{train_precision:.4f}", help="Proportion of positive predictions that were correct")

        with col3:
            st.metric("Recall", f"{train_recall:.4f}", help="Proportion of actual positives that were predicted correctly")

        with col4:
            st.metric("F1-Score", f"{train_f1:.4f}", help="Harmonic mean of precision and recall")

        # Validation metrics if available
        if st.session_state.get('anfis_X_val', None) is not None:
            st.markdown("---")
            st.markdown("**Validation Set**")

            y_val = st.session_state.anfis_y_val.ravel()
            y_pred_val = st.session_state.anfis_y_pred_val
            y_pred_val_class = np.round(y_pred_val).astype(int)
            y_val_class = np.round(y_val).astype(int)

            val_accuracy = accuracy_score(y_val_class, y_pred_val_class)
            val_precision = precision_score(y_val_class, y_pred_val_class, average=average_method, zero_division=0)
            val_recall = recall_score(y_val_class, y_pred_val_class, average=average_method, zero_division=0)
            val_f1 = f1_score(y_val_class, y_pred_val_class, average=average_method, zero_division=0)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                delta = val_accuracy - train_accuracy
                st.metric("Accuracy", f"{val_accuracy:.4f}", delta=f"{delta:.4f}", delta_color="normal")

            with col2:
                delta = val_precision - train_precision
                st.metric("Precision", f"{val_precision:.4f}", delta=f"{delta:.4f}", delta_color="normal")

            with col3:
                delta = val_recall - train_recall
                st.metric("Recall", f"{val_recall:.4f}", delta=f"{delta:.4f}", delta_color="normal")

            with col4:
                delta = val_f1 - train_f1
                st.metric("F1-Score", f"{val_f1:.4f}", delta=f"{delta:.4f}", delta_color="normal")

        # Test metrics if available
        if st.session_state.get('anfis_X_test', None) is not None:
            st.markdown("---")
            st.markdown("**Test Set**")

            y_test = st.session_state.anfis_y_test.ravel()
            y_pred_test = st.session_state.anfis_y_pred_test
            y_pred_test_class = np.round(y_pred_test).astype(int)
            y_test_class = np.round(y_test).astype(int)

            test_accuracy = accuracy_score(y_test_class, y_pred_test_class)
            test_precision = precision_score(y_test_class, y_pred_test_class, average=average_method, zero_division=0)
            test_recall = recall_score(y_test_class, y_pred_test_class, average=average_method, zero_division=0)
            test_f1 = f1_score(y_test_class, y_pred_test_class, average=average_method, zero_division=0)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Accuracy", f"{test_accuracy:.4f}")

            with col2:
                st.metric("Precision", f"{test_precision:.4f}")

            with col3:
                st.metric("Recall", f"{test_recall:.4f}")

            with col4:
                st.metric("F1-Score", f"{test_f1:.4f}")

        st.markdown("---")

        # Confusion Matrix and Classification Report
        st.markdown("**Detailed Analysis**")

        # Dataset selector for confusion matrix
        available_sets = ['Training']
        if st.session_state.get('anfis_X_val', None) is not None:
            available_sets.append('Validation')
        if st.session_state.get('anfis_X_test', None) is not None:
            available_sets.append('Test')

        selected_set = st.radio(
            "Select dataset for detailed analysis",
            available_sets,
            horizontal=True,
            key='classification_dataset_selector'
        )

        st.markdown("")

        # Get appropriate data
        if selected_set == 'Training':
            y_true_selected = y_train_class
            y_pred_selected = y_pred_train_class
        elif selected_set == 'Validation':
            y_val = st.session_state.anfis_y_val.ravel()
            y_pred_val = st.session_state.anfis_y_pred_val
            y_true_selected = np.round(y_val).astype(int)
            y_pred_selected = np.round(y_pred_val).astype(int)
        else:  # Test
            y_test = st.session_state.anfis_y_test.ravel()
            y_pred_test = st.session_state.anfis_y_pred_test
            y_true_selected = np.round(y_test).astype(int)
            y_pred_selected = np.round(y_pred_test).astype(int)

        # Confusion Matrix
        render_confusion_matrix(y_true_selected, y_pred_selected, classes, selected_set)

        st.markdown("")

        # Classification Report
        render_classification_report(y_true_selected, y_pred_selected, classes)


def render_confusion_matrix(y_true, y_pred, classes, dataset_name='Training'):
    """Render confusion matrix visualization"""

    st.markdown("**Confusion Matrix**")

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # Create heatmap
    fig = go.Figure()

    # Normalize for better visualization
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig.add_trace(go.Heatmap(
        z=cm,
        x=[f'Pred {c}' for c in classes],
        y=[f'True {c}' for c in classes],
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 14},
        colorscale='Blues',
        showscale=True,
        hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
    ))

    fig.update_layout(
        title=f'Confusion Matrix ({dataset_name} Set)',
        xaxis_title='Predicted Class',
        yaxis_title='True Class',
        template='plotly_white',
        height=400,
        xaxis={'side': 'bottom'},
        yaxis={'autorange': 'reversed'}
    )

    st.plotly_chart(fig, width="stretch")

    # Show metrics per class
    st.caption("💡 **Diagonal values** show correct predictions. **Off-diagonal values** show misclassifications.")


def render_classification_report(y_true, y_pred, classes):
    """Render detailed classification report"""

    st.markdown("**Detailed Classification Report**")

    # Generate classification report
    report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)

    # Convert to dataframe
    report_data = []

    for class_label in classes:
        class_key = str(class_label)
        if class_key in report:
            report_data.append({
                'Class': f'Class {class_label}',
                'Precision': f"{report[class_key]['precision']:.4f}",
                'Recall': f"{report[class_key]['recall']:.4f}",
                'F1-Score': f"{report[class_key]['f1-score']:.4f}",
                'Support': int(report[class_key]['support'])
            })

    # Add overall metrics
    report_data.append({
        'Class': '---',
        'Precision': '---',
        'Recall': '---',
        'F1-Score': '---',
        'Support': '---'
    })

    if 'weighted avg' in report:
        report_data.append({
            'Class': 'Weighted Avg',
            'Precision': f"{report['weighted avg']['precision']:.4f}",
            'Recall': f"{report['weighted avg']['recall']:.4f}",
            'F1-Score': f"{report['weighted avg']['f1-score']:.4f}",
            'Support': int(report['weighted avg']['support'])
        })

    if 'macro avg' in report:
        report_data.append({
            'Class': 'Macro Avg',
            'Precision': f"{report['macro avg']['precision']:.4f}",
            'Recall': f"{report['macro avg']['recall']:.4f}",
            'F1-Score': f"{report['macro avg']['f1-score']:.4f}",
            'Support': int(report['macro avg']['support'])
        })

    df_report = pd.DataFrame(report_data)

    st.dataframe(df_report, width="stretch", hide_index=True)

    st.caption("💡 **Support** shows the number of samples for each class in the training set.")
