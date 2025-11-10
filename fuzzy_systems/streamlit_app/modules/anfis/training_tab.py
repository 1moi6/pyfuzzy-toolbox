"""
Training Tab for ANFIS Module
Handles model configuration and training
"""

import streamlit as st
import numpy as np
import time
from fuzzy_systems.learning.anfis import ANFIS


def render():
    """Render training tab"""

    # Check if dataset is loaded and split
    if st.session_state.get('anfis_X_train', None) is None:
        st.warning("⚠️ Please load and split a dataset first (Dataset tab)")
        return

    # Model configuration section
    render_model_configuration()

    st.markdown("")

    # Training configuration section
    render_training_configuration()

    st.markdown("")

    # Training results (if model trained)
    if st.session_state.get('anfis_trained', False) and st.session_state.anfis_model is not None:
        render_training_results()

    # Add space at the end
    st.markdown("")
    st.markdown("")


def render_model_configuration():
    """Render ANFIS model architecture configuration"""

    with st.expander("**Model Architecture** - ANFIS Configuration", expanded=True):

        # Get number of inputs from data
        n_inputs = st.session_state.anfis_X_train.shape[1]
        feature_names = st.session_state.get('anfis_feature_names',
                                            [f'X{i+1}' for i in range(n_inputs)])

        st.markdown(f"**Inputs:** {n_inputs} features")
        st.caption(f"Features: {', '.join(feature_names)}")

        st.markdown("")

        # Membership functions configuration
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("**Membership Functions**")

            mf_type = st.selectbox(
                "Type",
                ["gaussmf", "gbellmf", "sigmf"],
                format_func=lambda x: {
                    "gaussmf": "Gaussian",
                    "gbellmf": "Generalized Bell",
                    "sigmf": "Sigmoid"
                }[x],
                key='anfis_mf_type',
                help="gaussmf: μ(x) = exp(-(x-c)²/2σ²)\ngbellmf: μ(x) = 1/(1+|(x-c)/a|^(2b))\nsigmf: μ(x) = 1/(1+exp(-a(x-c)))"
            )

            # MFs per input
            use_same_mf = st.checkbox(
                "Same number of MFs for all inputs",
                value=True,
                key='anfis_same_mf'
            )

            if use_same_mf:
                n_mfs_value = st.number_input(
                    "Number of MFs per input",
                    min_value=2,
                    max_value=10,
                    value=3,
                    step=1,
                    key='anfis_n_mfs_single',
                    help="More MFs = more rules but higher complexity"
                )
                n_mfs = n_mfs_value
            else:
                st.markdown("**MFs per input:**")
                n_mfs_list = []
                cols = st.columns(min(n_inputs, 3))
                for i in range(n_inputs):
                    with cols[i % 3]:
                        n_mf = st.number_input(
                            feature_names[i],
                            2, 5, 3,
                            key=f'anfis_n_mfs_{i}'
                        )
                        n_mfs_list.append(n_mf)
                n_mfs = n_mfs_list

        with col2:
            st.markdown("**Model Info**")

            # Calculate total rules
            if isinstance(n_mfs, int):
                total_rules = n_mfs ** n_inputs
            else:
                total_rules = np.prod(n_mfs)

            st.metric("Total Rules", total_rules)

            # Calculate parameters
            if isinstance(n_mfs, int):
                if mf_type == 'gaussmf':
                    premise_params = n_inputs * n_mfs * 2  # center, sigma
                elif mf_type == 'gbellmf':
                    premise_params = n_inputs * n_mfs * 3  # a, b, c
                else:  # sigmf
                    premise_params = n_inputs * n_mfs * 2  # a, c
            else:
                if mf_type == 'gaussmf':
                    premise_params = sum(n_mfs) * 2
                elif mf_type == 'gbellmf':
                    premise_params = sum(n_mfs) * 3
                else:
                    premise_params = sum(n_mfs) * 2

            consequent_params = total_rules * (n_inputs + 1)
            total_params = premise_params + consequent_params

            st.metric("Parameters", total_params)
            st.caption(f"Premise: {premise_params}")
            st.caption(f"Consequent: {consequent_params}")

        # Store basic configuration (will be updated with advanced options in training config)
        st.session_state.anfis_config = {
            'n_inputs': n_inputs,
            'n_mfs': n_mfs,
            'mf_type': mf_type,
        }


def render_training_configuration():
    """Render training method and hyperparameters"""

    with st.expander("**Training** - Method & Hyperparameters", expanded=True):

        # Training method selector
        training_method = st.radio(
            "Training Method",
            ["Hybrid Learning", "Metaheuristic Optimization"],
            horizontal=True,
            key='anfis_training_method_radio',
            help="Hybrid: Fast, combines LSE + Gradient Descent\nMetaheuristic: Slower, global optimization"
        )

        st.session_state.anfis_training_method = training_method

        st.markdown("")

        if training_method == "Hybrid Learning":
            render_hybrid_config()
        else:
            render_metaheuristic_config()

        # Advanced options and train button side by side
        col_advanced, col_button = st.columns([2, 1],vertical_alignment='bottom')

        with col_advanced:
            with st.popover("Advanced Options", width='stretch'):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Regularization**")

                    lambda_l1 = st.number_input(
                        "L1 (Lasso)",
                        0.0, 1.0, 0.0, 0.001,
                        format="%.3f",
                        key='anfis_lambda_l1',
                        help="L1 regularization for sparse models"
                    )

                    lambda_l2 = st.number_input(
                        "L2 (Ridge)",
                        0.0, 1.0, 0.0, 0.001,
                        format="%.3f",
                        key='anfis_lambda_l2',
                        help="L2 regularization to prevent overfitting"
                    )

                with col2:
                    st.markdown("**Training Options**")

                    use_adaptive_lr = st.checkbox(
                        "Adaptive learning rate",
                        value=True,
                        key='anfis_use_adaptive_lr',
                        help="Automatically adjust learning rate during training"
                    )

                # Method-specific options
                st.markdown("---")

                if training_method == "Hybrid Learning":
                    col1, col2 = st.columns(2)

                    with col1:
                        train_premises = st.checkbox(
                            "Train premises (MF params)",
                            value=True,
                            key='anfis_train_premises',
                            help="If disabled, only consequent parameters are trained"
                        )

                    with col2:
                        verbose = st.checkbox(
                            "Verbose output",
                            value=True,
                            key='anfis_verbose',
                            help="Show training progress"
                        )

                    use_validation = st.session_state.anfis_X_val is not None
                    if use_validation:
                        restore_best = st.checkbox(
                            "Restore best weights",
                            value=True,
                            key='anfis_restore_best',
                            help="Restore best model from validation"
                        )
                    else:
                        restore_best = False

                    # Update training config
                    if 'anfis_train_config' in st.session_state:
                        st.session_state.anfis_train_config.update({
                            'train_premises': train_premises,
                            'verbose': verbose,
                            'restore_best_weights': restore_best
                        })

                else:  # Metaheuristic
                    col1, col2 = st.columns(2)

                    with col1:
                        verbose = st.checkbox(
                            "Verbose output",
                            value=True,
                            key='anfis_meta_verbose',
                            help="Show training progress"
                        )

                    with col2:
                        use_validation = st.session_state.anfis_X_val is not None
                        if use_validation:
                            restore_best = st.checkbox(
                                "Restore best weights",
                                value=True,
                                key='anfis_meta_restore_best',
                                help="Restore best model from validation"
                            )
                        else:
                            restore_best = False

                    # Update training config
                    if 'anfis_train_config' in st.session_state:
                        st.session_state.anfis_train_config.update({
                            'verbose': verbose,
                            'restore_best_weights': restore_best
                        })

            # Update config with advanced options
            if 'anfis_config' in st.session_state:
                st.session_state.anfis_config.update({
                    'lambda_l1': lambda_l1,
                    'lambda_l2': lambda_l2,
                    'use_adaptive_lr': use_adaptive_lr
                })

        with col_button:
            if st.button("🚀 Train Model", type="primary", width="stretch", key='train_button'):
                train_model()


def render_hybrid_config():
    """Render hybrid learning configuration"""

    # Main parameters in 4 columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        epochs = st.number_input(
            "Epochs",
            10, 1000, 100, 10,
            key='anfis_epochs',
            help="Number of training iterations"
        )

    with col2:
        learning_rate = st.number_input(
            "Learning Rate",
            0.001, 1.0, 0.01, 0.001,
            format="%.3f",
            key='anfis_learning_rate',
            help="Step size for gradient descent"
        )

    with col3:
        use_validation = st.session_state.anfis_X_val is not None

        if use_validation:
            early_stopping_patience = st.number_input(
                "Early stopping patience",
                5, 100, 20,
                key='anfis_early_stopping',
                help="Stop if no improvement for N epochs"
            )
        else:
            st.text_input(
                "Early stopping patience",
                value="N/A (no validation)",
                disabled=True,
                key='anfis_early_stopping_disabled'
            )
            early_stopping_patience = None

    with col4:
        batch_size = st.number_input(
            "Batch size",
            8, 512, 32, 8,
            key='anfis_batch_size_input',
            help="Mini-batch size for training"
        )

    # Store basic training config (will be updated with advanced options)
    train_premises = st.session_state.get('anfis_train_premises', True)
    verbose = st.session_state.get('anfis_verbose', True)
    restore_best = st.session_state.get('anfis_restore_best', True) if use_validation else False

    st.session_state.anfis_train_config = {
        'epochs': epochs,
        'learning_rate': learning_rate,
        'train_premises': train_premises,
        'early_stopping_patience': early_stopping_patience if use_validation else None,
        'restore_best_weights': restore_best,
        'verbose': verbose
    }

    # Store batch size in config
    if 'anfis_config' not in st.session_state:
        st.session_state.anfis_config = {}
    st.session_state.anfis_config['batch_size'] = batch_size


def render_metaheuristic_config():
    """Render metaheuristic optimization configuration"""

    # Main parameters
    col1, col2, col3 = st.columns(3)

    with col1:
        optimizer = st.selectbox(
            "Optimizer",
            ["pso", "de", "ga"],
            format_func=lambda x: {
                "pso": "PSO (Particle Swarm)",
                "de": "DE (Differential Evolution)",
                "ga": "GA (Genetic Algorithm)"
            }[x],
            key='anfis_optimizer',
            help="Metaheuristic algorithm for global optimization"
        )

    with col2:
        n_particles = st.number_input(
            "Population size",
            10, 100, 30, 5,
            key='anfis_n_particles',
            help="Number of particles/individuals"
        )

    with col3:
        n_iterations = st.number_input(
            "Iterations",
            10, 500, 100, 10,
            key='anfis_n_iterations',
            help="Number of optimization iterations"
        )

    # Additional row for early stopping
    use_validation = st.session_state.anfis_X_val is not None

    if use_validation:
        early_stopping_patience = st.number_input(
            "Early stopping patience",
            5, 100, 20,
            key='anfis_meta_early_stopping',
            help="Stop if no improvement for N iterations"
        )
    else:
        early_stopping_patience = None

    # Store basic training config (will be updated with advanced options)
    verbose = st.session_state.get('anfis_meta_verbose', True)
    restore_best = st.session_state.get('anfis_meta_restore_best', True) if use_validation else False

    st.session_state.anfis_train_config = {
        'optimizer': optimizer,
        'n_particles': n_particles,
        'n_iterations': n_iterations,
        'early_stopping_patience': early_stopping_patience if use_validation else None,
        'restore_best_weights': restore_best,
        'verbose': verbose
    }


def train_model():
    """Train ANFIS model"""

    try:
        # Get configuration
        config = st.session_state.anfis_config
        train_config = st.session_state.anfis_train_config
        training_method = st.session_state.anfis_training_method

        # Get data
        X_train = st.session_state.anfis_X_train
        y_train = st.session_state.anfis_y_train.ravel()
        X_val = st.session_state.anfis_X_val
        y_val = st.session_state.anfis_y_val.ravel() if st.session_state.anfis_y_val is not None else None

        # Create progress containers
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Calculate input ranges from training data
        input_ranges = [(X_train[:, i].min(), X_train[:, i].max())
                       for i in range(config['n_inputs'])]

        status_text.info(f"Creating ANFIS model with {config['n_mfs']} MFs per input...")

        # Create ANFIS model
        model = ANFIS(
            n_inputs=config['n_inputs'],
            n_mfs=config['n_mfs'],
            mf_type=config['mf_type'],
            learning_rate=train_config.get('learning_rate', 0.01),
            input_ranges=input_ranges,
            lambda_l1=config['lambda_l1'],
            lambda_l2=config['lambda_l2'],
            batch_size=config['batch_size'],
            use_adaptive_lr=config['use_adaptive_lr'],
            classification=(st.session_state.get('anfis_problem_type', 'Regression') == 'Classification')
        )

        progress_bar.progress(20)
        status_text.info(f"Training with {training_method}...")

        # Train model
        start_time = time.time()

        if training_method == "Hybrid Learning":
            model.fit(
                X_train, y_train,
                epochs=train_config['epochs'],
                verbose=train_config['verbose'],
                train_premises=train_config['train_premises'],
                X_val=X_val,
                y_val=y_val,
                early_stopping_patience=train_config['early_stopping_patience'],
                restore_best_weights=train_config['restore_best_weights']
            )
        else:  # Metaheuristic
            model.fit_metaheuristic(
                X_train, y_train,
                optimizer=train_config['optimizer'],
                n_particles=train_config['n_particles'],
                n_iterations=train_config['n_iterations'],
                verbose=train_config['verbose'],
                X_val=X_val,
                y_val=y_val,
                early_stopping_patience=train_config['early_stopping_patience'],
                restore_best_weights=train_config['restore_best_weights']
            )

        training_time = time.time() - start_time

        progress_bar.progress(100)

        # Store model and results
        st.session_state.anfis_model = model
        st.session_state.anfis_trained = True
        st.session_state.anfis_training_time = training_time
        st.session_state.anfis_training_method_used = training_method

        # Calculate metrics
        y_pred_train = model.predict(X_train)
        train_rmse = np.sqrt(np.mean((y_train - y_pred_train) ** 2))

        if X_val is not None and y_val is not None:
            y_pred_val = model.predict(X_val)
            val_rmse = np.sqrt(np.mean((y_val - y_pred_val) ** 2))
        else:
            val_rmse = None

        st.session_state.anfis_train_rmse = train_rmse
        st.session_state.anfis_val_rmse = val_rmse

        # Success message
        status_text.empty()
        progress_bar.empty()

        st.success(f"✅ Training complete! ({training_time:.2f}s)")
        st.balloons()

        # Show quick results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training RMSE", f"{train_rmse:.4f}")
        with col2:
            if val_rmse is not None:
                st.metric("Validation RMSE", f"{val_rmse:.4f}")
        with col3:
            st.metric("Rules", model.n_rules)

        st.rerun()

    except Exception as e:
        st.error(f"❌ Training failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def render_training_results():
    """Display training results summary"""

    with st.expander("**Training Results** - Summary", expanded=False):

        model = st.session_state.anfis_model
        training_time = st.session_state.get('anfis_training_time', 0)
        method = st.session_state.get('anfis_training_method_used', 'Unknown')

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Method", method.split()[0])

        with col2:
            st.metric("Rules", model.n_rules)

        with col3:
            st.metric("Train RMSE", f"{st.session_state.anfis_train_rmse:.4f}")

        with col4:
            if st.session_state.get('anfis_val_rmse', None) is not None:
                st.metric("Val RMSE", f"{st.session_state.anfis_val_rmse:.4f}")
            else:
                st.metric("Val RMSE", "N/A")

        st.caption(f"Training time: {training_time:.2f}s")

        # Model details
        if hasattr(model, 'convergence_metrics') and model.convergence_metrics:
            st.markdown("")
            st.markdown("**Convergence Info:**")

            metrics = model.convergence_metrics

            if 'train_errors' in metrics:
                st.caption(f"Final train error: {metrics['train_errors'][-1]:.4f}")

            if 'val_errors' in metrics and len(metrics['val_errors']) > 0:
                st.caption(f"Final val error: {metrics['val_errors'][-1]:.4f}")

            if 'best_epoch' in metrics:
                st.caption(f"Best epoch: {metrics['best_epoch']}")

            if 'early_stopped' in metrics and metrics['early_stopped']:
                st.info("Early stopping triggered")
