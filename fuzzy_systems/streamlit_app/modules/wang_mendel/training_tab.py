"""
Training Tab for Wang-Mendel Module
Handles FIS configuration and training execution
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fuzzy_systems.inference import MamdaniSystem
from fuzzy_systems.learning import WangMendelLearning


def close_summary_dialog():
    """Callback to reset action selection"""
    st.session_state['summary_options_control'] = None


def export_fis_to_mamdani():
    """Export Wang-Mendel FIS to Mamdani/Inference page"""

    system = st.session_state.get('wm_system')
    if system is None:
        st.error("No FIS to export!")
        return

    # Initialize fis_list in session_state if it doesn't exist
    if 'fis_list' not in st.session_state:
        st.session_state.fis_list = []

    # Generate unique name
    fis_name = "Wang-Mendel FIS"
    existing_names = [fis['name'] for fis in st.session_state.fis_list]
    if fis_name in existing_names:
        counter = 2
        while f"{fis_name} ({counter})" in existing_names:
            counter += 1
        fis_name = f"{fis_name} ({counter})"

    # Extract metadata for UI display (but system object will be used for inference)
    # Input variables metadata
    input_variables = []
    for var_name, var in system.input_variables.items():
        terms = []
        for term_name, mf in var.terms.items():
            # Simple metadata - just name and type for display
            # Detect type by checking attributes
            if hasattr(mf, 'a') and hasattr(mf, 'b') and hasattr(mf, 'c') and hasattr(mf, 'd'):
                mf_type = 'trapezoidal'
                params = [mf.a, mf.b, mf.c, mf.d]
            elif hasattr(mf, 'a') and hasattr(mf, 'b') and hasattr(mf, 'c'):
                mf_type = 'triangular'
                params = [mf.a, mf.b, mf.c]
            elif hasattr(mf, 'mean') and hasattr(mf, 'std'):
                mf_type = 'gaussian'
                params = [mf.mean, mf.std]
            else:
                mf_type = 'triangular'
                params = []

            terms.append({
                'name': term_name,
                'mf_type': mf_type,
                'params': params
            })

        input_variables.append({
            'name': var_name,
            'min': var.universe[0],
            'max': var.universe[1],
            'terms': terms
        })

    # Output variables metadata
    output_variables = []
    for var_name, var in system.output_variables.items():
        terms = []
        for term_name, mf in var.terms.items():
            # Detect type by checking attributes
            if hasattr(mf, 'a') and hasattr(mf, 'b') and hasattr(mf, 'c') and hasattr(mf, 'd'):
                mf_type = 'trapezoidal'
                params = [mf.a, mf.b, mf.c, mf.d]
            elif hasattr(mf, 'a') and hasattr(mf, 'b') and hasattr(mf, 'c'):
                mf_type = 'triangular'
                params = [mf.a, mf.b, mf.c]
            elif hasattr(mf, 'mean') and hasattr(mf, 'std'):
                mf_type = 'gaussian'
                params = [mf.mean, mf.std]
            else:
                mf_type = 'triangular'
                params = []

            terms.append({
                'name': term_name,
                'mf_type': mf_type,
                'params': params
            })

        output_variables.append({
            'name': var_name,
            'min': var.universe[0],
            'max': var.universe[1],
            'terms': terms
        })

    # Rules metadata
    fuzzy_rules = []
    for rule in system.rule_base.rules:
        antecedents = {}
        for var_name, term_value in rule.antecedents.items():
            # Extract term name
            if isinstance(term_value, tuple) and len(term_value) == 2:
                _, term_name = term_value
            else:
                term_name = str(term_value)
            antecedents[var_name] = term_name

        consequents = {}
        for var_name, term_value in rule.consequent.items():
            # Extract term name
            if isinstance(term_value, tuple) and len(term_value) == 2:
                _, term_name = term_value
            else:
                term_name = str(term_value)
            consequents[var_name] = term_name

        fuzzy_rules.append({
            'antecedents': antecedents,
            'consequents': consequents,
            'weight': 1.0
        })

    # Create FIS entry with both system object AND metadata for UI
    new_fis = {
        'name': fis_name,
        'type': 'Mamdani',
        'system': system,  # MamdaniSystem object for inference (no conversion needed!)
        'input_variables': input_variables,  # Metadata for UI display
        'output_variables': output_variables,  # Metadata for UI display
        'fuzzy_rules': fuzzy_rules  # Metadata for UI display
    }

    # Add to fis_list and set as active
    st.session_state.fis_list.append(new_fis)
    st.session_state.active_fis_idx = len(st.session_state.fis_list) - 1



def render():
    """Render combined configuration and training tab"""

    # Check if dataset is loaded
    if st.session_state.get('wm_X_train', None) is None:
        st.warning("⚠️ Please load and split a dataset first (see **Dataset** tab)")
        return

    # Step 1: FIS Configuration
    render_fis_configuration()

    st.markdown("")

    # Step 2: Training Options (only if system is configured)
    if st.session_state.get('wm_system', None) is not None:
        render_training_options()


    # Show training results
    if st.session_state.get('wm_trained', False):
        render_training_results()


def render_fis_configuration():
    """Render FIS configuration section"""

    with st.expander("**FIS Configuration** - Fuzzy Inference System Setup", expanded=True):

        # Get dataset info
        n_inputs = st.session_state.wm_X_train.shape[1]
        feature_names = st.session_state.get('wm_feature_names', [f'X{i+1}' for i in range(n_inputs)])
        target_name = st.session_state.get('wm_target_name', 'Y')
        task = st.session_state.get('wm_task', 'regression')

        # Configuration mode selector
        config_mode = st.radio(
            "Configuration mode",
            ["Automatic", "Manual"],
            key='wm_config_mode_selector',
            horizontal=True,
            help="**Automatic**: Quick setup with uniform partitions\n\n**Manual**: Full control (Coming soon)"
        )

        st.markdown("")

        if config_mode == "Automatic":
            render_automatic_config(n_inputs, feature_names, target_name, task)
        else:
            render_manual_config()
        
        if st.session_state.get('wm_system', None) is not None:
            selection = st.segmented_control("Summary Options", ['System Summary','Membership Functions','Variable Details'],default=None,
                                                selection_mode="single",
                                                width='stretch',
                                                label_visibility = 'hidden',
                                                key = 'summary_options_control')
            if selection is not None:
                render_system_summary(selection)

def render_automatic_config(n_inputs, feature_names, target_name, task):
    """Render automatic configuration interface"""

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("**Membership Function Settings**")

        # MF type
        mf_type = st.selectbox(
            "Membership function type",
            ["triangular", "trapezoidal", "gaussian"],
            key='wm_auto_mf_type',
            help="**Triangular**: Simple, interpretable\n**Trapezoidal**: Flat plateaus\n**Gaussian**: Smooth curves"
        )

        # Number of partitions
        n_partitions = st.slider(
            "Partitions per variable",
            min_value=3,
            max_value=11,
            value=5,
            step=2,
            key='wm_auto_n_partitions',
            help="More partitions → finer granularity but more rules"
        )

        # Show rule complexity
        max_rules = n_partitions ** n_inputs
        st.caption(f"Maximum possible rules: **{max_rules}** (actual will be less after conflict resolution)")

    with col2:
        st.markdown("**Variable Ranges**")
        st.caption("Auto-detected from training data with 10% margin")

        # Get data ranges
        X_train = st.session_state.wm_X_train
        y_train = st.session_state.wm_y_train

        margin = 0.1

        # Input ranges
        st.markdown("**Inputs:**")
        for i, name in enumerate(feature_names):
            x_min, x_max = X_train[:, i].min(), X_train[:, i].max()
            x_range = x_max - x_min
            x_min_ext = x_min - margin * x_range
            x_max_ext = x_max + margin * x_range
            st.caption(f"• **{name}**: `[{x_min_ext:.2f}, {x_max_ext:.2f}]`")

        # Output range
        st.markdown("**Output:**")
        if task == 'regression':
            y_min, y_max = y_train.min(), y_train.max()
            y_range = y_max - y_min
            y_min_ext = y_min - margin * y_range
            y_max_ext = y_max + margin * y_range
            st.caption(f"• **{target_name}**: `[{y_min_ext:.2f}, {y_max_ext:.2f}]`")
        else:
            n_classes = y_train.shape[1]
            st.caption(f"• **{target_name}**: {n_classes} classes")

        st.markdown("")

    # Create button
        with col1:
            create_btn = st.button(
                "Create Fuzzy System",
                type="primary",
                use_container_width=True,
                key='wm_create_auto_system'
            )

            if create_btn:
                create_automatic_system(n_inputs, feature_names, target_name, task, mf_type, n_partitions)


def create_automatic_system(n_inputs, feature_names, target_name, task, mf_type, n_partitions):
    """Create Mamdani system with automatic membership functions"""

    try:
        with st.spinner("Creating fuzzy system..."):
            # Create system
            system = MamdaniSystem(name='WangMendelSystem')

            # Get data ranges
            X_train = st.session_state.wm_X_train
            y_train = st.session_state.wm_y_train

            margin = 0.1

            # Add input variables with automatic membership functions
            for i, name in enumerate(feature_names):
                x_min, x_max = X_train[:, i].min(), X_train[:, i].max()
                x_range = x_max - x_min
                x_min_ext = x_min - margin * x_range
                x_max_ext = x_max + margin * x_range

                system.add_input(name, (x_min_ext, x_max_ext))
                system.add_auto_mfs(name, n_mfs=n_partitions, mf_type=mf_type, label_prefix='In')

            # Add output variable
            if task == 'regression':
                y_min, y_max = y_train.min(), y_train.max()
                y_range = y_max - y_min
                y_min_ext = y_min - margin * y_range
                y_max_ext = y_max + margin * y_range

                system.add_output(target_name, (y_min_ext, y_max_ext))
                system.add_auto_mfs(target_name, n_mfs=n_partitions, mf_type=mf_type, label_prefix='Out')

            else:  # classification
                # For classification, create one output variable per class (binary: no/yes)
                n_classes = y_train.shape[1]

                for i in range(n_classes):
                    output_name = f"{target_name}_class_{i}"
                    system.add_output(output_name, (0, 1))
                    system.add_term(output_name, 'no', 'triangular', (0, 0, 1))
                    system.add_term(output_name, 'yes', 'triangular', (0, 1, 1))

            # Store system in session state
            st.session_state.wm_system = system
            st.session_state.wm_n_partitions = n_partitions
            st.session_state.wm_mf_type = mf_type
            st.session_state.wm_config_mode = 'automatic'

            # Reset training state
            if 'wm_trained' in st.session_state:
                del st.session_state.wm_trained
            if 'wm_model' in st.session_state:
                del st.session_state.wm_model

            st.success("✅ Fuzzy system created successfully!")
            st.rerun()

    except Exception as e:
        st.error(f"❌ Error creating system: {str(e)}")
        import traceback
        with st.expander("Debug Info"):
            st.code(traceback.format_exc())


def render_manual_config():
    """Render manual configuration (placeholder)"""

    st.warning("Manual configuration is under development")

    st.markdown("""
    Manual mode will allow you to:

    - Define custom membership function shapes
    - Set different numbers of partitions per variable
    - Choose custom linguistic labels
    - Fine-tune membership function parameters
    - Mix different MF types

    **For now, please use Automatic mode** which works great for most use cases.
    """)


def render_training_options():
    """Render training configuration options"""

    with st.expander("**Training Options** - Algorithm Configuration", expanded=False):
        col1, col2 = st.columns([2, 1])

        with col1:
            task_display = st.radio(
                "Task type",
                ["Auto-detect", "Regression", "Classification"],
                index=0,
                key='wm_task_option',
                horizontal=True,
                help="**Auto-detect**: Algorithm detects from data\n**Regression**: Continuous output\n**Classification**: Discrete classes"
            )

            # Map to API parameter
            if task_display == "Auto-detect":
                task_param = 'auto'
            elif task_display == "Regression":
                task_param = 'regression'
            else:
                task_param = 'classification'

            st.session_state.wm_task_param = task_param

        with col2:
            verbose = st.checkbox(
                "Verbose output",
                value=True,
                key='wm_verbose',
                help="Show detailed training progress"
            )

            scale_classification = st.checkbox(
                "Scale classification output",
                value=True,
                key='wm_scale_classification',
                help="Scale output for classification (recommended)"
            )

        if st.button("Run Wang-Mendel Algorithm", type="primary", use_container_width=True, key='wm_train_button'):
            task_param = st.session_state.get('wm_task_param', 'auto')
            verbose = st.session_state.get('wm_verbose', True)
            scale_classification = st.session_state.get('wm_scale_classification', True)
            run_wang_mendel(task_param, verbose, scale_classification)


def run_wang_mendel(task, verbose, scale_classification):
    """Execute Wang-Mendel algorithm"""

    try:
        with st.spinner("Running Wang-Mendel algorithm..."):
            # Get data
            X_train = st.session_state.wm_X_train
            y_train = st.session_state.wm_y_train
            system = st.session_state.wm_system

            # Create WangMendelLearning instance
            wm = WangMendelLearning(
                system=system,
                X=X_train,
                y=y_train,
                task=task,
                scale_classification=scale_classification,
                verbose_init=False
            )

            # Capture verbose output
            if verbose:
                import io
                import sys
                old_stdout = sys.stdout
                sys.stdout = captured_output = io.StringIO()

            # Fit
            trained_system = wm.fit(verbose=verbose)

            # Restore stdout
            if verbose:
                sys.stdout = old_stdout
                verbose_output = captured_output.getvalue()
                st.session_state.wm_verbose_output = verbose_output

            # Get training statistics
            stats = wm.get_training_stats()

            # Store in session state
            st.session_state.wm_model = wm
            st.session_state.wm_system = trained_system
            st.session_state.wm_training_stats = stats
            st.session_state.wm_trained = True

            # Make predictions on training set (and test if available)
            y_pred_train = wm.predict(X_train)

            # Handle regression reshape
            if st.session_state.get('wm_task', 'regression') == 'regression':
                y_pred_train = y_pred_train.reshape(-1, 1)

            st.session_state.wm_y_pred_train = y_pred_train

            # Test predictions
            if st.session_state.get('wm_X_test', None) is not None:
                y_pred_test = wm.predict(st.session_state.wm_X_test)
                if st.session_state.get('wm_task', 'regression') == 'regression':
                    y_pred_test = y_pred_test.reshape(-1, 1)
                st.session_state.wm_y_pred_test = y_pred_test

            st.success("✅ Training complete")
            st.rerun()

    except Exception as e:
        st.error(f"❌ Error during training: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def render_training_results():
    """Display training results and statistics"""

    with st.expander("**Training Results** - Rule Generation Statistics", expanded=True):
        stats = st.session_state.get('wm_training_stats', {})

        # Statistics cards
        col1, col2, col3 = st.columns(3)

        with col1:
            candidate_rules = stats.get('candidate_rules', 0)
            st.metric("Candidate Rules", candidate_rules, help="Total rules generated from training data")

        with col2:
            final_rules = stats.get('final_rules', 0)
            st.metric("Final Rules", final_rules, help="Rules after conflict resolution")

        with col3:
            conflicts = stats.get('conflicts_resolved', 0)
            reduction_pct = ((candidate_rules - final_rules) / candidate_rules * 100) if candidate_rules > 0 else 0
            st.metric("Conflicts Resolved", conflicts, f"-{reduction_pct:.1f}%", help="Rules removed due to conflicts")

        st.markdown("")

        # Rule efficiency
        n_inputs = st.session_state.wm_X_train.shape[1]
        n_partitions = st.session_state.get('wm_n_partitions', 5)
        max_possible_rules = n_partitions ** n_inputs

        efficiency = (final_rules / max_possible_rules * 100) if max_possible_rules > 0 else 0

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("**Rule Base Efficiency**")
            st.progress(efficiency / 100, text=f"{final_rules} / {max_possible_rules} rules ({efficiency:.1f}% of maximum)")
            st.caption("Lower percentage = more efficient (fewer rules needed)")

        with col2:
            st.markdown("**Compression Ratio**")
            compression = (candidate_rules / final_rules) if final_rules > 0 else 1
            st.metric("", f"{compression:.2f}x", help="How much the rule base was compressed")

    # Verbose output
    if st.session_state.get('wm_verbose_output', None):
        st.markdown("")
        with st.expander("**Training Log** - Detailed Output", expanded=False):
            st.code(st.session_state.wm_verbose_output, language='text')
    
    st.markdown("")
    st.markdown("")

@st.dialog('Summary of Created System',on_dismiss=close_summary_dialog,width = 'large')
def render_system_summary(type):
    """Render summary of created system"""

    system = st.session_state.get('wm_system')
    if type=='System Summary':

        # Main metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            n_inputs = len(system.input_variables)
            st.metric("Input Variables", n_inputs, help="Number of input features")

        with col2:
            n_outputs = len(system.output_variables)
            st.metric("Output Variables", n_outputs, help="Number of outputs")

        with col3:
            total_input_mfs = sum(len(var.terms) for var in system.input_variables.values())
            st.metric("Input MFs", total_input_mfs, help="Total input membership functions")

        with col4:
            total_output_mfs = sum(len(var.terms) for var in system.output_variables.values())
            st.metric("Output MFs", total_output_mfs, help="Total output membership functions")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Configuration:**")
            mode = st.session_state.get('wm_config_mode', 'unknown')
            n_partitions = st.session_state.get('wm_n_partitions', 'N/A')
            mf_type = st.session_state.get('wm_mf_type', 'N/A')

            st.markdown(f"- Mode: `{mode}`")
            st.markdown(f"- Partitions: `{n_partitions}`")
            st.markdown(f"- MF Type: `{mf_type}`")

        with col2:
            st.markdown("**Variables:**")

            # Input names
            input_names = list(system.input_variables.keys())
            st.markdown(f"- Inputs: `{', '.join(input_names)}`")

            # Output names
            output_names = list(system.output_variables.keys())
            if len(output_names) <= 3:
                st.markdown(f"- Outputs: `{', '.join(output_names)}`")
            else:
                st.markdown(f"- Outputs: `{len(output_names)} variables` (classification)")

    # Visualize membership functions
    if type=='Membership Functions':
        st.markdown("**Membership Functions** - Linguistic Terms")

        try:
            # Get all variables
            all_vars = list(system.input_variables.items()) + list(system.output_variables.items())
            n_vars = len(all_vars)

            # Display in columns (max 2 per row)
            if n_vars <= 2:
                cols = st.columns(n_vars)
            else:
                cols = st.columns(2)

            # Plot each variable in separate figures
            for idx, (var_name, var) in enumerate(all_vars):
                col_idx = idx % len(cols)

                with cols[col_idx]:
                    st.markdown(f"**{var_name}**")

                    # Create individual figure
                    fig = go.Figure()

                    # Create points for plotting
                    x_range = var.universe
                    x_points = np.linspace(x_range[0], x_range[1], 200)

                    # Plot each membership function
                    for term_name, mf in var.terms.items():
                        y_points = mf.membership(x_points)

                        fig.add_trace(
                            go.Scatter(
                                x=x_points,
                                y=y_points,
                                mode='lines',
                                name=term_name,
                                hovertemplate=f'<b>{term_name}</b><br>x=%{{x:.3f}}<br>μ=%{{y:.3f}}<extra></extra>'
                            )
                        )

                    # Update layout
                    fig.update_layout(
                        xaxis_title=var_name,
                        yaxis_title='Membership (μ)',
                        yaxis_range=[0, 1.05],
                        height=300,
                        template='plotly_white',
                        showlegend=True,
                        legend=dict(
                            orientation="v",
                            yanchor="top",
                            y=1,
                            xanchor="right",
                            x=1.15
                        ),
                        margin=dict(l=40, r=80, t=20, b=40)
                    )

                    st.plotly_chart(fig, use_container_width=True, key=f'mf_{var_name}')

                    # Add caption with details
                    st.caption(f"Universe: [{x_range[0]:.2f}, {x_range[1]:.2f}] | "
                              f"Terms: {len(var.terms)}")

        except Exception as e:
            st.error(f"Error plotting membership functions: {str(e)}")

    # Variable details
    if type=='Variable Details':
        st.markdown("**Input Variables**")

        for var_name, var in system.input_variables.items():
            st.markdown(f"**{var_name}**")
            st.caption(f"Universe: [{var.universe[0]:.2f}, {var.universe[1]:.2f}]")
            st.caption(f"Terms: {', '.join(var.terms.keys())}")
            st.markdown("")

        st.markdown("**Output Variables**")

        for var_name, var in system.output_variables.items():
            st.markdown(f"**{var_name}**")
            st.caption(f"Universe: [{var.universe[0]:.2f}, {var.universe[1]:.2f}]")
            st.caption(f"Terms: {', '.join(var.terms.keys())}")
            st.markdown("")
