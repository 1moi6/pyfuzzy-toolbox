"""
Wang-Mendel Module for pyfuzzy-toolbox Streamlit Interface
==========================================================
Main module that coordinates Wang-Mendel interface tabs

Author: Moiseis Cecconello
"""

import streamlit as st
from .wang_mendel import (
    dataset_tab,
    training_tab,
    metrics_tab,
    prediction_tab,
    analysis_tab,
    overview_tab
)


def run():
    """Main function to render Wang-Mendel interface"""

    # Initialize session state
    init_session_state()

    # Render sidebar
    render_sidebar()

    # Render main content
    render_main_content()


def init_session_state():
    """Initialize session state variables for Wang-Mendel module"""

    defaults = {
        'wm_model': None,
        'wm_system': None,
        'wm_trained': False,
        'wm_dataset': None,
        'wm_dataset_name': None,
        'wm_X_train': None,
        'wm_y_train': None,
        'wm_X_val': None,
        'wm_y_val': None,
        'wm_X_test': None,
        'wm_y_test': None,
        'wm_scaler_X': None,
        'wm_scaler_y': None,
        'wm_feature_names': None,
        'wm_target_name': None,
        'wm_task': 'regression',
        'wm_n_partitions': 5,
        'wm_mf_type': 'triangular',
        'wm_config_mode': 'automatic',
        'wm_training_stats': None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar():
    """Render sidebar with Wang-Mendel quick actions"""

    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 0.25rem 0 0.125rem 0; margin-top: 0.5rem;">
            <h2 style="margin: 0.25rem 0 0.125rem 0; color: #667eea;">Wang-Mendel</h2>
            <p style="color: #6b7280; font-size: 0.9rem; margin: 0;">
                Automatic Fuzzy Rule Generation
            </p>
        </div>
        <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 0.25rem 0 0.5rem 0;">
        """, unsafe_allow_html=True)

        # Dataset source selector (if in Dataset tab)
        dataset_tab.render_sidebar_controls()

        st.markdown("<hr style='border: none; border-top: 1px solid #e5e7eb; margin: 0.75rem 0;'>",
                   unsafe_allow_html=True)

        # Quick Status
        st.markdown("#### System Status")

        col1, col2 = st.columns(2)

        # Dataset status: check if dataset loaded AND preprocessed (split done)
        with col1:
            has_dataset = st.session_state.get('wm_dataset', None) is not None
            has_split = st.session_state.get('wm_X_train', None) is not None

            if has_dataset and has_split:
                dataset_status = "✅"
            elif has_dataset:
                dataset_status = "⚙️"  # Loaded but not processed
            else:
                dataset_status = "⏳"

            st.metric("Dataset", dataset_status)

        # Model status: check if trained
        with col2:
            is_trained = st.session_state.get('wm_trained', False)
            model_status = "✅" if is_trained else "⏳"
            st.metric("Model", model_status)

        # Dataset info
        if st.session_state.get('wm_dataset', None) is not None:
            dataset_name = st.session_state.get('wm_dataset_name', 'Custom')
            n_samples = len(st.session_state.wm_dataset)

            if st.session_state.get('wm_X_train', None) is not None:
                st.info(f"📦 **{dataset_name}**\n\n{n_samples} samples (split ready)")
            else:
                st.warning(f"📦 **{dataset_name}**\n\n{n_samples} samples (not split)")

        # Model info
        if st.session_state.get('wm_trained', False) and st.session_state.get('wm_system', None) is not None:
            task = st.session_state.get('wm_task', 'regression')
            icon = "🎯" if task == 'classification' else "📈"
            task_label = task.capitalize()

            stats = st.session_state.get('wm_training_stats', {})
            n_rules = stats.get('final_rules', 0)

            st.success(f"{icon} **{task_label} Model**\n\n{n_rules} rules generated")

            # Export button (only when trained)
            st.markdown("")
            if st.button("📤 Export to Mamdani", use_container_width=True,
                        help="Export this FIS to the Inference (Mamdani) page"):
                from modules.wang_mendel import training_tab
                training_tab.export_fis_to_mamdani()
                st.success("✓ Exported to Inference page!")
                st.rerun()

        st.markdown("<hr style='border: none; border-top: 1px solid #e5e7eb; margin: 0.75rem 0;'>",
                   unsafe_allow_html=True)

        # Reset button
        st.markdown("#### Actions")
        if st.button("🔄 Reset All", use_container_width=True,
                    help="Clear all data and start fresh"):
            # Reset all Wang-Mendel session state
            keys_to_reset = [
                'wm_model', 'wm_system', 'wm_trained', 'wm_dataset', 'wm_dataset_name',
                'wm_X_train', 'wm_y_train', 'wm_X_val', 'wm_y_val', 'wm_X_test', 'wm_y_test',
                'wm_scaler_X', 'wm_scaler_y', 'wm_feature_names', 'wm_target_name',
                'wm_training_stats', 'wm_y_pred_train', 'wm_y_pred_val', 'wm_y_pred_test',
                'wm_manual_prediction_result', 'wm_manual_prediction_class',
                'wm_manual_prediction_prob', 'wm_manual_prediction_inputs',
                'wm_batch_prediction_results', 'wm_batch_is_classification',
                'wm_dataset_source', 'wm_dataset_source_prev',
                'wm_classic_dataset_choice', 'wm_classic_dataset_prev'
            ]
            for key in keys_to_reset:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("✓ Reset complete!")
            st.rerun()

        st.markdown("<hr style='border: none; border-top: 1px solid #e5e7eb; margin: 0.75rem 0;'>",
                   unsafe_allow_html=True)

        # Quick info
        st.markdown("#### Quick Info")

        with st.expander("What is Wang-Mendel?", expanded=False):
            st.markdown("""
            **Wang-Mendel** is a fast algorithm for automatically generating fuzzy rules from data.

            **Key Features:**
            - ⚡ One-shot learning (no iterations)
            - 📝 Interpretable fuzzy rules
            - 🎯 Works for regression & classification
            - 🚀 Fast training

            **5 Steps:**
            1. Partition domains
            2. Generate candidate rules
            3. Assign rule degrees
            4. Resolve conflicts
            5. Create final system

            See **Overview** tab for details!
            """)

        with st.expander("Workflow", expanded=False):
            st.markdown("""
            **1. Dataset**
            - Load or generate data
            - Split into train/test
            - Optional normalization

            **2. Training**
            - Configure fuzzy system (FIS)
            - Set membership functions
            - Run Wang-Mendel algorithm
            - View rule generation stats

            **3. Metrics**
            - Detailed performance analysis
            - Classification: confusion matrix
            - Regression: R², RMSE, MAE

            **4. Prediction**
            - Manual input prediction
            - Batch CSV prediction
            - Export results

            **5. Analysis**
            - Visualize rule base
            - Membership functions
            - Decision surface (1D/2D)

            **6. Overview**
            - Theory and algorithm
            - Examples and references
            """)


def render_main_content():
    """Render main content tabs"""

    st.markdown("""
    <div style="text-align: center; padding: 0.5rem 0;">
        <h3 style="color: #6b7280; font-weight: 500; margin: 0; font-size: 1.1rem;">
            Wang-Mendel algorithm for automatically generating fuzzy rule bases
        </h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='border-bottom: 1px solid #e5e7eb; margin: 0.5rem 0 1.5rem 0;'></div>",
               unsafe_allow_html=True)

    # Tab layout
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Dataset",
        "Training",
        "Metrics",
        "Prediction",
        "Analysis",
        "Overview"
    ])

    with tab1:
        dataset_tab.render()

    with tab2:
        training_tab.render()

    with tab3:
        metrics_tab.render()

    with tab4:
        prediction_tab.render()

    with tab5:
        analysis_tab.render()

    with tab6:
        overview_tab.render()
