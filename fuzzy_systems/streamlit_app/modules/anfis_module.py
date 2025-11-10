"""
ANFIS Module for pyfuzzy-toolbox Streamlit Interface
====================================================
Main module that coordinates ANFIS interface tabs

Author: Moiseis Cecconello
"""

import streamlit as st
from .anfis import dataset_tab
from .anfis import training_tab
from .anfis import metrics_tab
from .anfis import prediction_tab
from .anfis import analysis_tab
from .anfis import overview_tab


def run():
    """Main function to render ANFIS interface"""

    # Initialize session state
    init_session_state()

    # Render sidebar
    render_sidebar()

    # Render main content
    render_main_content()


def init_session_state():
    """Initialize session state variables for ANFIS module"""

    defaults = {
        'anfis_model': None,
        'anfis_trained': False,
        'anfis_dataset': None,
        'anfis_dataset_name': None,
        'anfis_X_train': None,
        'anfis_y_train': None,
        'anfis_X_val': None,
        'anfis_y_val': None,
        'anfis_X_test': None,
        'anfis_y_test': None,
        'anfis_scaler_X': None,
        'anfis_scaler_y': None,
        'anfis_training_method': 'Hybrid',
        'anfis_feature_names': None,
        'anfis_target_name': None,
        'anfis_problem_type': 'Regression',
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar():
    """Render sidebar with ANFIS quick actions"""

    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 0.25rem 0 0.125rem 0; margin-top: 0.5rem;">
            <h2 style="margin: 0.25rem 0 0.125rem 0; color: #667eea;">ANFIS</h2>
            <p style="color: #6b7280; font-size: 0.9rem; margin: 0;">
                Adaptive Neuro-Fuzzy Inference System
            </p>
        </div>
        <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 0.25rem 0 0.5rem 0;">
        """, unsafe_allow_html=True)

        # Problem type selector
        st.markdown("### 🎯 Problem Type")

        problem_type = st.radio(
            "Select task type:",
            options=["Regression", "Classification"],
            index=0 if st.session_state.get('anfis_problem_type', 'Regression') == 'Regression' else 1,
            key='anfis_problem_type_selector',
            horizontal=True,
            help="**Regression**: Predict continuous values (e.g., temperature, price)\n\n**Classification**: Predict discrete classes (e.g., 0/1, categories)"
        )

        # Update session state
        if problem_type != st.session_state.get('anfis_problem_type', 'Regression'):
            st.session_state.anfis_problem_type = problem_type
            # Clear predictions when changing type
            for key in ['anfis_y_pred_train', 'anfis_y_pred_val', 'anfis_y_pred_test',
                       'manual_prediction_result', 'manual_prediction_class', 'manual_prediction_prob',
                       'batch_prediction_results', 'batch_is_classification']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        st.markdown("<hr style='border: none; border-top: 1px solid #e5e7eb; margin: 0.75rem 0;'>",
                   unsafe_allow_html=True)

        # Quick Status
        st.markdown("### 📊 Status")

        col1, col2 = st.columns(2)

        # Dataset status: check if dataset loaded AND preprocessed (split done)
        with col1:
            has_dataset = st.session_state.get('anfis_dataset', None) is not None
            has_split = st.session_state.get('anfis_X_train', None) is not None

            if has_dataset and has_split:
                dataset_status = "✅"
            elif has_dataset:
                dataset_status = "⚙️"  # Loaded but not processed
            else:
                dataset_status = "⏳"

            st.metric("Dataset", dataset_status)

        # Model status: check if trained AND valid for current problem type
        with col2:
            is_trained = st.session_state.get('anfis_trained', False)
            has_model = st.session_state.get('anfis_model', None) is not None

            # Check if model is valid for current problem type
            model_valid = False
            if is_trained and has_model:
                current_problem = st.session_state.get('anfis_problem_type', 'Regression')
                model_is_classification = st.session_state.anfis_model.classification
                model_valid = (current_problem == 'Classification') == model_is_classification

            model_status = "✅" if model_valid else "⏳"
            st.metric("Model", model_status)

        # Dataset info
        if st.session_state.get('anfis_dataset', None) is not None:
            dataset_name = st.session_state.get('anfis_dataset_name', 'Custom')
            n_samples = len(st.session_state.anfis_dataset)

            if st.session_state.get('anfis_X_train', None) is not None:
                st.info(f"📦 **{dataset_name}**\n\n{n_samples} samples (split ready)")
            else:
                st.warning(f"📦 **{dataset_name}**\n\n{n_samples} samples (not split)")

        # Model info
        if st.session_state.get('anfis_trained', False) and st.session_state.get('anfis_model', None) is not None:
            current_problem = st.session_state.get('anfis_problem_type', 'Regression')
            model_is_classification = st.session_state.anfis_model.classification

            icon = "🎯" if model_is_classification else "📈"
            model_type = "Classification" if model_is_classification else "Regression"

            # Check if model matches current problem type
            if (current_problem == 'Classification') == model_is_classification:
                st.success(f"{icon} **{st.session_state.anfis_model.n_rules}** rules ({model_type})")
            else:
                st.error(f"⚠️ Model is {model_type} but problem type is {current_problem}!\n\nPlease retrain.")

        st.markdown("<hr style='border: none; border-top: 1px solid #e5e7eb; margin: 1rem 0;'>",
                   unsafe_allow_html=True)

        # Dataset source selector (from dataset_tab)
        dataset_tab.render_sidebar_controls()

        st.markdown("<hr style='border: none; border-top: 1px solid #e5e7eb; margin: 1rem 0;'>",
                   unsafe_allow_html=True)

        # Quick Actions
        st.markdown("### 🚀 Quick Actions")

        if st.button("🔄 Reset All", use_container_width=True, type="secondary"):
            reset_all()

        if st.session_state.get('anfis_model', None) is not None:
            st.markdown("#### 💾 Model I/O")

            import pickle

            if st.button("📥 Export Model", use_container_width=True):
                try:
                    model_bytes = pickle.dumps(st.session_state.anfis_model)
                    st.download_button(
                        label="💾 Download Model (.pkl)",
                        data=model_bytes,
                        file_name="anfis_model.pkl",
                        mime="application/octet-stream",
                        key='export_model_download'
                    )
                except Exception as e:
                    st.error(f"❌ Export failed: {str(e)}")

            uploaded_model = st.file_uploader(
                "📂 Import Model",
                type=['pkl'],
                key='anfis_model_import_sidebar'
            )

            if uploaded_model is not None:
                try:
                    model = pickle.load(uploaded_model)
                    if hasattr(model, 'predict') and hasattr(model, 'n_inputs'):
                        st.session_state.anfis_model = model
                        st.session_state.anfis_trained = True
                        st.success("✅ Model imported successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid ANFIS model file!")
                except Exception as e:
                    st.error(f"❌ Import failed: {str(e)}")

        st.markdown("<hr style='border: none; border-top: 1px solid #e5e7eb; margin: 1rem 0;'>",
                   unsafe_allow_html=True)

        # Help
        with st.expander("❓ Help"):
            st.markdown("""
            **Workflow:**
            1. 📊 Load dataset
            2. 🎯 Configure & train
            3. 📈 View metrics
            4. 🔮 Make predictions

            **Tips:**
            - Start with 2-3 MFs per input
            - Use validation split (20%)
            - Normalize data for better convergence
            - Try hybrid learning first
            """)


def render_main_content():
    """Render main content area with tabs"""

    st.markdown("""
    <div style="text-align: center; padding: 0.5rem 0;">
        <h3 style="color: #6b7280; font-weight: 500; margin: 0; font-size: 1.1rem;">
            ANFIS - Adaptive Neuro-Fuzzy Inference System
        </h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='border-bottom: 1px solid #e5e7eb; margin: 0.5rem 0 1.5rem 0;'></div>",
               unsafe_allow_html=True)

    # Tabs for different views
    tabs = st.tabs([
        "Dataset",
        "Training",
        "Metrics",
        "Prediction",
        "Model Analysis",
        "What is ANFIS?"
    ])

    with tabs[0]:
        dataset_tab.render()

    with tabs[1]:
        training_tab.render()

    with tabs[2]:
        metrics_tab.render()

    with tabs[3]:
        prediction_tab.render()

    with tabs[4]:
        analysis_tab.render()

    with tabs[5]:
        overview_tab.render()


def reset_all():
    """Reset all ANFIS session state"""
    keys_to_reset = [
        'anfis_model', 'anfis_trained', 'anfis_dataset', 'anfis_dataset_name',
        'anfis_X_train', 'anfis_y_train', 'anfis_X_val', 'anfis_y_val',
        'anfis_X_test', 'anfis_y_test', 'anfis_scaler_X', 'anfis_scaler_y',
        'anfis_feature_names', 'anfis_target_name',
        'anfis_custom_formula', 'anfis_custom_formula_range',
        'anfis_y_pred_train', 'anfis_y_pred_val', 'anfis_y_pred_test',
        'manual_prediction_result', 'manual_prediction_inputs',
        'manual_prediction_class', 'manual_prediction_prob', 'manual_prediction_proba',
        'batch_prediction_results', 'batch_is_classification',
        'anfis_ovr_applied', 'anfis_ovr_selected_class', 'anfis_ovr_class_names',
        'anfis_dataset_original'
    ]

    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

    st.success("✅ All data cleared!")
    st.rerun()


if __name__ == "__main__":
    run()
