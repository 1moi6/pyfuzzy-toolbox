"""
Dataset Tab for ANFIS Module
Handles dataset loading, splitting, and preprocessing
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes


def reset_dataset_state():
    """Reset all dataset-related states when loading a new dataset"""
    keys_to_reset = [
        # Split and preprocessing states
        'anfis_X_train', 'anfis_X_val', 'anfis_X_test',
        'anfis_y_train', 'anfis_y_val', 'anfis_y_test',
        'anfis_scaler_X', 'anfis_scaler_y',
        'anfis_feature_names', 'anfis_target_name',

        # One-vs-Rest states
        'anfis_ovr_applied', 'anfis_ovr_selected_class',
        'anfis_ovr_class_names', 'anfis_original_y_train',
        'anfis_original_y_val', 'anfis_original_y_test',
        'anfis_dataset_original',

        # Model states
        'anfis_trained', 'anfis_model',
        'anfis_training_losses', 'anfis_val_losses',
        'anfis_train_rmse', 'anfis_val_rmse',
        'anfis_config', 'anfis_train_config',

        # Prediction states
        'anfis_y_pred_train', 'anfis_y_pred_val', 'anfis_y_pred_test',
        'manual_prediction_result', 'manual_prediction_class',
        'manual_prediction_prob', 'manual_prediction_proba',
        'manual_prediction_inputs',
        'batch_prediction_results', 'batch_is_classification'
    ]

    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]


# Classic datasets information
CLASSIC_DATASETS = {
    "Iris": {
        "icon": "🌸",
        "description": "Iris flower species classification",
        "samples": 150,
        "features": 4,
        "target": "3 classes (Setosa, Versicolor, Virginica)",
        "task": "Classification",
        "loader": load_iris,
        "details": """
        Classic pattern recognition dataset with 4 morphological features (sepal/petal length and width)
        for classifying iris flowers into 3 species. Ideal for learning classification algorithms.
        """
    },
    "Wine": {
        "icon": "🍷",
        "description": "Wine cultivar recognition",
        "samples": 178,
        "features": 13,
        "target": "3 classes",
        "task": "Classification",
        "loader": load_wine,
        "details": """
        Results of chemical analysis of wines grown in the same region in Italy but derived from
        three different cultivars. Contains 13 constituents found in each wine type.
        """
    },
    "Breast Cancer": {
        "icon": "🎗️",
        "description": "Breast cancer Wisconsin diagnostic",
        "samples": 569,
        "features": 30,
        "target": "2 classes (Malignant, Benign)",
        "task": "Classification",
        "loader": load_breast_cancer,
        "details": """
        Features computed from digitized image of fine needle aspirate of breast mass.
        Describes characteristics of cell nuclei present in the image. Binary classification
        for medical diagnosis (malignant vs benign).
        """
    },
    "Diabetes": {
        "icon": "💉",
        "description": "Diabetes disease progression",
        "samples": 442,
        "features": 10,
        "target": "Continuous (disease progression)",
        "task": "Regression",
        "loader": load_diabetes,
        "details": """
        Ten baseline variables (age, sex, BMI, blood pressure, and six blood serum measurements)
        to predict a quantitative measure of disease progression one year after baseline.
        """
    }
}


def render_sidebar_controls():
    """Render dataset controls in sidebar"""

    st.markdown("### Dataset Source")

    # Get previous source to detect changes
    previous_source = st.session_state.get('anfis_dataset_source_selected', None)

    dataset_source = st.selectbox(
        "Choose data source",
        ["Upload CSV", "Classic Datasets", "Synthetic Data"],
        key='anfis_dataset_source',
        label_visibility='collapsed'
    )

    # If source changed, clear the dataset
    if previous_source is not None and previous_source != dataset_source:
        if 'anfis_dataset' in st.session_state:
            del st.session_state.anfis_dataset
        if 'anfis_dataset_name' in st.session_state:
            del st.session_state.anfis_dataset_name
        reset_dataset_state()

    st.session_state.anfis_dataset_source_selected = dataset_source

    # Classic dataset selector
    if dataset_source == "Classic Datasets":
        st.markdown("---")

        # Get previous selection to detect changes
        previous_classic = st.session_state.get('anfis_selected_classic', None)

        selected_dataset = st.selectbox(
            "Select dataset",
            list(CLASSIC_DATASETS.keys()),
            format_func=lambda x: f"{CLASSIC_DATASETS[x]['icon']} {x}",
            key='anfis_classic_dataset_choice',
            label_visibility='collapsed'
        )

        # If classic dataset selection changed, clear the current dataset
        if previous_classic is not None and previous_classic != selected_dataset:
            if 'anfis_dataset' in st.session_state:
                del st.session_state.anfis_dataset
            if 'anfis_dataset_name' in st.session_state:
                del st.session_state.anfis_dataset_name
            reset_dataset_state()

        st.session_state.anfis_selected_classic = selected_dataset

        info = CLASSIC_DATASETS[selected_dataset]
        st.caption(f"{info['samples']} samples × {info['features']} features")
        st.caption(f"**{info['task']}**")


def render():
    """Render dataset management tab"""

    source = st.session_state.get('anfis_dataset_source_selected', 'Upload CSV')

    # Render appropriate section
    if source == "Upload CSV":
        render_upload_section()
    elif source == "Classic Datasets":
        render_classic_dataset_section()
    else:
        render_synthetic_section()

    # Dataset info, split, and preprocessing (only show if dataset is loaded)
    if st.session_state.get('anfis_dataset', None) is not None:
        st.markdown("")
        render_dataset_info()

        st.markdown("")
        render_preprocessing_section()

        # Add space at the end
        st.markdown("")
        st.markdown("")


def render_upload_section():
    """Render CSV upload section"""

    st.markdown("### Upload CSV")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        key='anfis_dataset_upload'
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # Reset all dataset-related states
            reset_dataset_state()

            st.session_state.anfis_dataset = df
            st.session_state.anfis_dataset_name = uploaded_file.name.replace('.csv', '')
            st.success(f"Dataset loaded: {df.shape[0]} samples × {df.shape[1]} columns")
            st.rerun()
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")

    with st.expander("CSV Format Help"):
        st.code("""feature1,feature2,feature3,target
1.2,3.4,5.6,0.5
2.3,4.5,6.7,1.2""", language='csv')
        st.caption("Rows = samples | Columns = features | Last column = target")


def render_classic_dataset_section():
    """Render classic dataset section"""

    selected = st.session_state.get('anfis_selected_classic', 'Iris')
    info = CLASSIC_DATASETS[selected]

    st.markdown(f"##### {info['icon']} {selected} - {info['description']}")
    # st.info(info['description'])

    st.markdown(info['details'])

    if st.button(f"Load {selected} Dataset", type="primary", width="stretch", key='load_classic_btn'):
        load_classic_dataset(selected, info['loader'], info['task'])


def load_classic_dataset(name, loader_func, task):
    """Load a classic sklearn dataset"""

    try:
        with st.spinner(f"Loading {name}..."):
            data = loader_func()

            feature_names = data.feature_names if hasattr(data, 'feature_names') else [f"X{i+1}" for i in range(data.data.shape[1])]

            df = pd.DataFrame(data.data, columns=feature_names)
            df['target'] = data.target

            # Reset all dataset-related states
            reset_dataset_state()

            st.session_state.anfis_dataset = df
            st.session_state.anfis_dataset_name = name
            st.session_state.anfis_dataset_task = task

            st.success(f"{name} dataset loaded successfully!")
            st.balloons()
            st.rerun()

    except Exception as e:
        st.error(f"Error loading {name}: {str(e)}")


def render_synthetic_section():
    """Render synthetic data generation section"""

    st.markdown("### Synthetic Data")

    # Get previous synthetic config to detect changes
    previous_synth_config = st.session_state.get('anfis_synth_config', None)

    col1, col2 = st.columns(2)

    with col1:
        n_samples = st.number_input("Samples", 100, 5000, 500, step=100, key='synth_samples')
        n_features = st.number_input("Features", 1, 5, 2, key='synth_features')

    with col2:
        noise = st.slider("Noise level", 0.0, 1.0, 0.1, step=0.05, key='synth_noise')
        function_type = st.selectbox(
            "Function type",
            ['Sine + Cosine', 'Polynomial', 'Exponential', 'Mixed', 'Custom Formula'],
            key='synth_function'
        )

    # Create current config signature
    current_synth_config = (n_samples, n_features, noise, function_type)

    # If synthetic config changed, clear the current dataset
    if previous_synth_config is not None and previous_synth_config != current_synth_config:
        if 'anfis_dataset' in st.session_state:
            del st.session_state.anfis_dataset
        if 'anfis_dataset_name' in st.session_state:
            del st.session_state.anfis_dataset_name
        reset_dataset_state()

    # Store current config
    st.session_state.anfis_synth_config = current_synth_config

    # Custom formula inputs
    custom_formula = None
    x_min, x_max = -3, 3

    if function_type == 'Custom Formula':
        st.markdown("---")
        st.markdown("**Custom Function Configuration**")

        col1, col2 = st.columns([2, 1])

        with col1:
            custom_formula = st.text_area(
                "Python expression",
                value="np.sin(X[:, 0]) * np.exp(-X[:, 1]**2)",
                height=80,
                key='custom_formula_input',
                help="Use numpy as 'np' and features as X[:, 0], X[:, 1], etc."
            )

        with col2:
            st.markdown("**Input Range**")
            x_min = st.number_input("Min", -10.0, 0.0, -3.0, step=0.5, key='custom_x_min')
            x_max = st.number_input("Max", 0.0, 10.0, 3.0, step=0.5, key='custom_x_max')

        # Examples
        with st.expander("Formula Examples"):
            st.code("""# Trigonometric
np.sin(X[:, 0]) + np.cos(X[:, 1])

# Polynomial
X[:, 0]**2 + X[:, 1]**2 - X[:, 0] * X[:, 1]

# Exponential
np.exp(-X[:, 0]**2) * np.cos(X[:, 1])

# Mixed
np.sin(X[:, 0]) * np.exp(-X[:, 1]**2)

# Nonlinear
np.tanh(X[:, 0]) + np.sqrt(np.abs(X[:, 1]))

# Available: np.sin, np.cos, np.exp, np.log, np.sqrt,
#            np.tanh, np.abs, basic operators (+, -, *, /, **)
""", language='python')
            st.caption("⚠️ Use X[:, 0] for first feature, X[:, 1] for second, etc.")

    # Formula display
    with st.expander("📐 Function Formula", expanded=(function_type != 'Custom Formula')):
        if function_type == 'Sine + Cosine':
            if n_features == 1:
                st.latex(r"y = \sin(x_1) + \cos(x_1) + \epsilon")
            else:
                st.latex(r"y = \sin(x_1) + \cos(x_2) + \epsilon")
            st.caption(r"where $\epsilon \sim \mathcal{N}(0, \sigma^2)$ with $\sigma = " + f"{noise:.3f}$")
            st.caption(r"Input range: $x_i \in [-3, 3]$")

        elif function_type == 'Polynomial':
            if n_features == 1:
                st.latex(r"y = x_1^2 + x_1 + \epsilon")
            else:
                st.latex(r"y = x_1^2 + x_2^2 + \epsilon")
            st.caption(r"where $\epsilon \sim \mathcal{N}(0, \sigma^2)$ with $\sigma = " + f"{noise:.3f}$")
            st.caption(r"Input range: $x_i \in [-3, 3]$")

        elif function_type == 'Exponential':
            if n_features == 1:
                st.latex(r"y = e^{-x_1^2} + x_1 + \epsilon")
            else:
                st.latex(r"y = e^{-x_1^2} + e^{-x_2^2} + \epsilon")
            st.caption(r"where $\epsilon \sim \mathcal{N}(0, \sigma^2)$ with $\sigma = " + f"{noise:.3f}$")
            st.caption(r"Input range: $x_i \in [-3, 3]$")

        elif function_type == 'Mixed':
            if n_features == 1:
                st.latex(r"y = \sin(x_1) + \cos(2x_1) + \epsilon")
            else:
                st.latex(r"y = \sin(x_1) \times \cos(x_2) + \epsilon")
            st.caption(r"where $\epsilon \sim \mathcal{N}(0, \sigma^2)$ with $\sigma = " + f"{noise:.3f}$")
            st.caption(r"Input range: $x_i \in [-3, 3]$")

        else:  # Custom Formula
            if custom_formula:
                st.code(f"y = {custom_formula} + ε", language='python')
                st.caption(r"where $\epsilon \sim \mathcal{N}(0, \sigma^2)$ with $\sigma = " + f"{noise:.3f}$")
                st.caption(r"Input range: $x_i \in [" + f"{x_min:.1f}, {x_max:.1f}" + r"]$")
            else:
                st.info("Enter a custom formula above")

    st.markdown("")

    if st.button("Generate Dataset", type="primary", width="stretch", key='generate_synth_btn'):
        generate_synthetic_dataset(n_samples, n_features, noise, function_type, custom_formula, x_min, x_max)


def generate_synthetic_dataset(n_samples, n_features, noise, function_type, custom_formula=None, x_min=-3, x_max=3):
    """Generate synthetic dataset"""

    try:
        with st.spinner("Generating data..."):
            X = np.random.uniform(x_min, x_max, (n_samples, n_features))

            if function_type == 'Sine + Cosine':
                if n_features == 1:
                    y = np.sin(X[:, 0]) + np.cos(X[:, 0])
                else:
                    y = np.sin(X[:, 0]) + np.cos(X[:, 1])

            elif function_type == 'Polynomial':
                if n_features == 1:
                    y = X[:, 0]**2 + X[:, 0]
                else:
                    y = X[:, 0]**2 + X[:, 1]**2

            elif function_type == 'Exponential':
                if n_features == 1:
                    y = np.exp(-X[:, 0]**2) + X[:, 0]
                else:
                    y = np.exp(-X[:, 0]**2) + np.exp(-X[:, 1]**2)

            elif function_type == 'Mixed':
                if n_features == 1:
                    y = np.sin(X[:, 0]) + np.cos(2 * X[:, 0])
                else:
                    y = np.sin(X[:, 0]) * np.cos(X[:, 1])

            elif function_type == 'Custom Formula':
                if not custom_formula:
                    st.error("Please enter a custom formula")
                    return

                # Evaluate custom formula
                try:
                    # Create safe namespace with numpy
                    namespace = {'np': np, 'X': X}
                    y = eval(custom_formula, {"__builtins__": {}}, namespace)

                    # Ensure y is 1D array
                    if isinstance(y, np.ndarray):
                        y = y.ravel()
                    else:
                        y = np.array([y] * n_samples)

                    if len(y) != n_samples:
                        st.error(f"Formula output size ({len(y)}) doesn't match samples ({n_samples})")
                        return

                except Exception as e:
                    st.error(f"Error evaluating formula: {str(e)}\n\nMake sure to use X[:, 0], X[:, 1], etc. for features")
                    return

            else:
                st.error(f"Unknown function type: {function_type}")
                return

            # Add noise
            y += np.random.normal(0, noise, n_samples)

            # Create dataframe
            columns = [f'X{i+1}' for i in range(n_features)] + ['Y']
            df = pd.DataFrame(np.column_stack([X, y]), columns=columns)

            # Reset all dataset-related states
            reset_dataset_state()

            # Store in session state
            st.session_state.anfis_dataset = df
            st.session_state.anfis_dataset_name = f"Synthetic ({function_type})"

            # Store custom formula if used
            if function_type == 'Custom Formula' and custom_formula:
                st.session_state.anfis_custom_formula = custom_formula
                st.session_state.anfis_custom_formula_range = (x_min, x_max)

            st.success(f"✅ Generated {n_samples} samples × {n_features} features")
            st.balloons()
            st.rerun()

    except Exception as e:
        st.error(f"❌ Error generating dataset: {str(e)}")


def render_dataset_info():
    """Display dataset information"""

    df = st.session_state.get('anfis_dataset')
    if df is None:
        return
    dataset_name = st.session_state.get('anfis_dataset_name', 'Dataset')

    with st.expander(f"**{dataset_name}** - Dataset Info", expanded=False):
        # Get dataset details if it's a classic dataset
        dataset_info = None
        if dataset_name in CLASSIC_DATASETS:
            info = CLASSIC_DATASETS[dataset_name]
            dataset_info = {
                'Samples': info['samples'],
                'Features': info['features'],
                'Target': info['target'],
                'Task': info['task'],
                'Details': info['details']
            }

        # Show dataset details first (if available)
        if dataset_info:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Samples:** {dataset_info['Samples']}")
                st.markdown(f"**Features:** {dataset_info['Features']}")

            with col2:
                st.markdown(f"**Target:** {dataset_info['Target']}")
                st.markdown(f"**Task:** {dataset_info['Task']}")

            st.markdown("")
            st.markdown(dataset_info['Details'])
            st.markdown("---")

        # Show formula for synthetic datasets
        elif dataset_name.startswith('Synthetic'):
            # Extract function type from dataset name
            function_type = dataset_name.replace('Synthetic (', '').replace(')', '')

            st.markdown("**📐 Generation Formula**")
            st.markdown("")

            if function_type == 'Sine + Cosine':
                st.latex(r"y = \sin(x_1) + \cos(x_2) + \epsilon")
                st.caption(r"where $\epsilon \sim \mathcal{N}(0, \sigma^2)$ - Gaussian noise")

            elif function_type == 'Polynomial':
                st.latex(r"y = x_1^2 + x_2^2 + \epsilon")
                st.caption(r"where $\epsilon \sim \mathcal{N}(0, \sigma^2)$ - Quadratic function")

            elif function_type == 'Exponential':
                st.latex(r"y = e^{-x_1^2} + e^{-x_2^2} + \epsilon")
                st.caption(r"where $\epsilon \sim \mathcal{N}(0, \sigma^2)$ - Gaussian basis functions")

            elif function_type == 'Mixed':
                st.latex(r"y = \sin(x_1) \times \cos(x_2) + \epsilon")
                st.caption(r"where $\epsilon \sim \mathcal{N}(0, \sigma^2)$ - Trigonometric product")

            elif function_type == 'Custom Formula':
                # Retrieve stored custom formula
                custom_formula = st.session_state.get('anfis_custom_formula', None)
                custom_range = st.session_state.get('anfis_custom_formula_range', (-3, 3))

                if custom_formula:
                    st.code(f"y = {custom_formula} + ε", language='python')
                    st.caption(r"where $\epsilon \sim \mathcal{N}(0, \sigma^2)$ - Custom noise")
                    st.caption(f"Input range: $x_i \\in [{custom_range[0]:.1f}, {custom_range[1]:.1f}]$")
                else:
                    st.info("Custom formula dataset")

            st.markdown("---")

        # Data preview tabs
        tab1, tab2, tab3 = st.tabs(["Preview", "Statistics", "Columns"])

        with tab1:
            st.dataframe(df.head(10), width="stretch")

        with tab2:
            st.dataframe(df.describe(), width="stretch")

        with tab3:
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.values,
                'Non-Null': df.count().values,
                'Unique': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(col_info, width="stretch")


def render_preprocessing_section():
    """Render preprocessing section with split and normalization"""

    with st.expander("**Preprocessing** - Split & Normalize", expanded=False):

        # One-vs-Rest strategy (only for classification problems)
        is_classification = st.session_state.get('anfis_problem_type', 'Regression') == 'Classification'

        if is_classification:
            st.markdown("##### Multiclass Strategy")
            render_one_vs_rest()
            st.markdown("---")

        # Dataset split
        st.markdown("#### Dataset Split")
        render_dataset_split()

        st.markdown("---")

        # Data normalization
        st.markdown("#### Normalization")
        render_data_normalization()


def render_one_vs_rest():
    """Render One-vs-Rest transformation for multiclass problems"""

    df = st.session_state.get('anfis_dataset')
    if df is None:
        return

    # Try to detect target column (last column by default)
    target_col = st.session_state.get('anfis_target_column', df.columns[-1])

    # Check if target column exists in dataframe
    if target_col not in df.columns:
        target_col = df.columns[-1]

    y = df[target_col].values

    # Check if it's a multiclass problem
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)

    # Check if already applied
    ovr_applied = st.session_state.get('anfis_ovr_applied', False)

    # If OvR already applied, show status (even if now binary)
    if ovr_applied:
        # Show current transformation
        selected_class = st.session_state.get('anfis_ovr_selected_class', unique_classes[0])
        class_names = st.session_state.get('anfis_ovr_class_names', {})

        st.success(f"✅ **One-vs-Rest Applied**")

        col1, col2 = st.columns(2,vertical_alignment='bottom')

        with col1:
            if isinstance(selected_class, str):
                st.markdown(f"**Positive class:** `{selected_class}`")
            else:
                class_name = class_names.get(selected_class, f"Class {selected_class}")
                st.markdown(f"**Positive class:** {class_name} (value = {selected_class})")
            st.caption("All other classes are grouped as negative class")

        with col2:
            if st.button("Reset to original", width='stretch', key='reset_ovr_button'):
                restore_original_dataset()

        

    else:
        # OvR not applied yet
        if n_classes <= 2:
            st.info(f"📌 Binary classification detected ({n_classes} classes). One-vs-Rest not needed.")
            return

        # Multiclass problem - show configuration
        st.warning(f"⚠️ **Multiclass problem detected** ({n_classes} classes)")
        st.caption("ANFIS works best with binary classification. Use One-vs-Rest strategy.")

        st.markdown("")

        # Class selector and checkbox in 50-50 columns
        col1, col2 = st.columns(2,vertical_alignment='top')

        with col1:
            # Check if classes are string labels or numeric
            if isinstance(unique_classes[0], (str, np.str_)):
                # String labels
                selected_class = st.selectbox(
                    "Select positive class",
                    unique_classes.tolist(),
                    key='ovr_class_selector',
                    help="This class will be labeled as 1 (positive). All others will be 0 (negative)."
                )
                class_names = {cls: cls for cls in unique_classes}

            else:
                # Numeric labels - try to get class names if available
                dataset_name = st.session_state.get('anfis_dataset_name', '')

                # For classic datasets, get class names
                if dataset_name in CLASSIC_DATASETS:
                    info = CLASSIC_DATASETS[dataset_name]
                    if 'target' in info:
                        # Try to extract class names from target description
                        class_names = {i: f"Class {i}" for i in unique_classes}
                else:
                    class_names = {i: f"Class {i}" for i in unique_classes}

                selected_class = st.selectbox(
                    "Select positive class",
                    unique_classes.tolist(),
                    format_func=lambda x: class_names.get(x, f"Class {x}"),
                    key='ovr_class_selector',
                    help="This class will be labeled as 1 (positive). All others will be 0 (negative)."
                )

        with col2:
            # Checkbox to apply One-vs-Rest
            # st.markdown("")  # Align with selectbox
            current_ovr_state = st.session_state.get('anfis_ovr_applied', False)

            apply_ovr = st.checkbox(
                "Apply One-vs-Rest",
                value=current_ovr_state,
                key='apply_ovr_checkbox',
                help="Check to apply the transformation"
            )

        # Info caption below columns
            st.caption(f"Value `{selected_class}` → 1 (positive) | Others → 0 (negative)")

        # Apply or remove transformation based on checkbox
        if apply_ovr and not current_ovr_state:
            # Checkbox was just checked - apply transformation
            apply_one_vs_rest(df, target_col, selected_class, class_names)
        elif not apply_ovr and current_ovr_state:
            # Checkbox was just unchecked - restore original
            restore_original_dataset()

        st.markdown("")

        # Show class distribution
        with st.expander("📊 Class Distribution Preview"):
            class_counts = pd.Series(y).value_counts().sort_index()

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Original Distribution**")
                for cls in unique_classes:
                    count = class_counts.get(cls, 0)
                    pct = (count / len(y)) * 100
                    cls_name = class_names.get(cls, f"Class {cls}")
                    st.caption(f"• {cls_name}: {count} ({pct:.1f}%)")

            with col2:
                st.markdown("**After One-vs-Rest**")
                positive_count = np.sum(y == selected_class)
                negative_count = len(y) - positive_count
                pos_pct = (positive_count / len(y)) * 100
                neg_pct = (negative_count / len(y)) * 100

                st.caption(f"• Positive (1): {positive_count} ({pos_pct:.1f}%)")
                st.caption(f"• Negative (0): {negative_count} ({neg_pct:.1f}%)")


def apply_one_vs_rest(df, target_col, selected_class, class_names):
    """Apply One-vs-Rest transformation to dataset"""

    try:
        # Store original dataset
        if 'anfis_dataset_original' not in st.session_state:
            st.session_state.anfis_dataset_original = df.copy()

        # Transform target column
        y_binary = (df[target_col] == selected_class).astype(int)

        # Create new dataframe with binary target
        df_binary = df.copy()
        df_binary[target_col] = y_binary

        # Update session state
        st.session_state.anfis_dataset = df_binary
        st.session_state.anfis_ovr_applied = True
        st.session_state.anfis_ovr_selected_class = selected_class
        st.session_state.anfis_ovr_class_names = class_names

        # Reset any previous split/normalization
        st.session_state.anfis_X_train = None
        st.session_state.anfis_y_train = None
        st.session_state.anfis_X_val = None
        st.session_state.anfis_y_val = None
        st.session_state.anfis_X_test = None
        st.session_state.anfis_y_test = None
        st.session_state.anfis_scaler_X = None
        st.session_state.anfis_scaler_y = None

        positive_count = np.sum(y_binary == 1)
        negative_count = np.sum(y_binary == 0)

        st.success(f"✅ One-vs-Rest transformation applied successfully!\n\n" +
                   f"Positive class (1): {positive_count} samples | Negative class (0): {negative_count} samples")
        st.rerun()

    except Exception as e:
        st.error(f"❌ Error applying One-vs-Rest: {str(e)}")


def restore_original_dataset():
    """Restore original dataset (undo One-vs-Rest transformation)"""
    try:
        # Restore original dataset
        if 'anfis_dataset_original' in st.session_state:
            st.session_state.anfis_dataset = st.session_state.anfis_dataset_original.copy()

        # Clear OVR state
        st.session_state.anfis_ovr_applied = False
        if 'anfis_ovr_selected_class' in st.session_state:
            del st.session_state.anfis_ovr_selected_class
        if 'anfis_ovr_class_names' in st.session_state:
            del st.session_state.anfis_ovr_class_names
        if 'anfis_dataset_original' in st.session_state:
            del st.session_state.anfis_dataset_original

        # Reset split and normalization
        st.session_state.anfis_X_train = None
        st.session_state.anfis_y_train = None
        st.session_state.anfis_X_val = None
        st.session_state.anfis_y_val = None
        st.session_state.anfis_X_test = None
        st.session_state.anfis_y_test = None
        st.session_state.anfis_scaler_X = None
        st.session_state.anfis_scaler_y = None

        st.info("ℹ️ Original dataset restored")
        st.rerun()

    except Exception as e:
        st.error(f"❌ Error restoring dataset: {str(e)}")


def render_dataset_split():
    """Render dataset splitting configuration"""

    df = st.session_state.get('anfis_dataset')
    if df is None:
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        target_col = st.selectbox(
            "Target column",
            df.columns.tolist(),
            index=len(df.columns) - 1,
            key='anfis_target_column'
        )

        feature_cols = st.multiselect(
            "Feature columns",
            [col for col in df.columns if col != target_col],
            default=[col for col in df.columns if col != target_col],
            key='anfis_feature_columns'
        )

    with col2:
        st.markdown("**Split ratios (%)**")

        train_size = st.slider("Train", 50, 90, 70, 5, key='split_train')

        remaining = 100 - train_size
        val_size = st.slider("Val", 0, remaining, min(20, remaining), 5, key='split_val')

        test_size = remaining - val_size
        st.metric("Test", f"{test_size}%")

    if len(feature_cols) == 0:
        st.warning("Please select at least one feature column")
        return

    if st.button("Split Dataset", type="primary", width="stretch", key='split_button'):
        split_dataset(df, feature_cols, target_col, train_size, val_size, test_size)

    msg = ""
    if st.session_state.get('anfis_X_train', None) is not None:
        tr_size = len(st.session_state.anfis_X_train)
        msg += f"Train size: {tr_size} | "
    if st.session_state.get('anfis_X_val', None) is not None:
        val_size = len(st.session_state.anfis_X_val)
        msg += f"Train size: {val_size} | "
    if st.session_state.get('anfis_X_test', None) is not None:
        test_size = len(st.session_state.anfis_X_test)
        msg += f"Train size: {test_size}"
    
    if st.session_state.get('anfis_X_train', None) is not None:
        st.success("**Split completed!** "+msg)

def split_dataset(df, feature_cols, target_col, train_pct, val_pct, test_pct):
    """Split dataset into train/val/test sets"""

    try:
        X = df[feature_cols].values
        y = df[target_col].values.reshape(-1, 1)

        st.session_state.anfis_feature_names = feature_cols
        st.session_state.anfis_target_name = target_col

        # Split logic
        if test_pct > 0 or val_pct > 0:
            test_val_size = (val_pct + test_pct) / 100
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=test_val_size, random_state=42
            )

            if val_pct > 0 and test_pct > 0:
                test_ratio = test_pct / (val_pct + test_pct)
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=test_ratio, random_state=42
                )
            elif val_pct > 0:
                X_val, X_test = X_temp, None
                y_val, y_test = y_temp, None
            else:
                X_val, X_test = None, X_temp
                y_val, y_test = None, y_temp
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
            X_test, y_test = None, None

        # Store in session state
        st.session_state.anfis_X_train = X_train
        st.session_state.anfis_y_train = y_train
        st.session_state.anfis_X_val = X_val
        st.session_state.anfis_y_val = y_val
        st.session_state.anfis_X_test = X_test
        st.session_state.anfis_y_test = y_test

        # Reset scalers
        st.session_state.anfis_scaler_X = None
        st.session_state.anfis_scaler_y = None

        msg = f"Dataset split: {len(X_train)} train"
        if X_val is not None:
            msg += f", {len(X_val)} val"
        if X_test is not None:
            msg += f", {len(X_test)} test"

        st.success(msg)
        st.rerun()

    except Exception as e:
        st.error(f"Error splitting dataset: {str(e)}")


def render_data_normalization():
    """Render data normalization options"""

    if st.session_state.get('anfis_X_train', None) is None:
        st.info("Split the dataset first to enable normalization")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Features (X)**")

        normalize_X = st.checkbox(
            "Normalize features",
            value=False,
            key='anfis_normalize_x'
        )

        if normalize_X:
            scaler_type_X = st.selectbox(
                "Scaler",
                ["StandardScaler", "MinMaxScaler"],
                key='anfis_scaler_type_x'
            )

    with col2:
        st.markdown("**Target (y)**")

        normalize_y = st.checkbox(
            "Normalize target",
            value=False,
            key='anfis_normalize_y'
        )

        if normalize_y:
            scaler_type_y = st.selectbox(
                "Scaler",
                ["StandardScaler", "MinMaxScaler"],
                key='anfis_scaler_type_y'
            )

    if st.button("Apply Normalization", type="secondary", width="stretch", key='preprocess_button'):
        apply_preprocessing(normalize_X, normalize_y,
                          scaler_type_X if normalize_X else None,
                          scaler_type_y if normalize_y else None)

   
    
    if not st.session_state.get('anfis_scaler_X',None) is None:
        msg = "**Normalization applied:**"
        msg += f" X ({scaler_type_X})"
        if not st.session_state.get('anfis_scaler_y',None) is None:
            msg += f" y ({scaler_type_y})"
        st.success(msg)

def apply_preprocessing(normalize_X, normalize_y, scaler_type_X, scaler_type_y):
    """Apply preprocessing to dataset"""

    try:
        if normalize_X:
            scaler_X = StandardScaler() if scaler_type_X == "StandardScaler" else MinMaxScaler()

            st.session_state.anfis_X_train = scaler_X.fit_transform(st.session_state.anfis_X_train)

            if st.session_state.get('anfis_X_val', None) is not None:
                st.session_state.anfis_X_val = scaler_X.transform(st.session_state.anfis_X_val)

            if st.session_state.get('anfis_X_test', None) is not None:
                st.session_state.anfis_X_test = scaler_X.transform(st.session_state.anfis_X_test)

            st.session_state.anfis_scaler_X = scaler_X

        if normalize_y:
            scaler_y = StandardScaler() if scaler_type_y == "StandardScaler" else MinMaxScaler()

            st.session_state.anfis_y_train = scaler_y.fit_transform(st.session_state.anfis_y_train)

            if st.session_state.get('anfis_y_val', None) is not None:
                st.session_state.anfis_y_val = scaler_y.transform(st.session_state.anfis_y_val)

            if st.session_state.get('anfis_y_test', None) is not None:
                st.session_state.anfis_y_test = scaler_y.transform(st.session_state.anfis_y_test)

            st.session_state.anfis_scaler_y = scaler_y

        
        st.rerun()

    except Exception as e:
        st.error(f"Error applying normalization: {str(e)}")
