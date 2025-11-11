"""
Dataset Tab for Wang-Mendel Module
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
        'wm_X_train', 'wm_X_val', 'wm_X_test',
        'wm_y_train', 'wm_y_val', 'wm_y_test',
        'wm_scaler_X', 'wm_scaler_y',
        'wm_feature_names', 'wm_target_name',

        # Model states
        'wm_trained', 'wm_model', 'wm_system',
        'wm_training_stats',

        # Prediction states
        'wm_y_pred_train', 'wm_y_pred_val', 'wm_y_pred_test',
        'wm_manual_prediction_result', 'wm_manual_prediction_class',
        'wm_manual_prediction_prob', 'wm_manual_prediction_inputs',
        'wm_batch_prediction_results', 'wm_batch_is_classification'
    ]

    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]


# Classic datasets information (same as ANFIS)
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

    # Initialize default if not set
    if 'wm_dataset_source' not in st.session_state:
        st.session_state.wm_dataset_source = "Upload CSV"

    # Get previous source to detect changes
    previous_source = st.session_state.get('wm_dataset_source_prev', None)

    dataset_source = st.selectbox(
        "Choose data source",
        ["Upload CSV", "Classic Datasets", "Synthetic Data"],
        key='wm_dataset_source',
        label_visibility='collapsed'
    )

    # If source changed, clear the dataset
    if previous_source is not None and previous_source != dataset_source:
        if 'wm_dataset' in st.session_state:
            del st.session_state.wm_dataset
        if 'wm_dataset_name' in st.session_state:
            del st.session_state.wm_dataset_name
        reset_dataset_state()

    # Store current value as previous for next run
    st.session_state.wm_dataset_source_prev = dataset_source

    # Classic dataset selector
    if dataset_source == "Classic Datasets":
        st.markdown("---")

        # Initialize default if not set
        if 'wm_classic_dataset_choice' not in st.session_state:
            st.session_state.wm_classic_dataset_choice = list(CLASSIC_DATASETS.keys())[0]

        # Get previous selection to detect changes
        previous_classic = st.session_state.get('wm_classic_dataset_prev', None)

        selected_dataset = st.selectbox(
            "Select dataset",
            list(CLASSIC_DATASETS.keys()),
            format_func=lambda x: f"{CLASSIC_DATASETS[x]['icon']} {x}",
            key='wm_classic_dataset_choice',
            label_visibility='collapsed'
        )

        # If classic dataset selection changed, clear the current dataset
        if previous_classic is not None and previous_classic != selected_dataset:
            if 'wm_dataset' in st.session_state:
                del st.session_state.wm_dataset
            if 'wm_dataset_name' in st.session_state:
                del st.session_state.wm_dataset_name
            reset_dataset_state()

        # Store current value as previous for next run
        st.session_state.wm_classic_dataset_prev = selected_dataset

        info = CLASSIC_DATASETS[selected_dataset]
        st.caption(f"{info['samples']} samples × {info['features']} features")
        st.caption(f"**{info['task']}**")


def render():
    """Render dataset management tab"""

    source = st.session_state.get('wm_dataset_source', 'Upload CSV')

    # Render appropriate section
    if source == "Upload CSV":
        render_upload_section()
    elif source == "Classic Datasets":
        render_classic_dataset_section()
    else:
        render_synthetic_section()

    # Dataset info, split, and preprocessing (only show if dataset is loaded)
    if st.session_state.get('wm_dataset', None) is not None:
        st.markdown("")
        render_preprocessing_section()

        st.markdown("")
        render_dataset_info()

        # Add space at the end
        st.markdown("")
        st.markdown("")


def render_upload_section():
    """Render CSV upload section"""

    st.markdown("### Upload CSV")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        key='wm_dataset_upload'
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # Reset all dataset-related states
            reset_dataset_state()

            st.session_state.wm_dataset = df
            st.session_state.wm_dataset_name = uploaded_file.name.replace('.csv', '')
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

    selected = st.session_state.get('wm_classic_dataset_choice', 'Iris')
    info = CLASSIC_DATASETS[selected]

    st.markdown(f"##### {info['icon']} {selected} - {info['description']}")

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

            st.session_state.wm_dataset = df
            st.session_state.wm_dataset_name = name
            st.session_state.wm_dataset_task = task

            st.success(f"{name} dataset loaded successfully!")
            st.balloons()
            st.rerun()

    except Exception as e:
        st.error(f"Error loading {name}: {str(e)}")


def render_synthetic_section():
    """Render synthetic data generation section"""

    st.markdown("### Synthetic Data")

    # Get previous synthetic config to detect changes
    previous_synth_config = st.session_state.get('wm_synth_config', None)

    col1, col2 = st.columns(2)

    with col1:
        n_samples = st.number_input("Samples", 100, 5000, 500, step=100, key='wm_synth_samples')
        n_features = st.number_input("Features", 1, 5, 2, key='wm_synth_features')

    with col2:
        noise = st.slider("Noise level", 0.0, 1.0, 0.1, step=0.05, key='wm_synth_noise')
        function_type = st.selectbox(
            "Function type",
            ['Sine', 'Linear', 'Nonlinear'],
            key='wm_synth_function'
        )

    # Create current config signature
    current_synth_config = (n_samples, n_features, noise, function_type)

    # If synthetic config changed, clear the current dataset
    if previous_synth_config is not None and previous_synth_config != current_synth_config:
        if 'wm_dataset' in st.session_state:
            del st.session_state.wm_dataset
        if 'wm_dataset_name' in st.session_state:
            del st.session_state.wm_dataset_name
        reset_dataset_state()

    # Store current config
    st.session_state.wm_synth_config = current_synth_config

    # Formula display
    with st.expander("Function Formula", expanded=True):
        if function_type == 'Sine':
            if n_features == 1:
                st.latex(r"y = \sin(x) + 0.1x + \epsilon")
            else:
                st.latex(r"y = \sin(x_1) + \cos(x_2) + \epsilon")
            st.caption(r"where $\epsilon \sim \mathcal{N}(0, \sigma^2)$ with $\sigma = " + f"{noise:.3f}$")
            st.caption(r"Input range: $x_i \in [0, 2\pi]$")

        elif function_type == 'Linear':
            if n_features == 1:
                st.latex(r"y = -2x + 5 + \epsilon")
            else:
                st.latex(r"y = -2x_1 + 3x_2 + 5 + \epsilon")
            st.caption(r"where $\epsilon \sim \mathcal{N}(0, \sigma^2)$ with $\sigma = " + f"{noise:.3f}$")
            st.caption(r"Input range: $x_i \in [0, 10]$")

        elif function_type == 'Nonlinear':
            if n_features == 1:
                st.latex(r"y = x^2 - 2x + \epsilon")
            else:
                st.latex(r"y = x_1^2 + x_2^2 - x_1 \cdot x_2 + \epsilon")
            st.caption(r"where $\epsilon \sim \mathcal{N}(0, \sigma^2)$ with $\sigma = " + f"{noise:.3f}$")
            st.caption(r"Input range: $x_i \in [-5, 5]$")

    st.markdown("")

    if st.button("Generate Dataset", type="primary", width="stretch", key='wm_generate_synth_btn'):
        generate_synthetic_dataset(n_samples, n_features, noise, function_type)


def generate_synthetic_dataset(n_samples, n_features, noise, function_type):
    """Generate synthetic dataset"""

    try:
        with st.spinner("Generating data..."):
            if function_type == 'Sine':
                X = np.random.uniform(0, 2*np.pi, (n_samples, n_features))
                if n_features == 1:
                    y = np.sin(X[:, 0]) + 0.1 * X[:, 0]
                else:
                    y = np.sin(X[:, 0]) + np.cos(X[:, 1])

            elif function_type == 'Linear':
                X = np.random.uniform(0, 10, (n_samples, n_features))
                if n_features == 1:
                    y = -2 * X[:, 0] + 5
                else:
                    y = -2 * X[:, 0] + 3 * X[:, 1] + 5

            elif function_type == 'Nonlinear':
                X = np.random.uniform(-5, 5, (n_samples, n_features))
                if n_features == 1:
                    y = X[:, 0]**2 - 2 * X[:, 0]
                else:
                    y = X[:, 0]**2 + X[:, 1]**2 - X[:, 0] * X[:, 1]

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
            st.session_state.wm_dataset = df
            st.session_state.wm_dataset_name = f"Synthetic ({function_type})"

            st.success(f"✅ Generated {n_samples} samples × {n_features} features")
            st.balloons()
            st.rerun()

    except Exception as e:
        st.error(f"❌ Error generating dataset: {str(e)}")


def render_dataset_info():
    """Display dataset information"""

    df = st.session_state.get('wm_dataset')
    if df is None:
        return
    dataset_name = st.session_state.get('wm_dataset_name', 'Dataset')

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
            function_type = dataset_name.replace('Synthetic (', '').replace(')', '')

            st.markdown("**📐 Generation Formula**")
            st.markdown("")

            if function_type == 'Sine':
                st.latex(r"y = \sin(x) + 0.1x + \epsilon")
                st.caption(r"where $\epsilon \sim \mathcal{N}(0, \sigma^2)$ - Trigonometric function")

            elif function_type == 'Linear':
                st.latex(r"y = -2x + 5 + \epsilon")
                st.caption(r"where $\epsilon \sim \mathcal{N}(0, \sigma^2)$ - Linear function")

            elif function_type == 'Nonlinear':
                st.latex(r"y = x^2 - 2x + \epsilon")
                st.caption(r"where $\epsilon \sim \mathcal{N}(0, \sigma^2)$ - Quadratic function")

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

        # Dataset split
        st.markdown("#### Dataset Split")
        render_dataset_split()

        st.markdown("---")

        # Data normalization
        st.markdown("#### Normalization")
        render_data_normalization()


def render_dataset_split():
    """Render dataset splitting configuration"""

    df = st.session_state.get('wm_dataset')
    if df is None:
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        target_col = st.selectbox(
            "Target column",
            df.columns.tolist(),
            index=len(df.columns) - 1,
            key='wm_target_column'
        )

        feature_cols = st.multiselect(
            "Feature columns",
            [col for col in df.columns if col != target_col],
            default=[col for col in df.columns if col != target_col],
            key='wm_feature_columns'
        )

    with col2:
        st.markdown("**Split ratios (%)**")

        train_size = st.slider("Train", 50, 90, 70, 5, key='wm_split_train')

        remaining = 100 - train_size
        val_size = st.slider("Val", 0, remaining, min(20, remaining), 5, key='wm_split_val')

        test_size = remaining - val_size
        st.metric("Test", f"{test_size}%")

    if len(feature_cols) == 0:
        st.warning("Please select at least one feature column")
        return

    if st.button("Split Dataset", type="primary", width="stretch", key='wm_split_button'):
        split_dataset(df, feature_cols, target_col, train_size, val_size, test_size)

    msg = ""
    if st.session_state.get('wm_X_train', None) is not None:
        tr_size = len(st.session_state.wm_X_train)
        msg += f"Train: {tr_size}"
        if st.session_state.get('wm_X_val', None) is not None:
            val_size = len(st.session_state.wm_X_val)
            msg += f" | Val: {val_size}"
        if st.session_state.get('wm_X_test', None) is not None:
            test_size = len(st.session_state.wm_X_test)
            msg += f" | Test: {test_size}"
        st.success("**Split completed!** " + msg)


def split_dataset(df, feature_cols, target_col, train_pct, val_pct, test_pct):
    """Split dataset into train/val/test sets"""

    try:
        X = df[feature_cols].values
        y = df[target_col].values

        # Check if classification (integer labels) or regression
        if np.issubdtype(y.dtype, np.integer) or len(np.unique(y)) < 20:
            task = 'classification'
            # For classification, apply one-hot encoding
            from sklearn.preprocessing import OneHotEncoder
            encoder = OneHotEncoder(sparse_output=False)
            y = encoder.fit_transform(y.reshape(-1, 1))
            # Store encoder for later use
            st.session_state.wm_encoder = encoder
        else:
            task = 'regression'
            # For regression, ensure 2D column vector
            y = y.reshape(-1, 1)
            st.session_state.wm_encoder = None

        st.session_state.wm_feature_names = feature_cols
        st.session_state.wm_target_name = target_col
        st.session_state.wm_task = task

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
        st.session_state.wm_X_train = X_train
        st.session_state.wm_y_train = y_train
        st.session_state.wm_X_val = X_val
        st.session_state.wm_y_val = y_val
        st.session_state.wm_X_test = X_test
        st.session_state.wm_y_test = y_test

        # Reset scalers
        st.session_state.wm_scaler_X = None
        st.session_state.wm_scaler_y = None

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

    if st.session_state.get('wm_X_train', None) is None:
        st.info("Split the dataset first to enable normalization")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Features (X)**")

        normalize_X = st.checkbox(
            "Normalize features",
            value=False,
            key='wm_normalize_x'
        )

        if normalize_X:
            scaler_type_X = st.selectbox(
                "Scaler",
                ["StandardScaler", "MinMaxScaler"],
                key='wm_scaler_type_x'
            )

    with col2:
        st.markdown("**Target (y)**")

        # Only show normalization for regression
        task = st.session_state.get('wm_task', 'regression')
        if task == 'regression':
            normalize_y = st.checkbox(
                "Normalize target",
                value=False,
                key='wm_normalize_y'
            )

            if normalize_y:
                scaler_type_y = st.selectbox(
                    "Scaler",
                    ["StandardScaler", "MinMaxScaler"],
                    key='wm_scaler_type_y'
                )
        else:
            st.info("Target normalization not needed for classification")
            normalize_y = False
            scaler_type_y = None

    if st.button("Apply Normalization", type="secondary", width="stretch", key='wm_preprocess_button'):
        apply_preprocessing(normalize_X, normalize_y,
                          scaler_type_X if normalize_X else None,
                          scaler_type_y if normalize_y else None)

    if st.session_state.get('wm_scaler_X', None) is not None:
        msg = "**Normalization applied:**"
        msg += f" X ({st.session_state.get('wm_scaler_type_x', 'unknown')})"
        if st.session_state.get('wm_scaler_y', None) is not None:
            msg += f", y ({st.session_state.get('wm_scaler_type_y', 'unknown')})"
        st.success(msg)


def apply_preprocessing(normalize_X, normalize_y, scaler_type_X, scaler_type_y):
    """Apply preprocessing to dataset"""

    try:
        if normalize_X:
            scaler_X = StandardScaler() if scaler_type_X == "StandardScaler" else MinMaxScaler()

            st.session_state.wm_X_train = scaler_X.fit_transform(st.session_state.wm_X_train)

            if st.session_state.get('wm_X_val', None) is not None:
                st.session_state.wm_X_val = scaler_X.transform(st.session_state.wm_X_val)

            if st.session_state.get('wm_X_test', None) is not None:
                st.session_state.wm_X_test = scaler_X.transform(st.session_state.wm_X_test)

            st.session_state.wm_scaler_X = scaler_X

        if normalize_y and st.session_state.get('wm_task', 'regression') == 'regression':
            scaler_y = StandardScaler() if scaler_type_y == "StandardScaler" else MinMaxScaler()

            st.session_state.wm_y_train = scaler_y.fit_transform(st.session_state.wm_y_train)

            if st.session_state.get('wm_y_val', None) is not None:
                st.session_state.wm_y_val = scaler_y.transform(st.session_state.wm_y_val)

            if st.session_state.get('wm_y_test', None) is not None:
                st.session_state.wm_y_test = scaler_y.transform(st.session_state.wm_y_test)

            st.session_state.wm_scaler_y = scaler_y

        st.rerun()

    except Exception as e:
        st.error(f"Error applying normalization: {str(e)}")
