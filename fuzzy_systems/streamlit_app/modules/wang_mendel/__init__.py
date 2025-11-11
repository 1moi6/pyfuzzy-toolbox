"""Wang-Mendel Learning Module for pyfuzzy-toolbox Streamlit Interface."""

from . import dataset_tab
from . import training_tab
from . import metrics_tab
from . import prediction_tab
from . import analysis_tab
from . import overview_tab

__all__ = [
    'dataset_tab',
    'training_tab',
    'metrics_tab',
    'prediction_tab',
    'analysis_tab',
    'overview_tab',
]
