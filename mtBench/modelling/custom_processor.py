from hibayes.analysis import AnalysisState
from hibayes.process import process
from hibayes.ui import ModellingDisplay
import jax.numpy as jnp

@process
def add_pairwise_model_dims(
    model_feature: str = "left_model",
    group_features: list[str] | None = None,
):
    """Add dimensions for pairwise model comparison variables."""
    group_features = group_features or []
    
    def _process(
        state: AnalysisState,
        display: ModellingDisplay | None = None,
    ) -> AnalysisState:
        if not state.dims:
            state.dims = {}
        
        state.dims['model_mean'] = [model_feature]
        
        for group in group_features:
            effects_name = f"{group}_{model_feature}_effects"
            state.dims[effects_name] = [group, model_feature]
        
        # Grader × model effects
        state.dims['grader_model_shift'] = ['grader', model_feature]
        state.dims['grader_model_shift_raw'] = ['grader', model_feature]
        
        # Grader × continuous effects
        state.dims['grader_position_numeric_effects'] = ['grader']
        state.dims['grader_length_diff_prop_effects'] = ['grader']
        state.dims['grader_length_diff_prop_squared_effects'] = ['grader']  
        state.dims['grader_position_numeric_effects_raw'] = ['grader']
        state.dims['grader_length_diff_prop_effects_raw'] = ['grader']
        state.dims['grader_length_diff_prop_squared_effects_raw'] = ['grader'] 
        
        if display:
            display.logger.info(f"Added dims for model and grader effects")
        
        return state
    
    return _process

@process
def add_squared_column(
    feature: str,
):
    """
    Add squared version of a feature to the raw data BEFORE extract_features.
    
    Args:
        feature: Name of the column to square (e.g., 'length_diff_prop')
    """
    def _process(
        state: AnalysisState,
        display: ModellingDisplay | None = None,
    ) -> AnalysisState:
        squared_name = f"{feature}_squared"
        
        # Check which dataframe has the data
        if state.processed_data is not None and feature in state.processed_data.columns:
            # If processed_data exists, add to it
            state.processed_data[squared_name] = state.processed_data[feature] ** 2
            df_name = "processed_data"
        elif feature in state.data.columns:
            # Otherwise add to raw data
            state.data[squared_name] = state.data[feature] ** 2
            df_name = "data"
        else:
            if display:
                display.logger.warning(f"Feature '{feature}' not found in data or processed_data")
            return state
        
        if display:
            df = state.processed_data if df_name == "processed_data" else state.data
            display.logger.info(
                f"Added squared column '{squared_name}' to {df_name} "
                f"(range: [{df[squared_name].min():.3f}, "
                f"{df[squared_name].max():.3f}])"
            )
        
        return state
    
    return _process