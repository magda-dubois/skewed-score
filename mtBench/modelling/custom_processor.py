from hibayes.analysis import AnalysisState
from hibayes.process import process
from hibayes.ui import ModellingDisplay

@process
def add_pairwise_model_dims(
    model_feature: str = "left_model",
    group_features: list[str] | None = None,
):
    """
    Add dimensions for pairwise model comparison variables.
    
    Args:
        model_feature: The categorical feature representing models (e.g., 'left_model')
        group_features: List of grouping variables that have model effects (e.g., ['question_id'])
    """
    group_features = group_features or []
    
    def process(
        state: AnalysisState,
        display: ModellingDisplay | None = None,
    ) -> AnalysisState:
        if not state.dims:
            state.dims = {}
        
        # Add dim for model_mean (marginal model effects)
        state.dims['model_mean'] = [model_feature]
        
        # Add dims for hierarchical model effects (e.g., question_id × left_model)
        for group in group_features:
            effects_name = f"{group}_{model_feature}_effects"
            state.dims[effects_name] = [group, model_feature]
            
            if display:
                display.logger.info(
                    f"Added dims for '{effects_name}': [{group}, {model_feature}]"
                )
        
        if display:
            display.logger.info(f"Added dims for 'model_mean': [{model_feature}]")
        
        return state
    
    return process