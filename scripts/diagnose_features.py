import logging
from typing import Dict
import numpy as np
import pandas as pd
from ..services.adaptive_risk_assessment import AdaptiveRiskAssessment

logger = logging.getLogger(__name__)

def diagnose_feature_mismatch(risk_assessment: AdaptiveRiskAssessment, request_text: str, request_type: str) -> Dict:
    """Diagnose feature extraction issues"""
    
    try:
        # Get feature names
        feature_names = risk_assessment.feature_names
        
        # Extract features
        features = risk_assessment.feature_pipeline.transform(request_text, request_type)
        
        # Analyze features
        diagnosis = {
            "total_features": len(feature_names),
            "features_extracted": features.shape[1],
            "non_zero_features": np.count_nonzero(features),
            "feature_statistics": {
                "mean": float(np.mean(features)),
                "std": float(np.std(features)),
                "min": float(np.min(features)),
                "max": float(np.max(features))
            },
            "top_active_features": []
        }
        
        # Get top active features
        non_zero_indices = np.nonzero(features[0])[0]
        for idx in non_zero_indices:
            if idx < len(feature_names):
                diagnosis["top_active_features"].append({
                    "name": feature_names[idx],
                    "value": float(features[0][idx])
                })
        
        return diagnosis
        
    except Exception as e:
        logger.error(f"Feature diagnosis failed: {e}")
        return {"error": str(e)} 