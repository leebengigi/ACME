import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Suppress sklearn warnings
warnings.filterwarnings('ignore')

# Ensure project root is on path for local imports
project_root = Path(__file__).resolve().parent.parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(project_root))  # noqa: E402

from src.models.data_models import SecurityRequest  # noqa: E402
from src.models.enums import Outcome  # noqa: E402

logger = logging.getLogger(__name__)


class AdaptiveRiskAssessment:
    """
    Adaptive risk assessment engine that trains on historical data and predicts risk scores.

    Attributes:
        risk_model: Trained regression model for risk prediction
        feature_names: List of feature names extracted
        learned_thresholds: Thresholds for approval, rejection, and confidence
    """

    def __init__(self) -> None:
        self.risk_model: Optional[RandomForestRegressor] = None
        self.feature_names: List[str] = []
        self.learned_thresholds: Dict[str, float] = {
            'approval_threshold': 6.5,
            'rejection_threshold': 8.8,
            'confidence_threshold': 1.0,
        }
        self._is_trained: bool = False

    def train(self, historical_data: pd.DataFrame) -> None:
        """
        Train the risk assessment model using historical security data.

        Args:
            historical_data: DataFrame with 'request_summary' and 'security_risk_score' columns
        """
        logger.info("Training AdaptiveRiskAssessment...")

        if 'security_risk_score' not in historical_data:
            logger.error("Missing 'security_risk_score' column in data")
            return

        data = historical_data.dropna(subset=['request_summary', 'security_risk_score'])
        if data.empty:
            logger.error("No valid training data found after cleaning")
            return

        features = self._extract_features(data)
        if features.size == 0:
            logger.error("Feature extraction resulted in empty array")
            return

        self._train_model(features, data['security_risk_score'])
        self._learn_thresholds(data)

        self._is_trained = True
        logger.info("AdaptiveRiskAssessment training completed successfully")

    def assess_risk(
        self,
        request_text: str,
        request_type: str,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Assess the risk of a single security request.

        Args:
            request_text: The text of the security request
            request_type: The type/category of the request
            additional_context: Optional context data

        Returns:
            Tuple of (risk_score, info_dict)
        """
        if not self._is_trained or self.risk_model is None:
            return self._fallback_assessment(request_text)

        feature_vector = self._extract_single_features(request_text, request_type)
        try:
            score = float(self.risk_model.predict([feature_vector])[0])
            score = max(1.0, min(10.0, score))
            info = {
                'method': 'ml_prediction',
                'confidence': 1.0,
                'explanation': 'Using trained ML model',
            }
            return score, info
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return self._fallback_assessment(request_text)

    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract numeric features from historical data.
        """
        try:
            rows: List[List[float]] = []
            for _, row in data.iterrows():
                text = str(row['request_summary']).lower()
                req_type = str(row.get('request_type', 'other')).lower()
                rows.append([
                    len(text),
                    len(text.split()),
                    text.count('admin'),
                    text.count('production'),
                    text.count('database'),
                    text.count('access'),
                    text.count('urgent'),
                    1 if 'permission' in req_type else 0,
                    1 if 'data' in req_type else 0,
                    1 if 'network' in req_type else 0,
                ])
            features = np.array(rows)
            self.feature_names = [
                'text_length', 'word_count', 'admin_count', 'production_count',
                'database_count', 'access_count', 'urgent_count', 'is_permission',
                'is_data', 'is_network',
            ]
            logger.info(f"Extracted {features.shape[1]} features")
            return features
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return np.empty((0, 0))

    def _extract_single_features(self, text: str, request_type: str) -> np.ndarray:
        """
        Extract features for a single request.
        """
        text_lower = text.lower()
        req_type = request_type.lower()
        return np.array([
            len(text_lower),
            len(text_lower.split()),
            text_lower.count('admin'),
            text_lower.count('production'),
            text_lower.count('database'),
            text_lower.count('access'),
            text_lower.count('urgent'),
            1 if 'permission' in req_type else 0,
            1 if 'data' in req_type else 0,
            1 if 'network' in req_type else 0,
        ])

    def _train_model(self, features: np.ndarray, targets: pd.Series) -> None:
        """
        Train the RandomForest risk model.
        """
        try:
            valid = (~np.isnan(targets)) & (~np.isinf(targets))
            X = features[valid]
            y = targets[valid]
            if X.size == 0:
                raise ValueError("No valid samples for training")

            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
            )
            model.fit(X, y)
            self.risk_model = model
            logger.info(f"Trained risk model on {len(y)} samples")
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            self._setup_fallback()

    def _learn_thresholds(self, data: pd.DataFrame) -> None:
        """
        Learn approval/rejection thresholds from historical outcomes.
        """
        try:
            if 'outcome' not in data:
                logger.warning("No 'outcome' column for threshold learning")
                return
            groups = data.groupby('outcome')['security_risk_score']
            approved = groups.get_group('Approved') if 'Approved' in groups.groups else pd.Series([])
            rejected = groups.get_group('Rejected') if 'Rejected' in groups.groups else pd.Series([])
            if not approved.empty and not rejected.empty:
                self.learned_thresholds['approval_threshold'] = float(np.percentile(approved, 75))
                self.learned_thresholds['rejection_threshold'] = float(np.percentile(rejected, 25))
                self.learned_thresholds['confidence_threshold'] = 0.7
                logger.info(f"Learned thresholds: {self.learned_thresholds}")
        except Exception as e:
            logger.error(f"Threshold learning failed: {e}")

    def _fallback_assessment(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """
        Simple keyword-based fallback risk assessment.
        """
        text_lower = text.lower()
        score = 5.0
        high_indicators = [
            ('admin', 2.0), ('root', 2.5), ('production', 1.5), ('database', 1.5),
            ('urgent', 0.5), ('critical', 1.0)
        ]
        low_indicators = [
            ('read-only', -1.5), ('test', -0.5), ('staging', -0.5)
        ]
        for keyword, weight in high_indicators:
            if keyword in text_lower:
                score += weight
        for keyword, weight in low_indicators:
            if keyword in text_lower:
                score += weight
        score = max(1.0, min(10.0, score))
        info = {'method': 'keyword_fallback', 'confidence': 0.5}
        return score, info

    def _setup_fallback(self) -> None:
        """
        Configure fallback mode when training or prediction fails.
        """
        self._is_trained = True
        self.risk_model = None
        logger.info("Fallback risk assessment configured")
