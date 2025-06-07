"""
Adaptive Decision Engine for ACME Security Bot.

This module implements a sophisticated decision-making system that combines
machine learning, statistical analysis, and rule-based approaches to make
security-related decisions. The engine uses an ensemble of models and adapts
its decision-making process based on historical data and context.
"""
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')

from src.models.enums import Outcome, RequestType
from src.models.data_models import SecurityRequest

logger = logging.getLogger(__name__)


class AdaptiveDecisionEngine:
    """
    Adaptive Decision Engine for security request evaluation.
    
    This engine combines multiple approaches to make security decisions:
    1. Machine learning ensemble (Random Forest, Gradient Boosting, etc.)
    2. Feature-based analysis (text, risk scores, request types)
    3. Safety rules and critical violations
    4. Fallback decision logic
    
    The engine learns from historical data and adapts its decision-making
    process to maintain high accuracy and reliability.
    """
    
    def __init__(self):
        """
        Initialize the adaptive decision engine with default configurations.
        
        Sets up model components, feature extractors, and safety rules.
        The engine starts with default configurations and learns optimal
        parameters during training.
        """
        self.risk_assessment = None
        self.learned_boundaries: dict = {}
        self.decision_patterns: dict = {}
        self.context_specific_thresholds: dict = {}
        self.confidence_handling_rules: dict = {}
        self.training_samples: int = 0
        self.decision_accuracy: float = 0.0
        
        # ML model components
        self._is_trained = False
        self.models = {}
        self.model_scores = {}
        self.best_model_name = None
        self.vectorizer = None
        self.scaler = None
        self.label_encoder = None
        self.type_encoder = None
        
        # Feature extraction components
        self.feature_names = []
        
        # Critical security violations that trigger immediate rejection
        self.critical_violations = [
            'bypass security', 'disable security', 'override security',
            'unrestricted access', 'full access to all', 'admin to everything',
            'root access to production', 'delete all data'
        ]

    def set_risk_assessment(self, risk_assessment):
        """
        Set the risk assessment service for the decision engine.
        
        Args:
            risk_assessment: The risk assessment service instance to use
                           for evaluating security risks.
        """
        self.risk_assessment = risk_assessment
        logger.debug("Risk assessment service linked to decision engine")

    def train(self, historical_data: pd.DataFrame) -> Dict:
        """
        Train the decision engine using historical data.
        
        This method trains multiple ML models and selects the best performing one
        based on validation accuracy and cross-validation stability.
        
        Args:
            historical_data (pd.DataFrame): DataFrame containing historical
                                          security requests and their outcomes.
        
        Returns:
            Dict: Training results including model performance metrics and
                 feature information.
        """
        logger.info("🚀 Training Ensemble Decision Engine...")
        
        try:
            # Validate and prepare training data
            X, y, feature_info = self._prepare_training_data(historical_data)
            if X is None:
                logger.error("Failed to prepare training data")
                return {'error': 'Data preparation failed'}
            
            # Split data for validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train multiple models and select the best one
            model_results = self._train_multiple_models(X_train, X_val, y_train, y_val)
            best_result = self._select_best_model(model_results)
            
            # Perform final validation
            final_score = self._final_validation(X_val, y_val)
            
            self._is_trained = True
            self.training_samples = len(historical_data)
            self.decision_accuracy = final_score
            
            logger.info(f"✅ Ensemble training complete! Best model: {self.best_model_name} ({final_score:.1%} accuracy)")
            
            return {
                'best_model': self.best_model_name,
                'best_accuracy': final_score,
                'all_model_scores': self.model_scores,
                'training_samples': self.training_samples,
                'feature_count': X.shape[1],
                'feature_info': feature_info
            }
            
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
            return {'error': str(e)}

    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
        """
        Prepare comprehensive training data for model training.
        
        This method handles data cleaning, feature extraction, and target encoding.
        It ensures the data is properly formatted for ML model training.
        
        Args:
            df (pd.DataFrame): Raw training data.
            
        Returns:
            Tuple containing:
            - Features array (or None if preparation failed)
            - Target array (or None if preparation failed)
            - Feature information dictionary
        """
        # Validate required columns
        required_cols = ['request_summary', 'security_risk_score', 'outcome']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return None, None, {}
        
        # Clean and normalize data
        df_clean = df.dropna(subset=required_cols).copy()
        df_clean = df_clean[df_clean['outcome'].isin(['Approved', 'Rejected', 'Info Requested'])]
        df_clean['outcome'] = df_clean['outcome'].replace('Info Requested', 'Needs More Info')
        
        if len(df_clean) < 50:
            logger.error(f"Insufficient training data: {len(df_clean)} samples")
            return None, None, {}
        
        logger.info(f"Cleaned training data: {len(df_clean)} samples")
        logger.info(f"Outcome distribution: {df_clean['outcome'].value_counts().to_dict()}")
        
        # Extract features and encode targets
        features = self._extract_comprehensive_features(df_clean)
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df_clean['outcome'])
        
        feature_info = {
            'feature_count': features.shape[1],
            'sample_count': len(df_clean),
            'outcome_classes': list(self.label_encoder.classes_),
            'feature_types': {
                'text_features': 'TF-IDF vectors',
                'risk_score': 'Normalized risk score',
                'request_type': 'Encoded request type',
                'text_stats': 'Length, word count, etc.'
            }
        }
        
        return features, y, feature_info

    def _extract_comprehensive_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract comprehensive features from the request data.
        
        This method combines multiple feature types:
        1. Text features using TF-IDF
        2. Normalized risk scores
        3. Encoded request types
        4. Text statistics and indicators
        
        Args:
            df (pd.DataFrame): Cleaned training data.
            
        Returns:
            np.ndarray: Combined feature matrix.
        """
        # Extract text features using TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            lowercase=True
        )
        text_features = self.vectorizer.fit_transform(df['request_summary'].fillna(''))
        
        # Normalize risk scores
        self.scaler = StandardScaler()
        risk_scores = self.scaler.fit_transform(df[['security_risk_score']])
        
        # Encode request types
        type_features = np.zeros((len(df), 1))
        if 'request_type' in df.columns:
            self.type_encoder = LabelEncoder()
            df_types = df['request_type'].fillna('other').astype(str)
            type_features = self.type_encoder.fit_transform(df_types).reshape(-1, 1)
        
        # Extract text statistics and indicators
        text_stats = []
        for text in df['request_summary'].fillna(''):
            text_lower = text.lower()
            stats = [
                len(text),  # Length
                len(text.split()),  # Word count
                text.count('!'),  # Urgency indicators
                text.count('?'),  # Questions
                sum(1 for c in text if c.isupper()) / max(len(text), 1),  # Caps ratio
                # Security keywords
                text_lower.count('admin'),
                text_lower.count('production'),
                text_lower.count('database'),
                text_lower.count('urgent'),
                text_lower.count('access'),
                # Risk indicators
                1 if any(word in text_lower for word in ['critical', 'emergency', 'asap']) else 0,
                1 if any(word in text_lower for word in ['read-only', 'readonly', 'view']) else 0,
            ]
            text_stats.append(stats)
        
        text_stats = np.array(text_stats)
        
        # Combine all features
        all_features = hstack([
            text_features,  # TF-IDF (1000 features)
            risk_scores,    # Risk score (1 feature)
            type_features,  # Request type (1 feature)
            text_stats      # Text statistics (12 features)
        ])
        
        logger.info(f"Extracted {all_features.shape[1]} total features")
        return all_features

    def _train_multiple_models(self, X_train, X_val, y_train, y_val) -> Dict:
        """
        Train multiple ML models and evaluate their performance.
        
        This method trains various models including Random Forest, Gradient Boosting,
        Logistic Regression, SVM, and Naive Bayes. It evaluates each model using
        validation accuracy and cross-validation scores.
        
        Args:
            X_train: Training features
            X_val: Validation features
            y_train: Training targets
            y_val: Validation targets
            
        Returns:
            Dict: Results for each trained model including performance metrics
        """
        # Define models to train with their configurations
        models_to_try = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42,
                C=1.0
            ),
            'svm': SVC(
                class_weight='balanced',
                probability=True,
                random_state=42,
                C=1.0,
                gamma='scale'
            ),
            'naive_bayes': GaussianNB()
        }
        
        results = {}
        
        # Train and evaluate each model
        for name, model in models_to_try.items():
            try:
                logger.info(f"Training {name}...")
                
                # Handle special case for Naive Bayes
                if name == 'naive_bayes':
                    model.fit(X_train.toarray(), y_train)
                    val_pred = model.predict(X_val.toarray())
                    val_proba = model.predict_proba(X_val.toarray())
                else:
                    model.fit(X_train, y_train)
                    val_pred = model.predict(X_val)
                    val_proba = model.predict_proba(X_val)
                
                # Calculate performance metrics
                val_accuracy = accuracy_score(y_val, val_pred)
                
                # Cross-validation scores
                if name == 'naive_bayes':
                    cv_scores = cross_val_score(model, X_train.toarray(), y_train, cv=3, scoring='accuracy')
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                # Store model and results
                self.models[name] = model
                results[name] = {
                    'validation_accuracy': val_accuracy,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'predictions': val_pred,
                    'probabilities': val_proba
                }
                
                logger.info(f"  {name}: Val={val_accuracy:.3f}, CV={cv_mean:.3f}±{cv_std:.3f}")
                
            except Exception as e:
                logger.warning(f"Failed to train {name}: {e}")
                continue
        
        return results

    def _select_best_model(self, results: Dict) -> Dict:
        """
        Select the best performing model based on combined metrics.
        
        The selection process considers:
        1. Validation accuracy (70% weight)
        2. Cross-validation mean (30% weight)
        3. Cross-validation stability (10% penalty)
        
        Args:
            results (Dict): Results from training multiple models.
            
        Returns:
            Dict: Results for the best performing model.
            
        Raises:
            ValueError: If no models were trained successfully.
        """
        if not results:
            raise ValueError("No models trained successfully")
        
        # Calculate combined scores for each model
        model_rankings = []
        for name, result in results.items():
            score = result['validation_accuracy']
            cv_score = result['cv_mean']
            cv_std = result['cv_std']
            
            # Combined score with weights
            combined_score = score * 0.7 + cv_score * 0.3 - cv_std * 0.1
            
            model_rankings.append((name, score, cv_score, combined_score))
            self.model_scores[name] = {
                'validation_accuracy': score,
                'cv_mean': cv_score,
                'cv_std': cv_std,
                'combined_score': combined_score
            }
        
        # Sort and select best model
        model_rankings.sort(key=lambda x: x[3], reverse=True)
        self.best_model_name = model_rankings[0][0]
        best_result = results[self.best_model_name]
        
        # Log rankings
        logger.info("🏆 Model Rankings:")
        for i, (name, val_acc, cv_acc, combined) in enumerate(model_rankings[:3]):
            rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            logger.info(f"  {rank_emoji} {name}: {val_acc:.3f} (combined: {combined:.3f})")
        
        return best_result

    def _final_validation(self, X_val, y_val) -> float:
        """
        Perform final validation of the best model.
        
        This method evaluates the selected model on the validation set and
        provides detailed performance metrics for each outcome class.
        
        Args:
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            float: Final accuracy score
        """
        best_model = self.models[self.best_model_name]
        
        # Get predictions
        if self.best_model_name == 'naive_bayes':
            y_pred = best_model.predict(X_val.toarray())
        else:
            y_pred = best_model.predict(X_val)
        
        accuracy = accuracy_score(y_val, y_pred)
        
        # Generate detailed performance report
        class_names = self.label_encoder.classes_
        report = classification_report(y_val, y_pred, target_names=class_names, output_dict=True)
        
        logger.info("📊 Final Model Performance:")
        logger.info(f"  Overall Accuracy: {accuracy:.3f}")
        for class_name in class_names:
            if class_name in report:
                precision = report[class_name]['precision']
                recall = report[class_name]['recall']
                f1 = report[class_name]['f1-score']
                logger.info(f"  {class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
        
        return accuracy

    def make_decision(self, request: SecurityRequest) -> Tuple[Outcome, str]:
        """
        Make a security decision for a given request.
        
        This method combines multiple approaches:
        1. Safety rules (highest priority)
        2. ML model prediction (if trained)
        3. Fallback decision logic
        
        Args:
            request (SecurityRequest): The security request to evaluate.
            
        Returns:
            Tuple[Outcome, str]: Decision outcome and explanation.
        """
        # Apply safety rules first
        safety_result = self._apply_safety_rules(request)
        if safety_result:
            return safety_result
        
        if not self._is_trained or not self.best_model_name:
            return self._fallback_decision(request)
        
        try:
            # Extract features and get ML prediction
            features = self._extract_single_request_features(request)
            best_model = self.models[self.best_model_name]
            
            if self.best_model_name == 'naive_bayes':
                prediction = best_model.predict(features.toarray())[0]
                probabilities = best_model.predict_proba(features.toarray())[0]
            else:
                prediction = best_model.predict(features)[0]
                probabilities = best_model.predict_proba(features)[0]
            
            # Convert prediction to outcome
            outcome_str = self.label_encoder.inverse_transform([prediction])[0]
            outcome = Outcome(outcome_str)
            
            # Get confidence and create rationale
            confidence = float(probabilities[prediction])
            risk_score = request.risk_score or 50.0
            rationale = f"ML DECISION ({self.best_model_name}): {outcome.value} (confidence: {confidence:.1%}, risk: {risk_score:.1f})"
            
            # Add warning for low confidence
            if confidence < 0.6:
                rationale += " [LOW CONFIDENCE - Consider manual review]"
            
            logger.debug(f"Decision: {outcome.value} with {confidence:.1%} confidence")
            return outcome, rationale
            
        except Exception as e:
            logger.error(f"ML decision failed: {e}")
            return self._fallback_decision(request)

    def _extract_single_request_features(self, request: SecurityRequest) -> np.ndarray:
        """
        Extract features for a single security request.
        
        This method processes a single request to create the same feature
        set used during training, including text features, risk scores,
        request type, and text statistics.
        
        Args:
            request (SecurityRequest): The request to process.
            
        Returns:
            np.ndarray: Feature vector for the request.
        """
        # Create temporary dataframe
        temp_df = pd.DataFrame({
            'request_summary': [request.request_text],
            'security_risk_score': [request.risk_score or 50.0],
            'request_type': [request.request_type.value if request.request_type else 'other']
        })
        
        # Extract text features
        text_features = self.vectorizer.transform(temp_df['request_summary'])
        
        # Process risk score
        risk_features = self.scaler.transform(temp_df[['security_risk_score']])
        
        # Encode request type
        type_features = np.zeros((1, 1))
        if self.type_encoder and request.request_type:
            try:
                type_features = self.type_encoder.transform([request.request_type.value]).reshape(-1, 1)
            except ValueError:
                # Unknown type, use 0
                pass
        
        # Extract text statistics
        text = request.request_text.lower()
        text_stats = np.array([[
            len(request.request_text),
            len(request.request_text.split()),
            request.request_text.count('!'),
            request.request_text.count('?'),
            sum(1 for c in request.request_text if c.isupper()) / max(len(request.request_text), 1),
            text.count('admin'),
            text.count('production'),
            text.count('database'),
            text.count('urgent'),
            text.count('access'),
            1 if any(word in text for word in ['critical', 'emergency', 'asap']) else 0,
            1 if any(word in text for word in ['read-only', 'readonly', 'view']) else 0,
        ]])
        
        # Combine all features
        features = hstack([text_features, risk_features, type_features, text_stats])
        return features

    def _apply_safety_rules(self, request: SecurityRequest) -> Optional[Tuple[Outcome, str]]:
        """
        Apply safety rules to the request.
        
        These rules take precedence over ML predictions and include:
        1. Critical security violations
        2. Extreme risk scores
        
        Args:
            request (SecurityRequest): The request to evaluate.
            
        Returns:
            Optional[Tuple[Outcome, str]]: Decision and explanation if safety rule applies,
                                         None otherwise.
        """
        text = request.request_text.lower()
        risk_score = request.risk_score or 50.0
        
        # Check for critical security violations
        for phrase in self.critical_violations:
            if phrase in text:
                return Outcome.REJECTED, f"SECURITY VIOLATION: '{phrase}' detected → automatic rejection"
        
        # Check extreme risk scores
        if risk_score >= 95.0:
            return Outcome.REJECTED, f"CRITICAL RISK: Score {risk_score:.1f}/100 ≥ 95 → automatic rejection"
        elif risk_score <= 25.0:
            return Outcome.APPROVED, f"MINIMAL RISK: Score {risk_score:.1f}/100 ≤ 25 → automatic approval"
        
        return None

    def _fallback_decision(self, request: SecurityRequest) -> Tuple[Outcome, str]:
        """
        Make a fallback decision when ML is not available.
        
        This method uses simple risk score thresholds to make decisions
        when the ML model is not trained or fails.
        
        Args:
            request (SecurityRequest): The request to evaluate.
            
        Returns:
            Tuple[Outcome, str]: Decision and explanation.
        """
        risk_score = request.risk_score or 50.0
        
        if risk_score <= 60.0:
            return Outcome.APPROVED, f"FALLBACK: Low risk ({risk_score:.1f}/100)"
        elif risk_score >= 85.0:
            return Outcome.REJECTED, f"FALLBACK: High risk ({risk_score:.1f}/100)"
        else:
            return Outcome.NEEDS_MORE_INFO, f"FALLBACK: Medium risk ({risk_score:.1f}/100)"

    def get_model_summary(self) -> Dict:
        """
        Get comprehensive model performance summary.
        
        Returns:
            Dict: Summary of model performance including:
                 - Best model name
                 - Accuracy
                 - Training samples
                 - All model scores
                 - Training status
        """
        if not self._is_trained:
            return {'error': 'Model not trained'}
        
        return {
            'best_model': self.best_model_name,
            'accuracy': self.decision_accuracy,
            'training_samples': self.training_samples,
            'all_models': self.model_scores,
            'is_trained': self._is_trained
        }