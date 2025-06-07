"""
Classification Service for ACME Security Bot.

This module provides a sophisticated classification system that combines rule-based,
machine learning, and LLM-based approaches to classify security-related requests.
The service is designed to be highly accurate and can fall back gracefully when
certain dependencies are not available.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import re
import logging
from typing import Tuple, Dict, List, Optional

# Add project root to path for local imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.enums import RequestType

logger = logging.getLogger(__name__)

# Import ML libraries with fallback
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import cross_val_score, train_test_split
    from scipy.sparse import hstack, csr_matrix
    from sklearn.metrics import accuracy_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("ML libraries not available - using rule-based classification only")

# Import LLM service with fallback
try:
    from src.services.LLM_services import LLMClassificationService
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger.warning("LLM service not available - using ML and rule-based classification only")

class ClassificationService:
    """    
    This service combines multiple classification approaches:
    1. Rule-based pattern matching
    2. Machine learning (when available)
    3. LLM-based classification (when available)
    
    The service automatically falls back to simpler methods if advanced
    dependencies are not available, ensuring continuous operation.
    """
    
    def __init__(self):
        """
        Initialize the classification service with all available classification methods.
        
        Sets up rule patterns, ML models (if available), and LLM classifier (if available).
        The service is designed to work with any combination of these methods.
        """
        self.rule_patterns = self._initialize_rule_patterns()
        self.vectorizer = None
        self.ensemble_models = {}
        self.label_encoder = None
        self._is_trained = False
        self.llm_classifier = None
        
        if LLM_AVAILABLE:
            try:
                # Initialize LLM classifier with optimized parameters for security classification
                self.llm_classifier = LLMClassificationService(
                    model_name="Qwen/Qwen3-0.6B",
                    max_length=512,
                    num_beams=4,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    no_repeat_ngram_size=3
                )
                logger.info("LLM classifier initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize LLM classifier: {e}")
                self.llm_classifier = None

        self.label_encoder = LabelEncoder()
        # Initialize with all RequestType values for consistent encoding
        all_types = [rt.value for rt in RequestType]
        self.label_encoder.fit(all_types)
    
    def _initialize_rule_patterns(self) -> Dict:
        """
        Initialize ACME-optimized rule patterns for request classification.
        
        Returns:
            Dict: A dictionary mapping RequestType to pattern configurations.
                 Each configuration includes regex patterns, keywords, and confidence scores.
        """
        return {
            RequestType.DEVTOOL_INSTALL: {
                'patterns': [
                    r'\b(install|setup|download)\s+(vscode|visual\s*studio|extension|docker|java|jdk|sdk|tool)\b',
                    r'\bsetup.*development\b',
                    r'\bdownload.*(tool|software|package)\b',
                    r'\b(java|python|node|npm|git)\s+(install|setup)\b'
                ],
                'keywords': ['install', 'setup', 'vscode', 'docker', 'development', 'java', 'jdk', 'sdk', 'tool'],
                'confidence': 0.95
            },
            
            RequestType.PERMISSION_CHANGE: {
                'patterns': [
                    r'\btemporary\s+admin\s+access\b',
                    r'\badmin\s+access\b',
                    r'\belevated\s+permission\b'
                ],
                'keywords': ['admin', 'access', 'permission', 'elevated', 'temporary'],
                'confidence': 0.94
            },
            
            RequestType.NETWORK_ACCESS: {
                'patterns': [
                    r'\bopen\s+port\s+\d+\b',
                    r'\bvpn\s+access\b',
                    r'\bnetwork\s+connectivity\b'
                ],
                'keywords': ['port', 'network', 'vpn', 'connection', 'firewall'],
                'confidence': 0.93
            },
            
            RequestType.FIREWALL_CHANGE: {
                'patterns': [
                    r'\ballow\s+ssh\b',
                    r'\bfirewall\s+rule\b',
                    r'\bsecurity\s+group\b'
                ],
                'keywords': ['firewall', 'allow', 'ssh', 'external', 'rule'],
                'confidence': 0.92
            },
            
            RequestType.DATA_EXPORT: {
                'patterns': [
                    r'\bexport.*data\b',
                    r'\bdatabase\s+access\b',
                    r'\bquery.*database\b'
                ],
                'keywords': ['data', 'export', 'database', 'query', 'analysis'],
                'confidence': 0.91
            },
            
            RequestType.CLOUD_RESOURCE_ACCESS: {
                'patterns': [
                    r'\b(aws|azure|gcp)\s+(instance|resource|vm|s3|bucket)\b',
                    r'\b(ec2|vm|container)\s+access\b',
                    r'\b(cloud|server)\s+(access|permission)\b',
                    r'\b(production|staging)\s+(server|environment)\b',
                    r'\b(azure|aws)\s+(virtual\s+machine|vm|instance)\b',
                    r'\b(s3|bucket|storage)\s+(access|permission)\b'
                ],
                'keywords': ['cloud', 'aws', 'azure', 'gcp', 'server', 'instance', 'ec2', 'vm', 's3', 'bucket'],
                'confidence': 0.90
            },
            
            RequestType.VENDOR_APPROVAL: {
                'patterns': [
                    r'\bonboard\s+vendor\b',
                    r'\bvendor\s+approval\b',
                    r'\bthird[-\s]?party\b'
                ],
                'keywords': ['vendor', 'contractor', 'onboard', 'approval'],
                'confidence': 0.89
            }
        }
    
    def _rule_based_classification(self, text: str) -> Tuple[Optional[RequestType], float]:
        """
        Perform high-precision rule-based classification of the request text.
        
        Args:
            text (str): The request text to classify.
            
        Returns:
            Tuple[Optional[RequestType], float]: A tuple containing the classified request type
                                               and confidence score. Returns (None, 0.0) if no match.
        """
        text_lower = text.lower()
        
        # Special case: Development tool installation with server access
        dev_tool_patterns = [
            r'\b(install|setup|download)\s+(docker|vscode|visual\s*studio|extension|java|jdk|sdk|tool)\b',
            r'\bsetup.*development\b',
            r'\bdownload.*(tool|software|package)\b',
            r'\b(java|python|node|npm|git)\s+(install|setup)\b'
        ]
        
        # Check for development tool installation with server access
        has_dev_tool = any(re.search(pattern, text_lower) for pattern in dev_tool_patterns)
        has_server_access = any(term in text_lower for term in ['server access', 'server permission', 'get server'])
        
        if has_dev_tool and has_server_access:
            # If it's primarily about installing a development tool, prioritize that
            dev_tool_score = sum(1 for pattern in dev_tool_patterns if re.search(pattern, text_lower))
            if dev_tool_score >= 2:
                return RequestType.DEVTOOL_INSTALL, 0.90
        
        # First check for cloud resource access as it should take precedence
        cloud_config = self.rule_patterns[RequestType.CLOUD_RESOURCE_ACCESS]
        cloud_score = 0.0
        
        # Calculate cloud resource access score
        for pattern in cloud_config['patterns']:
            if re.search(pattern, text_lower):
                cloud_score += 3.0
        
        cloud_keyword_matches = sum(1 for word in cloud_config['keywords'] if word in text_lower)
        cloud_score += cloud_keyword_matches * 1.0
        
        if cloud_score >= 2.0:
            cloud_confidence = min(cloud_config['confidence'], 0.5 + cloud_score * 0.1)
            if cloud_confidence > 0.8:
                return RequestType.CLOUD_RESOURCE_ACCESS, cloud_confidence
        
        # Check other categories with their respective patterns
        for request_type, config in self.rule_patterns.items():
            if request_type == RequestType.CLOUD_RESOURCE_ACCESS:
                continue  # Skip cloud resource access as we already checked it
                
            score = 0.0
            
            # Check regex patterns
            for pattern in config['patterns']:
                if re.search(pattern, text_lower):
                    score += 3.0
            
            # Check keywords
            keyword_matches = sum(1 for word in config['keywords'] if word in text_lower)
            score += keyword_matches * 1.0
            
            # Calculate confidence
            if score >= 2.0:
                confidence = min(config['confidence'], 0.5 + score * 0.1)
                if confidence > 0.8:
                    return request_type, confidence
        
        return None, 0.0
    
    def train(self, historical_data: pd.DataFrame) -> Dict:
        """Train the classification system"""
        logger.info("Training Enhanced Classification Service...")
        
        if LLM_AVAILABLE and self.llm_classifier:
            self.llm_classifier.train(historical_data)
        
        try:
            # Prepare data
            X_text = historical_data['request_summary'].astype(str)
            y = historical_data['request_type']
            
            # Map to RequestType values
            y_enum = []
            for label in y:
                # Normalize label
                normalized = label.lower().replace(' ', '_')
                
                # Find matching RequestType
                for req_type in RequestType:
                    if req_type.value == normalized:
                        y_enum.append(req_type.value)
                        break
                else:
                    y_enum.append(RequestType.OTHER.value)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_text, y_enum, test_size=0.2, random_state=42, stratify=y_enum
            )
            
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(y_enum)
            
            self.vectorizer = TfidfVectorizer(
                max_features=327,  # Match expected feature count
                ngram_range=(1, 2),
                stop_words='english',
                min_df=1,
                max_df=0.95
            )
            
            X_train_tfidf = self.vectorizer.fit_transform(X_train)
            

            X_test_tfidf = self.vectorizer.transform(X_test)
            
            if X_train_tfidf.shape[1] != 327:
                logger.warning(f"Feature count mismatch: got {X_train_tfidf.shape[1]}, expected 327. Padding features...")
                # Pad features if needed
                padding = csr_matrix((X_train_tfidf.shape[0], 327 - X_train_tfidf.shape[1]))
                X_train_tfidf = hstack([X_train_tfidf, padding])
                X_test_tfidf = hstack([X_test_tfidf, csr_matrix((X_test_tfidf.shape[0], 327 - X_test_tfidf.shape[1]))])
            
            # Train ensemble
            models = {
                'random_forest': RandomForestClassifier(
                    n_estimators=100,
                    class_weight='balanced',
                    random_state=42
                ),
                'logistic': LogisticRegression(
                    class_weight='balanced',
                    max_iter=1000,
                    random_state=42
                )
            }
            
            scores = {}
            test_scores = {}
            for name, model in models.items():
                try:
                    model.fit(X_train_tfidf, self.label_encoder.transform(y_train))
                    cv_scores = cross_val_score(model, X_train_tfidf, self.label_encoder.transform(y_train), cv=3)
                    scores[name] = cv_scores.mean()
                    y_pred = model.predict(X_test_tfidf)
                    test_acc = accuracy_score(self.label_encoder.transform(y_test), y_pred)
                    test_scores[name] = test_acc
                    self.ensemble_models[name] = model
                    logger.info(f"{name}: CV {cv_scores.mean():.3f}, Test {test_acc:.3f}")
                except Exception as e:
                    logger.warning(f"Failed to train {name}: {e}")
            
            self._is_trained = len(self.ensemble_models) > 0
            
            return {
                'ensemble_scores': scores,
                'test_scores': test_scores,
                'models_trained': len(self.ensemble_models),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'ml_available': True
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {'error': str(e), 'ml_available': True}
    
    def classify_request(self, request_text: str) -> Tuple[RequestType, float]:
        """Classify a security request"""
        
        # Step 1: Rule-based classification
        rule_result, rule_confidence = self._rule_based_classification(request_text)
        if rule_result is not None and rule_confidence > 0.85:
            return rule_result, rule_confidence
        
        # Step 2: ML classification (if available and trained)
        if self._is_trained and self.ensemble_models:
            try:
                X_tfidf = self.vectorizer.transform([request_text])
                
                # Ensure feature count matches
                if X_tfidf.shape[1] != 327:
                    logger.warning(f"Feature count mismatch in prediction: got {X_tfidf.shape[1]}, expected 327. Padding features...")
                    padding = csr_matrix((1, 327 - X_tfidf.shape[1]))
                    X_tfidf = hstack([X_tfidf, padding])
                
                predictions = {}
                confidences = {}
                
                for name, model in self.ensemble_models.items():
                    try:
                        pred_proba = model.predict_proba(X_tfidf)[0]
                        pred_class = np.argmax(pred_proba)
                        max_proba = pred_proba[pred_class]
                        
                        predicted_type = self.label_encoder.inverse_transform([pred_class])[0]
                        
                        predictions[name] = predicted_type
                        confidences[name] = max_proba
                        
                    except Exception as e:
                        logger.warning(f"Model {name} prediction failed: {e}")
                
                # Ensemble voting
                if predictions:
                    type_votes = {}
                    for name, pred_type in predictions.items():
                        confidence = confidences[name]
                        if pred_type not in type_votes:
                            type_votes[pred_type] = 0
                        type_votes[pred_type] += confidence
                    
                    best_type_str = max(type_votes.items(), key=lambda x: x[1])[0]
                    ml_confidence = type_votes[best_type_str] / len(predictions)
                    
                    # Convert to RequestType
                    for req_type in RequestType:
                        if req_type.value == best_type_str:
                            return req_type, ml_confidence
                            
            except Exception as e:
                logger.warning(f"ML classification failed: {e}")
        
         # Step 3: LLM classification (if available and trained)
        if LLM_AVAILABLE and self.llm_classifier and self._is_trained:
            try:
                llm_result, llm_confidence = self.llm_classifier.classify_request(request_text)
                if llm_confidence > 0.85:
                    return llm_result, llm_confidence
            except Exception as e:
                logger.warning(f"LLM classification failed: {e}")
        

        # Step 4: Enhanced keyword fallback
        return self._enhanced_keyword_classification(request_text)
    
    def _enhanced_keyword_classification(self, request_text: str) -> Tuple[RequestType, float]:
        """Enhanced keyword-based classification"""
        text_lower = request_text.lower()
        
        # Score each request type
        type_scores = {}
        
        for request_type, config in self.rule_patterns.items():
            score = sum(1 for keyword in config['keywords'] if keyword in text_lower)
            if score > 0:
                type_scores[request_type] = score
        
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])
            confidence = min(0.8, 0.4 + best_type[1] * 0.1)
            return best_type[0], confidence
        
        return RequestType.OTHER, 0.5
    
    def get_classification_explanation(self, request_text: str) -> str:
        """Get explanation for classification decision"""
        
        # Check rule-based first
        rule_result, rule_confidence = self._rule_based_classification(request_text)
        if rule_result and rule_confidence > 0.85:
            return f"High-confidence rule match for {rule_result.value}"
        
        # Check patterns
        text_lower = request_text.lower()
        explanations = []
        
        if any(word in text_lower for word in ['install', 'setup', 'vscode', 'docker']):
            explanations.append("Development tool indicators")
        
        if any(word in text_lower for word in ['admin', 'permission', 'access']):
            explanations.append("Permission/access keywords")
        
        if any(word in text_lower for word in ['port', 'network', 'vpn']):
            explanations.append("Network access terms")
        
        if any(word in text_lower for word in ['vendor', 'contractor', 'onboard']):
            explanations.append("Vendor approval keywords")
        
        if any(word in text_lower for word in ['data', 'export', 'database']):
            explanations.append("Data access indicators")
        
        if any(word in text_lower for word in ['cloud', 'aws', 'server']):
            explanations.append("Cloud resource terms")
        
        method = "Enhanced ML ensemble" if (ML_AVAILABLE and self._is_trained) else "Rule-based + keywords"
        
        if explanations:
            return f"{method}: {'; '.join(explanations[:3])}"
        else:
            return f"{method}: General classification"

    # Legacy compatibility methods
    def _classify_with_enhanced_patterns(self, text: str) -> Tuple[RequestType, float]:
        """Legacy method for compatibility"""
        return self._rule_based_classification(text) or (RequestType.OTHER, 0.5)
    
    def _simple_keyword_classification(self, text: str) -> Tuple[RequestType, float]:
        """Legacy method for compatibility"""
        return self._enhanced_keyword_classification(text)