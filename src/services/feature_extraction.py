"""
Feature Extraction Module for Security Request Analysis.

This module provides a comprehensive feature extraction pipeline for analyzing
security requests. It combines various text analysis techniques including:
- TF-IDF vectorization for text content
- Security keyword detection
- Text complexity analysis
- Request type encoding

The pipeline is designed to extract meaningful features that can be used for
security request classification and risk assessment.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)

class SecurityKeywordTransformer(BaseEstimator, TransformerMixin):
    """
    Extract security-related keyword features from text.
    
    This transformer identifies and counts occurrences of security-related keywords
    in different categories such as high-risk terms, access levels, urgency indicators,
    temporal aspects, and system environments.
    
    Attributes:
        security_keywords (Dict[str, List[str]]): Dictionary of keyword categories
            and their associated terms.
    """
    
    def __init__(self):
        """
        Initialize the transformer with predefined security keyword categories.
        
        Categories include:
        - high_risk_keywords: Terms indicating sensitive or critical systems
        - access_level_keywords: Terms describing access permissions
        - urgency_keywords: Terms indicating time sensitivity
        - temporal_keywords: Terms describing duration of access
        - system_keywords: Terms identifying system environments
        """
        self.security_keywords = {
            'high_risk_keywords': ['admin', 'root', 'production', 'prod', 'database', 'pii', 'personal', 'confidential'],
            'access_level_keywords': ['read-only', 'readonly', 'write', 'full', 'admin', 'administrator'],
            'urgency_keywords': ['urgent', 'emergency', 'immediate', 'asap', 'critical'],
            'temporal_keywords': ['temporary', 'permanent', 'short-term', 'long-term', 'indefinite'],
            'system_keywords': ['production', 'staging', 'development', 'test', 'prod', 'dev']
        }
        
    def fit(self, X, y=None):
        """
        Fit the transformer (no-op as keywords are predefined).
        
        Args:
            X: Input data (not used)
            y: Target values (not used)
            
        Returns:
            self: The transformer instance
        """
        return self
        
    def transform(self, X):
        """
        Transform input text into keyword feature counts.
        
        Args:
            X: List of text strings to transform
            
        Returns:
            np.ndarray: Feature matrix with keyword counts for each category
        """
        features = np.zeros((len(X), len(self.security_keywords)))
        for i, text in enumerate(X):
            text_lower = str(text).lower()
            for j, (_, keywords) in enumerate(self.security_keywords.items()):
                features[i, j] = sum(1 for keyword in keywords if keyword in text_lower)
        return features
    
    def get_feature_names_out(self, input_features=None):
        """
        Get the names of the output features.
        
        Returns:
            List[str]: Names of the keyword categories
        """
        return list(self.security_keywords.keys())

class TextComplexityTransformer(BaseEstimator, TransformerMixin):
    """
    Extract text complexity features from input text.
    
    This transformer calculates various metrics about text complexity including
    length, word count, average word length, punctuation usage, and capitalization.
    
    Attributes:
        feature_names (List[str]): Names of the extracted features
    """
    
    def __init__(self):
        """
        Initialize the transformer with feature names.
        
        Features include:
        - text_length: Total character count
        - word_count: Number of words
        - avg_word_length: Average length of words
        - exclamation_count: Number of exclamation marks
        - question_count: Number of question marks
        - caps_ratio: Ratio of uppercase characters
        """
        self.feature_names = [
            'text_length', 'word_count', 'avg_word_length',
            'exclamation_count', 'question_count', 'caps_ratio'
        ]
        
    def fit(self, X, y=None):
        """
        Fit the transformer (no-op as features are calculated directly).
        
        Args:
            X: Input data (not used)
            y: Target values (not used)
            
        Returns:
            self: The transformer instance
        """
        return self
        
    def transform(self, X):
        """
        Transform input text into complexity features.
        
        Args:
            X: List of text strings to transform
            
        Returns:
            np.ndarray: Feature matrix with complexity metrics
        """
        features = np.zeros((len(X), len(self.feature_names)))
        for i, text in enumerate(X):
            text = str(text)
            word_count = len(text.split())
            features[i] = [
                len(text),  # text_length
                word_count,  # word_count
                len(text) / word_count if word_count > 0 else 0,  # avg_word_length
                text.count('!'),  # exclamation_count
                text.count('?'),  # question_count
                sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0  # caps_ratio
            ]
        return features
    
    def get_feature_names_out(self, input_features=None):
        """
        Get the names of the output features.
        
        Returns:
            List[str]: Names of the complexity features
        """
        return self.feature_names

class RequestTypeTransformer(BaseEstimator, TransformerMixin):
    """
    Transform request types into encoded numerical values.
    
    This transformer uses LabelEncoder to convert categorical request types
    into numerical values suitable for machine learning models.
    
    Attributes:
        encoder (LabelEncoder): The label encoder instance
    """
    
    def __init__(self):
        """Initialize the transformer with a LabelEncoder."""
        self.encoder = LabelEncoder()
        
    def fit(self, X, y=None):
        """
        Fit the label encoder to the request types.
        
        Args:
            X: List of request type strings
            y: Target values (not used)
            
        Returns:
            self: The transformer instance
        """
        self.encoder.fit(X)
        return self
        
    def transform(self, X):
        """
        Transform request types into encoded values.
        
        Args:
            X: List of request type strings to encode
            
        Returns:
            np.ndarray: Encoded request types as a column vector
        """
        try:
            return self.encoder.transform(X).reshape(-1, 1)
        except:
            return np.zeros((len(X), 1))
    
    def get_feature_names_out(self, input_features=None):
        """
        Get the name of the output feature.
        
        Returns:
            List[str]: Name of the encoded request type feature
        """
        return ['request_type_encoded']

class FeatureExtractionPipeline:
    """
    Complete feature extraction pipeline for security requests.
    
    This class combines multiple feature extraction techniques into a single
    pipeline, including TF-IDF vectorization, security keyword detection,
    text complexity analysis, and request type encoding.
    
    Attributes:
        pipeline (Pipeline): The complete feature extraction pipeline
        feature_names (List[str]): Names of all extracted features
        _is_fitted (bool): Whether the pipeline has been fitted
    """
    
    def __init__(self):
        """Initialize the pipeline with default settings."""
        self.pipeline = None
        self.feature_names = None
        self._is_fitted = False
        
    def create_pipeline(self):
        """
        Create the feature extraction pipeline.
        
        The pipeline combines:
        - TF-IDF vectorization for text content
        - Security keyword detection
        - Text complexity analysis
        - Request type encoding
        """
        # Text features pipeline
        text_features = FeatureUnion([
            ('tfidf', TfidfVectorizer(
                max_features=500,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.95
            )),
            ('security_keywords', SecurityKeywordTransformer()),
            ('text_complexity', TextComplexityTransformer())
        ])
        
        # Complete pipeline
        self.pipeline = Pipeline([
            ('features', FeatureUnion([
                ('text_features', text_features),
                ('request_type', RequestTypeTransformer())
            ]))
        ])
        
    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Fit the pipeline and transform the input data.
        
        Args:
            data (pd.DataFrame): Input data containing 'request_summary' and
                               'request_type' columns
                               
        Returns:
            np.ndarray: Extracted features matrix
            
        Raises:
            ValueError: If required columns are missing
        """
        if self.pipeline is None:
            self.create_pipeline()
            
        logger.info("Fitting feature extraction pipeline...")
        
        # Prepare input data
        X_text = data['request_summary'].fillna('')
        X_type = data['request_type'].fillna('other')
        
        # Fit and transform
        features = self.pipeline.fit_transform(X_text)
        
        # Store feature names
        self.feature_names = self.pipeline.named_steps['features'].get_feature_names_out()
        self._is_fitted = True
        
        logger.info(f"Extracted {features.shape[1]} features")
        return features
    
    def transform(self, request_text: str, request_type: str) -> np.ndarray:
        """
        Transform a single request into features.
        
        Args:
            request_text (str): The request text to transform
            request_type (str): The type of request
            
        Returns:
            np.ndarray: Extracted features for the request
            
        Raises:
            ValueError: If the pipeline hasn't been fitted
        """
        if not self._is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
            
        # Transform single request
        features = self.pipeline.transform([request_text])
        
        if features.shape[1] != len(self.feature_names):
            logger.error(f"Feature dimension mismatch: got {features.shape[1]}, expected {len(self.feature_names)}")
            # Pad with zeros if necessary
            if features.shape[1] < len(self.feature_names):
                features = np.pad(features, ((0, 0), (0, len(self.feature_names) - features.shape[1])))
            else:
                features = features[:, :len(self.feature_names)]
                
        return features
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all extracted features.
        
        Returns:
            List[str]: Names of all features
            
        Raises:
            ValueError: If the pipeline hasn't been fitted
        """
        if not self._is_fitted:
            raise ValueError("Pipeline must be fitted before getting feature names")
        return self.feature_names 