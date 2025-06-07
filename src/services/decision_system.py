"""
Decision System Module for Security Request Processing.

This module provides an integrated decision-making system that combines
risk assessment and decision engine components to process security requests.
It ensures proper coordination between components and maintains consistency
in decision-making.

Key Features:
- Integrated risk assessment and decision making
- Threshold-based decision rules
- Consistency validation
- Safety rules enforcement
- Comprehensive logging and monitoring
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, Dict
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.enums import Outcome, RequestType
from src.models.data_models import SecurityRequest

logger = logging.getLogger(__name__)

class DecisionSystem:
    """
    Integrated system for security request decision making.
    
    This class coordinates between risk assessment and decision engine components
    to make consistent and safe decisions about security requests. It ensures
    proper integration of components and maintains decision-making thresholds.
    
    Attributes:
        risk_assessment: Component for assessing request risk
        decision_engine: Component for making final decisions
        shared_thresholds (Dict[str, float]): Common thresholds for decisions
        _is_integrated (bool): Whether components are properly integrated
        training_stats (Dict): Statistics from training process
    """
    
    def __init__(self):
        """Initialize the decision system with default settings."""
        # Risk Assessment Component
        self.risk_assessment = None
        
        # Decision Engine Component  
        self.decision_engine = None
        
        self.shared_thresholds = {
            'approval_threshold': 6.5,
            'rejection_threshold': 8.8,
            'confidence_threshold': 1.0
        }
        
        # Integration state
        self._is_integrated = False
        self.training_stats = {}

    def integrate_components(self, risk_assessment, decision_engine):
        """
        Properly integrate risk assessment and decision engine components.
        
        This method establishes the necessary connections between components:
        1. Links decision engine to risk assessment
        2. Shares thresholds between components
        3. Validates integration
        
        Args:
            risk_assessment: Risk assessment component
            decision_engine: Decision engine component
            
        Raises:
            ValueError: If components cannot be properly integrated
        """
        logger.info("🔗 Integrating Risk Assessment ↔ Decision Engine...")
        
        # Set component references
        self.risk_assessment = risk_assessment
        self.decision_engine = decision_engine
                
        # 1. Decision engine needs risk assessment for thresholds
        if hasattr(decision_engine, 'set_risk_assessment'):
            decision_engine.set_risk_assessment(risk_assessment)
            logger.info("✅ Decision engine → Risk assessment link established")
        
        # 2. Risk assessment needs to share thresholds with decision engine
        if hasattr(risk_assessment, 'get_learned_thresholds'):
            shared_thresholds = risk_assessment.get_learned_thresholds()
            if hasattr(decision_engine, 'risk_thresholds'):
                decision_engine.risk_thresholds.update(shared_thresholds)
                logger.info("✅ Risk assessment → Decision engine threshold sharing established")
        
        self._is_integrated = True
        logger.info("🎯 Components successfully integrated")

    def train_integrated_system(self, historical_data: pd.DataFrame):
        """
        Train both components with proper data sharing.
        
        This method coordinates the training process:
        1. Trains risk assessment first to learn thresholds
        2. Shares learned thresholds with decision engine
        3. Trains decision engine with updated thresholds
        4. Validates integration
        
        Args:
            historical_data (pd.DataFrame): Training data with request history
            
        Raises:
            RuntimeError: If components are not integrated
            Exception: If training fails
        """
        logger.info("🎯 Training Integrated Risk-Decision System...")
        
        if not self._is_integrated:
            logger.error("Components not integrated! Call integrate_components() first")
            return
        
        try:
            # STEP 1: Train Risk Assessment first (it learns the thresholds)
            logger.info("1️⃣ Training Risk Assessment...")
            self.risk_assessment.train(historical_data)
            
            # STEP 2: Extract learned thresholds from risk assessment
            if hasattr(self.risk_assessment, 'learned_thresholds'):
                learned_thresholds = self.risk_assessment.learned_thresholds
                self.shared_thresholds.update(learned_thresholds)
                logger.info(f"📊 Learned thresholds: {learned_thresholds}")
            
            # STEP 3: Train Decision Engine with shared thresholds
            logger.info("2️⃣ Training Decision Engine...")
            
            # Pass thresholds to decision engine BEFORE training
            if hasattr(self.decision_engine, 'risk_thresholds'):
                self.decision_engine.risk_thresholds.update(self.shared_thresholds)
            
            self.decision_engine.train(historical_data)
            
            # STEP 4: Validate integration
            self._validate_integration(historical_data)
            
            logger.info("✅ Integrated system training completed")
            
        except Exception as e:
            logger.error(f"Integrated training failed: {e}")
            raise

    def _validate_integration(self, historical_data: pd.DataFrame):
        """
        Validate that components work together properly.
        
        This method tests the integrated system on sample data to ensure:
        1. Risk assessment produces valid scores
        2. Decision engine makes appropriate decisions
        3. Components communicate effectively
        
        Args:
            historical_data (pd.DataFrame): Data to use for validation
        """
        logger.info("🔍 Validating component integration...")
        
        # Test on sample data
        sample_data = historical_data.head(10)
        correct_predictions = 0
        
        for _, row in sample_data.iterrows():
            try:
                # Create test request
                request = SecurityRequest(
                    user_id="integration_test",
                    channel_id="test",
                    thread_ts="",
                    request_text=row['request_summary'],
                    request_type=RequestType(row['request_type'])
                )
                
                # STEP 1: Risk Assessment
                risk_score, risk_info = self.risk_assessment.assess_risk(
                    request.request_text,
                    request.request_type.value
                )
                request.risk_score = risk_score
                
                # STEP 2: Decision Engine (using the risk score)
                predicted_outcome, rationale = self.decision_engine.make_decision(request)
                true_outcome = Outcome(row['outcome'])
                
                if predicted_outcome == true_outcome:
                    correct_predictions += 1
                
                logger.debug(f"Risk: {risk_score:.1f} → Decision: {predicted_outcome.value} (True: {true_outcome.value})")
                
            except Exception as e:
                logger.warning(f"Integration test failed on sample: {e}")
                continue
        
        accuracy = correct_predictions / len(sample_data) if len(sample_data) > 0 else 0
        self.training_stats['integration_accuracy'] = accuracy
        
        logger.info(f"🎯 Integration validation: {correct_predictions}/{len(sample_data)} = {accuracy:.1%}")

    def process_request_integrated(self, request: SecurityRequest) -> Tuple[SecurityRequest, Dict]:
        """
        Process request with proper component coordination.
        
        This method handles the complete request processing pipeline:
        1. Assesses risk using risk assessment component
        2. Makes decision using decision engine
        3. Validates consistency
        4. Applies safety rules
        
        Args:
            request (SecurityRequest): The security request to process
            
        Returns:
            Tuple[SecurityRequest, Dict]: Updated request and processing info
            
        Raises:
            Exception: If processing fails (handled with fallback)
        """
        processing_info = {
            'timestamp': datetime.now().isoformat(),
            'risk_assessment': {},
            'decision_info': {},
            'integration_method': 'coordinated'
        }
        
        try:
            # STEP 1: Risk Assessment
            logger.debug(f"🎲 Assessing risk for: {request.request_text[:50]}...")
            
            risk_score, risk_info = self.risk_assessment.assess_risk(
                request.request_text,
                request.request_type.value if request.request_type else 'other'
            )
            
            # Validate risk score
            if not (1.0 <= risk_score <= 10.0):
                logger.warning(f"Invalid risk score {risk_score}, clamping to [1.0, 10.0]")
                risk_score = max(1.0, min(10.0, risk_score))
            
            request.risk_score = risk_score
            processing_info['risk_assessment'] = risk_info
            
            logger.debug(f"✅ Risk assessment: {risk_score:.1f}")
            
            # STEP 2: Decision Engine (with proper threshold awareness)
            logger.debug(f"🤖 Making decision with risk score {risk_score:.1f}...")
            
            # Ensure decision engine has current thresholds
            if hasattr(self.decision_engine, 'risk_thresholds'):
                current_thresholds = self.risk_assessment.get_learned_thresholds()
                self.decision_engine.risk_thresholds.update(current_thresholds)
            
            outcome, rationale = self.decision_engine.make_decision(request)
            
            request.outcome = outcome
            request.rationale = rationale
            processing_info['decision_info'] = {
                'outcome': outcome.value,
                'method': 'integrated_decision',
                'thresholds_used': self.shared_thresholds
            }
            
            logger.debug(f"✅ Decision: {outcome.value}")
            
            # STEP 3: Consistency check
            self._check_decision_consistency(request, processing_info)
            
            return request, processing_info
            
        except Exception as e:
            logger.error(f"Integrated processing failed: {e}")
            
            # Emergency fallback
            request.risk_score = request.risk_score or 5.0
            request.outcome = Outcome.NEEDS_MORE_INFO
            request.rationale = f"Integration error - manual review required: {str(e)[:100]}"
            
            processing_info['error'] = str(e)
            processing_info['method'] = 'emergency_fallback'
            
            return request, processing_info

    def _check_decision_consistency(self, request: SecurityRequest, processing_info: Dict):
        """
        Check if risk score and decision are consistent.
        
        This method validates that the decision made is consistent with
        the risk score and established thresholds.
        
        Args:
            request (SecurityRequest): The processed request
            processing_info (Dict): Processing information to update
        """
        risk_score = request.risk_score
        outcome = request.outcome
        
        approve_threshold = self.shared_thresholds.get('approval_threshold', 4.0)
        reject_threshold = self.shared_thresholds.get('rejection_threshold', 7.0)
        
        # Consistency checks
        inconsistency_detected = False
        
        if risk_score <= approve_threshold and outcome == Outcome.REJECTED:
            logger.warning(f"⚠️ Inconsistency: Low risk ({risk_score:.1f}) but REJECTED")
            inconsistency_detected = True
        
        elif risk_score >= reject_threshold and outcome == Outcome.APPROVED:
            logger.warning(f"⚠️ Inconsistency: High risk ({risk_score:.1f}) but APPROVED")
            inconsistency_detected = True
        
        processing_info['consistency_check'] = {
            'risk_score': risk_score,
            'decision': outcome.value,
            'thresholds': {'approve': approve_threshold, 'reject': reject_threshold},
            'consistent': not inconsistency_detected
        }

    def _apply_safety_rules(self, request: SecurityRequest) -> Optional[Tuple[Outcome, str]]:
        """
        Apply safety rules to enforce security policies.
        
        This method implements critical safety checks that override
        normal decision-making for security-critical cases.
        
        Args:
            request (SecurityRequest): The request to check
            
        Returns:
            Optional[Tuple[Outcome, str]]: Override decision and rationale if safety rule applies
        """
        text = request.request_text.lower()
        score = request.risk_score or 5.0

        for phrase in self.critical_violations:
            if phrase in text:
                return Outcome.REJECTED, f"SECURITY VIOLATION: '{phrase}' found ⇒ auto‐reject"

        if score >= 9.0:
            return Outcome.REJECTED, f"CRITICAL RISK: {score:.1f}/10 ≥ 9.0 ⇒ auto‐reject"

        return None

