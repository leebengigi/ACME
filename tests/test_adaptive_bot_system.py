import pytest
import pandas as pd
from src.services.adaptive_bot_system import AdaptiveBotSystem
from src.models.data_models import SecurityRequest
from src.models.enums import RequestType, Outcome


class TestAdaptiveBotSystem:
    """Test the adaptive bot system"""

    @pytest.fixture
    def bot_system(self):
        """Create bot system instance"""
        return AdaptiveBotSystem()

    @pytest.fixture
    def sample_data(self):
        """Sample historical data for training"""
        return pd.DataFrame({
            'request_summary': [
                'Install Docker Desktop for development',
                'Need admin access to production database',
                'Open port 443 for HTTPS traffic',
                'Export customer data for analysis',
                'SSH access to AWS EC2 instance'
            ],
            'request_type': [
                'devtool_install',
                'permission_change',
                'network_access',
                'data_export',
                'cloud_resource_access'
            ],
            'outcome': [
                'Approved',
                'Rejected',
                'Approved',
                'Needs More Info',
                'Needs More Info'
            ],
            'security_risk_score': [3.0, 8.5, 6.0, 7.0, 6.5]
        })

    def test_initialization(self, bot_system):
        """Test system initializes correctly"""
        assert bot_system is not None
        assert hasattr(bot_system, 'risk_assessment')
        assert hasattr(bot_system, 'decision_engine')

    def test_training(self, bot_system, sample_data):
        """Test system training"""
        result = bot_system.train(sample_data)
        assert isinstance(result, dict)
        assert 'training_samples' in result
        assert result['training_samples'] > 0

    def test_process_request(self, bot_system, sample_data):
        """Test request processing"""
        # Train the system first
        bot_system.train(sample_data)

        # Create test request
        request = SecurityRequest(
            user_id="test_user",
            channel_id="test_channel",
            thread_ts="",
            request_text="Install Docker Desktop for development work",
            request_type=RequestType.DEVTOOL_INSTALL
        )

        # Process the request
        processed_request, processing_info = bot_system.process_request(request)

        # Verify results
        assert processed_request.risk_score is not None
        assert processed_request.outcome is not None
        assert processed_request.rationale is not None
        assert isinstance(processing_info, dict)
        assert 'risk_assessment' in processing_info

    def test_simple_risk_fallback(self, bot_system):
        """Test simple risk fallback method"""
        # Test high risk keywords
        high_risk_text = "Need admin access to production database"
        risk = bot_system._simple_risk_fallback(high_risk_text)
        assert risk > 5.0

        # Test low risk keywords
        low_risk_text = "Read-only access to development logs"
        risk = bot_system._simple_risk_fallback(low_risk_text)
        assert risk < 5.0

    def test_simple_decision_fallback(self, bot_system):
        """Test simple decision fallback method"""
        # Test low risk decision
        request = SecurityRequest(
            user_id="test",
            channel_id="test",
            thread_ts="",
            request_text="test",
            risk_score=3.0
        )
        outcome, rationale = bot_system._simple_decision_fallback(request)
        assert outcome == Outcome.APPROVED
        assert "low risk" in rationale.lower()

        # Test high risk decision
        request.risk_score = 8.0
        outcome, rationale = bot_system._simple_decision_fallback(request)
        assert outcome == Outcome.REJECTED
        assert "high risk" in rationale.lower()
