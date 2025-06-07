import pytest
from src.slack.message_formatter import MessageFormatter
from src.models.enums import RequestType, Outcome
from src.models.data_models import SecurityRequest
from datetime import datetime


class TestMessageFormatter:
    """Test message formatting functionality"""

    def test_format_follow_up_questions(self):
        """Test follow-up question formatting"""
        missing_fields = ['destination', 'port', 'protocol']
        request_type = RequestType.NETWORK_ACCESS

        message = MessageFormatter.format_follow_up_questions(missing_fields, request_type)

        assert isinstance(message, str)
        assert len(message) > 0
        assert 'Network Access' in message
        assert 'destination' in message.lower()
        assert 'port' in message.lower()
        assert 'protocol' in message.lower()

    def test_format_final_response(self):
        """Test final response formatting"""
        request = SecurityRequest(
            user_id="test",
            channel_id="test",
            thread_ts="",
            request_text="Test request",
            request_type=RequestType.DEVTOOL_INSTALL,
            risk_score=3.5,
            outcome=Outcome.APPROVED,
            rationale="Low risk development tool"
        )

        response = MessageFormatter.format_final_response(request)

        assert isinstance(response, str)
        assert 'APPROVED' in response or '✅' in response
        assert 'Devtool Install' in response
        assert '3.5' in response
        assert 'Low risk development tool' in response

    def test_format_adaptive_response(self):
        """Test adaptive response formatting"""
        request = SecurityRequest(
            user_id="test",
            channel_id="test",
            thread_ts="",
            request_text="Test request",
            request_type=RequestType.PERMISSION_CHANGE,
            risk_score=6.5,
            outcome=Outcome.NEEDS_MORE_INFO,
            rationale="Requires manager approval"
        )

        processing_info = {
            'risk_assessment': {
                'method': 'ml_prediction',
                'confidence': 0.85,
                'risk_factors': ['admin_access', 'production_system']
            },
            'learned_thresholds': {
                'approval_threshold': 4.0,
                'rejection_threshold': 8.0
            },
            'training_samples': 1000
        }

        response = MessageFormatter.format_adaptive_response(request, processing_info)

        assert isinstance(response, str)
        assert 'REVIEW NEEDED' in response or '❓' in response
        assert 'Permission Change' in response
        assert '6.5' in response
        assert '85%' in response or '0.85' in response
        assert 'Requires manager approval' in response

    def test_format_error_response(self):
        """Test error response formatting"""
        # Test default error
        response = MessageFormatter.format_error_response()
        assert isinstance(response, str)
        assert 'error' in response.lower()

        # Test custom error
        custom_error = "Custom error message"
        response = MessageFormatter.format_error_response(custom_error)
        assert custom_error in response

    def test_format_help_message(self):
        """Test help message formatting"""
        response = MessageFormatter.format_help_message()

        assert isinstance(response, str)
        assert len(response) > 100  # Should be comprehensive
        assert 'security' in response.lower()
        assert 'request' in response.lower()
        assert '/security-request' in response
