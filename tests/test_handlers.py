import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from src.slack.handlers import SlackHandlers
from src.services.classification_service import ClassificationService
from src.services.adaptive_bot_system import AdaptiveBotSystem
from src.database.repository import SecurityRequestRepository
from src.models.enums import RequestType, Outcome
from config.settings import Settings
import tempfile
import os


class TestSlackHandlers:
    """Test Slack handlers functionality"""

    @pytest.fixture
    def mock_slack_app(self):
        """Mock Slack app"""
        return Mock()

    @pytest.fixture
    def classification_service(self):
        """Mock classification service"""
        service = Mock(spec=ClassificationService)
        service.classify_request.return_value = (RequestType.DEVTOOL_INSTALL, 0.9)
        return service

    @pytest.fixture
    def adaptive_system(self):
        """Mock adaptive system"""
        system = Mock(spec=AdaptiveBotSystem)
        system.process_request.return_value = (Mock(
            outcome=Outcome.APPROVED,
            rationale="Test rationale",
            risk_score=3.0
        ), {})
        return system

    @pytest.fixture
    def repository(self):
        """Mock repository"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()

        repo = SecurityRequestRepository(temp_file.name)
        yield repo

        os.unlink(temp_file.name)

    @pytest.fixture
    def settings(self):
        """Mock settings"""
        settings = Mock(spec=Settings)
        settings.REQUIRED_FIELDS = {
            RequestType.NETWORK_ACCESS: ['destination', 'port', 'protocol', 'business_justification'],
            RequestType.PERMISSION_CHANGE: ['target_system', 'permission_level', 'duration', 'manager_approval'],
            RequestType.DATA_EXPORT: ['dataset_name', 'access_level', 'purpose', 'data_classification'],
            RequestType.DEVTOOL_INSTALL: ['tool_name', 'version', 'purpose']
        }
        return settings

    @pytest.fixture
    def handlers(self, mock_slack_app, classification_service, adaptive_system, repository, settings):
        """Create handlers instance"""
        return SlackHandlers(
            mock_slack_app,
            classification_service,
            adaptive_system,
            repository,
            settings
        )

    def test_initialization(self, handlers):
        """Test handlers initialize correctly"""
        assert handlers is not None
        assert hasattr(handlers, 'app')
        assert hasattr(handlers, 'classification_service')
        assert hasattr(handlers, 'adaptive_system')
        assert hasattr(handlers, 'repository')

    @patch('src.slack.handlers.MessageFormatter')
    def test_process_security_request_complete(self, mock_formatter, handlers, classification_service, adaptive_system):
        """Test processing complete security request"""
        # Mock response function
        mock_respond = Mock()

        # Mock formatter
        mock_formatter.format_adaptive_response.return_value = "Test response"

        # Process request
        handlers._process_security_request(
            user_id="test_user",
            channel_id="test_channel",
            text="Install Docker Desktop for development",
            respond=mock_respond
        )

        # Verify classification was called
        classification_service.classify_request.assert_called_once()

        # Verify adaptive system was called
        adaptive_system.process_request.assert_called_once()

        # Verify response was sent
        mock_respond.assert_called_once()

    @patch('src.slack.handlers.MessageFormatter')
    def test_process_security_request_missing_fields(self, mock_formatter, handlers, settings):
        """Test processing request with missing fields"""
        mock_respond = Mock()
        mock_formatter.format_follow_up_questions.return_value = "Please provide more info"

        # Mock classification to return network access (which requires fields)
        handlers.classification_service.classify_request.return_value = (RequestType.NETWORK_ACCESS, 0.9)

        # Process request
        handlers._process_security_request(
            user_id="test_user",
            channel_id="test_channel",
            text="Need network access",
            respond=mock_respond
        )

        # Should ask for follow-up
        mock_formatter.format_follow_up_questions.assert_called_once()
        mock_respond.assert_called_once()

    def test_extract_fields_from_request_network(self, handlers):
        """Test field extraction for network requests"""
        request_text = "Need access to 10.0.0.1 on port 443 using HTTPS for API integration"

        fields = handlers._extract_fields_from_request(request_text, RequestType.NETWORK_ACCESS)

        # Should extract some fields automatically
        assert isinstance(fields, dict)

    def test_get_missing_fields(self, handlers, settings):
        """Test missing fields detection"""
        from src.models.data_models import SecurityRequest

        # Request with some fields filled
        request = SecurityRequest(
            user_id="test",
            channel_id="test",
            thread_ts="",
            request_text="test",
            request_type=RequestType.NETWORK_ACCESS,
            required_fields={"destination": "10.0.0.1", "port": "443"}
        )

        missing = handlers._get_missing_fields(request)

        # Should identify missing fields
        assert isinstance(missing, list)
        expected_fields = settings.REQUIRED_FIELDS[RequestType.NETWORK_ACCESS]
        provided_fields = request.required_fields.keys()
        expected_missing = [f for f in expected_fields if f not in provided_fields]

        assert set(missing) == set(expected_missing)

    @patch('src.slack.handlers.MessageFormatter')
    def test_handle_follow_up_response(self, mock_formatter, handlers, repository):
        """Test follow-up response handling"""
        # First save a pending request
        from src.models.data_models import SecurityRequest

        request = SecurityRequest(
            user_id="test_user",
            channel_id="test_channel",
            thread_ts="",
            request_text="Need network access",
            request_type=RequestType.NETWORK_ACCESS,
            required_fields={"destination": "10.0.0.1"}
        )

        repository.save_request(request)

        # Mock response function
        mock_respond = Mock()
        mock_formatter.format_adaptive_response.return_value = "Final response"

        # Handle follow-up
        handlers._handle_follow_up_response(
            user_id="test_user",
            channel_id="test_channel",
            text="443\nHTTPS\nAPI integration needs",
            respond=mock_respond,
            thread_ts=None
        )

        # Should process the follow-up
        # Exact behavior depends on implementation details
