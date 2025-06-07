import pytest
from src.utils.validators import RequestValidator
from src.models.enums import RequestType


class TestRequestValidator:
    """Test validation utilities"""

    def test_validate_email(self):
        """Test email validation"""
        # Valid emails
        assert RequestValidator.validate_email("test@acme.com")
        assert RequestValidator.validate_email("user.name@company.co.uk")

        # Invalid emails
        assert not RequestValidator.validate_email("invalid-email")
        assert not RequestValidator.validate_email("@domain.com")
        assert not RequestValidator.validate_email("user@")
        assert not RequestValidator.validate_email("")

        # Slack mailto format
        assert RequestValidator.validate_email("<mailto:test@acme.com|test@acme.com>")

    def test_validate_duration(self):
        """Test duration validation"""
        # Valid durations
        assert RequestValidator.validate_duration("30 days")
        assert RequestValidator.validate_duration("2 hours")
        assert RequestValidator.validate_duration("1 week")
        assert RequestValidator.validate_duration("6 months")

        # Invalid durations
        assert not RequestValidator.validate_duration("invalid")
        assert not RequestValidator.validate_duration("30")
        assert not RequestValidator.validate_duration("")

    def test_validate_port(self):
        """Test port validation"""
        # Valid ports
        assert RequestValidator.validate_port("22")
        assert RequestValidator.validate_port("443")
        assert RequestValidator.validate_port("8080")
        assert RequestValidator.validate_port("65535")

        # Invalid ports
        assert not RequestValidator.validate_port("0")
        assert not RequestValidator.validate_port("65536")
        assert not RequestValidator.validate_port("invalid")
        assert not RequestValidator.validate_port("")

    def test_validate_required_fields(self):
        """Test required fields validation"""
        # Test network access validation
        fields = {
            'destination': '10.0.0.1',
            'port': '443',
            'protocol': 'HTTPS',
            'business_justification': 'API integration'
        }

        errors = RequestValidator.validate_required_fields(RequestType.NETWORK_ACCESS, fields)
        assert len(errors) == 0

        # Test with invalid port
        fields['port'] = '99999'
        errors = RequestValidator.validate_required_fields(RequestType.NETWORK_ACCESS, fields)
        assert 'port' in errors

        # Test permission change validation
        fields = {
            'target_system': 'Production DB',
            'permission_level': 'admin',
            'duration': '24 hours',
            'manager_approval': 'manager@acme.com'
        }

        errors = RequestValidator.validate_required_fields(RequestType.PERMISSION_CHANGE, fields)
        assert len(errors) == 0

        # Test with invalid email
        fields['manager_approval'] = 'invalid-email'
        errors = RequestValidator.validate_required_fields(RequestType.PERMISSION_CHANGE, fields)
        assert 'manager_approval' in errors
