import pytest
from src.utils.text_processing import TextProcessor
from src.models.enums import RequestType


class TestTextProcessor:
    """Test text processing utilities"""

    def test_clean_slack_text(self):
        """Test Slack text cleaning"""
        # Test user mention removal
        text = "<@U123456> need access to database"
        cleaned = TextProcessor.clean_slack_text(text)
        assert "<@U123456>" not in cleaned
        assert "need access to database" in cleaned

        # Test channel mention removal
        text = "Post in <#C123456|general> when done"
        cleaned = TextProcessor.clean_slack_text(text)
        assert "<#C123456|general>" not in cleaned
        assert "Post in when done" in cleaned.replace("  ", " ")

        # Test mailto handling
        text = "Contact <mailto:admin@acme.com|admin@acme.com>"
        cleaned = TextProcessor.clean_slack_text(text)
        assert "admin@acme.com" in cleaned
        assert "mailto:" not in cleaned

    def test_extract_email_from_slack_format(self):
        """Test email extraction from Slack format"""
        # Test normal email
        result = TextProcessor.extract_email_from_slack_format("admin@acme.com")
        assert result == "admin@acme.com"

        # Test Slack mailto format
        result = TextProcessor.extract_email_from_slack_format("<mailto:admin@acme.com|admin@acme.com>")
        assert result == "admin@acme.com"

        # Test empty input
        result = TextProcessor.extract_email_from_slack_format("")
        assert result == ""

    def test_parse_follow_up_answers(self):
        """Test follow-up answer parsing"""
        # Test simple line-by-line parsing
        text = "10.0.0.1\n22\nTCP\nBusiness need for database access"
        expected_fields = ['destination', 'port', 'protocol', 'business_justification']

        result = TextProcessor.parse_follow_up_answers(text, expected_fields)

        assert result['destination'] == '10.0.0.1'
        assert result['port'] == '22'
        assert result['protocol'] == 'tcp'
        assert result['business_justification'] == 'Business need for database access'

    def test_parse_network_request(self):
        """Test network request parsing"""
        text = "Need access to 10.0.0.1 on port 443 using HTTPS for API integration"
        expected_fields = ['destination', 'port', 'protocol', 'business_justification']

        result = TextProcessor._parse_network_request(text, expected_fields)

        assert '10.0.0.1' in str(result.get('destination', ''))
        assert '443' in str(result.get('port', ''))

    def test_parse_permission_request(self):
        """Test permission request parsing"""
        text = "Need admin access to production system for 24 hours with manager@acme.com approval"
        expected_fields = ['target_system', 'permission_level', 'duration', 'manager_approval']

        result = TextProcessor._parse_permission_request(text, expected_fields)

        assert 'admin' in str(result.get('permission_level', '')).lower()
        assert 'manager@acme.com' in str(result.get('manager_approval', ''))

