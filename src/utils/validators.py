import re
from typing import Optional, Dict, Any
from src.models.enums import RequestType


class RequestValidator:
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format, handling Slack mailto format"""
        if not email:
            return False

        # Extract email from Slack mailto format if present
        clean_email = RequestValidator._extract_email_from_slack_format(email)

        # Validate the cleaned email
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        return bool(re.match(pattern, clean_email))

    @staticmethod
    def _extract_email_from_slack_format(text: str) -> str:
        """Extract email from Slack's mailto format"""
        if not text:
            return text

        # Handle Slack mailto format: <mailto:email@domain.com|email@domain.com>
        mailto_match = re.search(r'<mailto:([^|>]+)\|[^>]+>', text)
        if mailto_match:
            return mailto_match.group(1)

        # Return original text if no mailto format found
        return text

    @staticmethod
    def validate_duration(duration: str) -> bool:
        """Validate duration format (e.g., '30 days', '2 hours')"""
        if not duration:
            return False

        pattern = r'^\d+\s*(hour|hours|day|days|week|weeks|month|months)'
        return bool(re.match(pattern, duration.lower()))

    @staticmethod
    def validate_port(port: str) -> bool:
        """Validate port number"""
        if not port:
            return False

        try:
            port_num = int(port)
            return 1 <= port_num <= 65535
        except ValueError:
            return False

    @staticmethod
    def validate_required_fields(request_type: RequestType, fields: Dict[str, Any]) -> Dict[str, str]:
        """Validate required fields for a request type"""
        errors = {}


        if request_type == RequestType.NETWORK_ACCESS:
            if 'port' in fields and fields['port']:
                if not RequestValidator.validate_port(fields['port']):
                    errors['port'] = "Invalid port number"

        elif request_type == RequestType.PERMISSION_CHANGE:
            if 'manager_approval' in fields and fields['manager_approval']:
                if not RequestValidator.validate_email(fields['manager_approval']):
                    errors['manager_approval'] = "Please provide a valid email address"

            if 'duration' in fields and fields['duration']:
                if not RequestValidator.validate_duration(fields['duration']):
                    errors['duration'] = "Please specify duration (e.g., '30 days', '2 hours')"
        return errors