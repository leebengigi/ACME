import os
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.enums import RequestType


class Settings:
    def __init__(self):
        # Slack Configuration
        self.SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
        self.SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")
        self.SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")

        # Model Configuration
        self.RISK_MODEL = os.environ.get("RISK_MODEL", "google/flan-t5-base")
        self.USE_GPU = os.environ.get("USE_GPU", "false").lower() == "true"

        # Database Configuration
        self.DATABASE_PATH = os.environ.get("DATABASE_PATH", r"data\security_requests.db")

        # Data Configuration
        self.HISTORICAL_DATA_PATH = os.environ.get("HISTORICAL_DATA_PATH", r"data\acme_security_tickets.csv")

        # Validate required environment variables
        self._validate_config()

        # LLM Configuration
        self.ENABLE_LLM = False  # Set to False to disable LLM and use ML-only classification
        self.LLM_MODEL = "Qwen/Qwen3-0.6B"  
        self.LLM_TIMEOUT = 10  # Timeout in seconds for LLM calls

        # Enhanced Classification Settings
        self.CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.6
        self.PATTERN_MATCHING_ENABLED = True
        self.ML_ENSEMBLE_ENABLED = True

        # Debugging
        self.DETAILED_CLASSIFICATION_LOGGING = True

    def _validate_config(self):
        """Validate that required configuration is present"""
        required_vars = [
            "SLACK_BOT_TOKEN",
            "SLACK_APP_TOKEN",
            "SLACK_SIGNING_SECRET"
        ]

        missing = [var for var in required_vars if not getattr(self, var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    @property
    def REQUIRED_FIELDS(self) -> Dict[RequestType, List[str]]:
        """Required fields per request type"""
        return {
        RequestType.VENDOR_APPROVAL: [
           " vendor_security_questionnaire", "data_classification", "legal_review"
        ],
        RequestType.PERMISSION_CHANGE: [
            "business_justification", "duration", "manager_approval","target_system", "permission_level"
        ],
        RequestType.NETWORK_ACCESS: [
            "business_justification", "source_CIDR", "engineering_lead_approval"
        ],
        RequestType.FIREWALL_CHANGE: [
            "destination_ip", "source system", "business_justification"
        ],
        RequestType.DEVTOOL_INSTALL: [
            "business_justification", "manager_approval"
        ],
        RequestType.DATA_EXPORT: [
            "business_justification", "data_destination", "PII_involved"
        ],
        RequestType.CLOUD_RESOURCE_ACCESS: [
            "data_sensitivty_level", "business_justification"
        ],
        RequestType.OTHER: [
            "detailed_description"
        ]
    }

