import pytest
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.classification_service import ClassificationService
from src.models.enums import RequestType


class TestClassificationService:
    """Test the enhanced classification service"""

    @pytest.fixture
    def classification_service(self):
        """Create a classification service instance"""
        return ClassificationService()

    @pytest.fixture
    def sample_training_data(self):
        """Sample training data for testing"""
        return pd.DataFrame({
            'request_summary': [
                'Install Docker Desktop for development',
                'Need admin access to production database',
                'Open port 443 for HTTPS traffic',
                'Export customer data for analysis',
                'SSH access to AWS EC2 instance',
                'Onboard vendor Microsoft for integration'
            ],
            'request_type': [
                'devtool_install',
                'permission_change',
                'network_access',
                'data_export',
                'cloud_resource_access',
                'vendor_approval'
            ]
        })

    def test_initialization(self, classification_service):
        """Test service initializes correctly"""
        assert classification_service is not None
        assert hasattr(classification_service, 'rule_patterns')
        assert len(classification_service.rule_patterns) > 0

    def test_classify_devtool_install(self, classification_service):
        """Test DevTool installation classification"""
        test_cases = [
            "Install Docker Desktop for development",
            "Setup Visual Studio Code with extensions",
            "Download Java JDK for programming"
        ]

        for text in test_cases:
            result_type, confidence = classification_service.classify_request(text)
            assert result_type == RequestType.DEVTOOL_INSTALL
            assert confidence > 0.5

    def test_classify_permission_change(self, classification_service):
        """Test permission change classification"""
        test_cases = [
            "Need admin access to production database",
            "Temporary elevated permissions for maintenance",
            "Root access for system updates"
        ]

        for text in test_cases:
            result_type, confidence = classification_service.classify_request(text)
            assert result_type == RequestType.PERMISSION_CHANGE
            assert confidence >= 0.5

    def test_classify_network_access(self, classification_service):
        """Test network access classification"""
        test_cases = [
            "Open port 443 for HTTPS traffic",
            "VPN access for remote work",
            "Network connectivity to partner systems"
        ]

        for text in test_cases:
            result_type, confidence = classification_service.classify_request(text)
            assert result_type == RequestType.NETWORK_ACCESS
            assert confidence > 0.5

    def test_classify_cloud_resource_access(self, classification_service):
        """Test cloud resource access classification"""
        test_cases = [
            "SSH access to AWS EC2 instance",
            "Access to Azure virtual machine",
            "AWS S3 bucket permissions"
        ]

        for text in test_cases:
            result_type, confidence = classification_service.classify_request(text)
            assert result_type == RequestType.CLOUD_RESOURCE_ACCESS
            assert confidence > 0.5

    def test_classify_data_export(self, classification_service):
        """Test data export classification"""
        test_cases = [
            "Export customer data for analysis",
            "Database access for reporting",
            "Query user logs for investigation"
        ]

        for text in test_cases:
            result_type, confidence = classification_service.classify_request(text)
            assert result_type == RequestType.DATA_EXPORT
            assert confidence >= 0.5

    def test_classify_vendor_approval(self, classification_service):
        """Test vendor approval classification"""
        test_cases = [
            "Onboard vendor Microsoft for integration",
            "Approve contractor for security audit",
            "Third-party service approval"
        ]

        for text in test_cases:
            result_type, confidence = classification_service.classify_request(text)
            assert result_type == RequestType.VENDOR_APPROVAL
            assert confidence >= 0.5

    def test_empty_text_handling(self, classification_service):
        """Test handling of empty or invalid text"""
        result_type, confidence = classification_service.classify_request("")
        assert result_type in RequestType
        assert 0 <= confidence <= 1

        result_type, confidence = classification_service.classify_request("   ")
        assert result_type in RequestType
        assert 0 <= confidence <= 1

    def test_training(self, classification_service, sample_training_data):
        """Test training functionality"""
        result = classification_service.train(sample_training_data)
        assert isinstance(result, dict)

        # Check that training improved classification
        test_text = "Install Docker for development"
        result_type, confidence = classification_service.classify_request(test_text)
        assert result_type == RequestType.DEVTOOL_INSTALL

    def test_get_classification_explanation(self, classification_service):
        """Test explanation functionality"""
        text = "Install Docker Desktop for development"
        explanation = classification_service.get_classification_explanation(text)
        assert isinstance(explanation, str)
        assert len(explanation) > 0

