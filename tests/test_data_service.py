import pytest
import pandas as pd
import tempfile
import os
from src.services.data_service import DataService
from src.models.enums import RequestType, Outcome


class TestDataService:
    """Test data service functionality"""
    
    @pytest.fixture
    def temp_csv(self):
        """Create temporary CSV file"""
        data = {
            'request_summary': [
                'Install Docker Desktop for development',
                'Need admin access to production database',
                'Open port 443 for HTTPS traffic'
            ],
            'request_type': ['devtool_install', 'permission_change', 'network_access'],
            'outcome': ['Approved', 'Rejected', 'Approved'],
            'security_risk_score': [3.0, 8.5, 6.0]
        }
        
        df = pd.DataFrame(data)
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        yield temp_file.name
        os.unlink(temp_file.name)
    
    def test_load_existing_data(self, temp_csv):
        """Test loading existing CSV data"""
        service = DataService(temp_csv)
        data = service.load_and_normalize_data()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) >= 3  # Original data plus synthetic
        assert 'request_summary' in data.columns
        assert 'request_type' in data.columns
        assert 'outcome' in data.columns
        assert 'security_risk_score' in data.columns
    
    def test_load_missing_data(self):
        """Test handling missing data file"""
        service = DataService("nonexistent_file.csv")
        data = service.load_and_normalize_data()
        
        # Should create sample data
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert 'request_summary' in data.columns
    
    def test_normalize_request_types(self, temp_csv):
        """Test request type normalization"""
        service = DataService(temp_csv)
        data = service.load_and_normalize_data()
        
        # Check that request types are normalized
        unique_types = data['request_type'].unique()
        for req_type in unique_types:
            # Should be valid RequestType values
            assert req_type in [e.value for e in RequestType]
    
    def test_normalize_outcomes(self, temp_csv):
        """Test outcome normalization"""
        service = DataService(temp_csv)
        data = service.load_and_normalize_data()
        
        # Check that outcomes are normalized
        unique_outcomes = data['outcome'].unique()
        for outcome in unique_outcomes:
            # Should be valid Outcome values
            assert outcome in [e.value for e in Outcome]
    
    def test_data_augmentation(self, temp_csv):
        """Test synthetic data augmentation"""
        service = DataService(temp_csv)
        original_data = pd.read_csv(temp_csv)
        augmented_data = service.load_and_normalize_data()
        
        # Should have more data after augmentation
        assert len(augmented_data) > len(original_data)
        
        # Should have better balance across request types
        type_counts = augmented_data['request_type'].value_counts()
        assert len(type_counts) >= 3  # Multiple request types
    
    def test_comprehensive_sample_data(self):
        """Test comprehensive sample data creation"""
        service = DataService("nonexistent.csv")
        data = service._create_comprehensive_sample_data()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) >= 10  # Should have reasonable amount of data
        
        # Should cover all major request types
        unique_types = data['request_type'].unique()
        expected_types = ['devtool_install', 'network_access', 'cloud_resource_access', 
                         'vendor_approval', 'permission_change', 'data_export']
        
        for expected_type in expected_types:
            assert expected_type in unique_types
