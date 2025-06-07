import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path


@pytest.fixture(scope="session")
def sample_csv_data():
    """Create sample CSV data for testing"""
    data = {
        'ticket_id': [f'TICKET-{i:04d}' for i in range(1, 101)],
        'created_at': ['2024-01-01 10:00:00'] * 100,
        'requester_department': ['Engineering'] * 50 + ['Marketing'] * 30 + ['Sales'] * 20,
        'requester_title': ['Developer'] * 40 + ['Manager'] * 30 + ['Analyst'] * 30,
        'request_type': (
                ['devtool_install'] * 20 +
                ['permission_change'] * 20 +
                ['network_access'] * 20 +
                ['data_export'] * 15 +
                ['cloud_resource_access'] * 15 +
                ['vendor_approval'] * 10
        ),
        'request_summary': [
                               'Install Docker Desktop for development',
                               'Need admin access to production database',
                               'Open port 443 for HTTPS traffic',
                               'Export customer data for analysis',
                               'SSH access to AWS EC2 instance'
                           ] * 20,
        'details': ['Detailed description here'] * 100,
        'mandatory_fields': ['field1,field2'] * 100,
        'fields_provided': ['field1'] * 100,
        'outcome': (
                ['Approved'] * 60 +
                ['Rejected'] * 25 +
                ['Needs More Info'] * 15
        ),
        'security_risk_score': [3.0, 8.5, 6.0, 7.0, 6.5] * 20,
        'resolution_time_hours': [2.5, 24.0, 4.0, 8.5, 12.0] * 20,
        'approver_role': ['Security Team'] * 100
    }

    return pd.DataFrame(data)


@pytest.fixture
def temp_csv_file(sample_csv_data):
    """Create temporary CSV file for testing"""
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
    sample_csv_data.to_csv(temp_file.name, index=False)
    temp_file.close()

    yield temp_file.name

    # Cleanup
    os.unlink(temp_file.name)

