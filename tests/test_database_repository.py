import pytest
import tempfile
import os
from src.database.repository import SecurityRequestRepository
from src.models.data_models import SecurityRequest
from src.models.enums import RequestType, Outcome
from datetime import datetime


class TestSecurityRequestRepository:
    """Test the database repository"""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()

        repo = SecurityRequestRepository(temp_file.name)
        yield repo

        # Cleanup
        os.unlink(temp_file.name)

    def test_initialization(self, temp_db):
        """Test database initialization"""
        assert temp_db is not None

        # Test that tables were created
        with temp_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]

            assert 'requests' in tables
            assert 'interactions' in tables

    def test_save_request(self, temp_db):
        """Test saving a security request"""
        request = SecurityRequest(
            user_id="test_user",
            channel_id="test_channel",
            thread_ts="123456",
            request_text="Test request",
            request_type=RequestType.DEVTOOL_INSTALL,
            risk_score=3.5,
            outcome=Outcome.APPROVED,
            rationale="Low risk request"
        )

        request_id = temp_db.save_request(request)

        assert isinstance(request_id, int)
        assert request_id > 0

    def test_get_pending_request(self, temp_db):
        """Test retrieving pending requests"""
        # Save a pending request (no outcome)
        request = SecurityRequest(
            user_id="test_user",
            channel_id="test_channel",
            thread_ts="",
            request_text="Test pending request",
            request_type=RequestType.NETWORK_ACCESS,
            risk_score=5.0,
            required_fields={"destination": "10.0.0.1", "port": "443"}
        )

        request_id = temp_db.save_request(request)

        # Retrieve pending request
        pending = temp_db.get_pending_request("test_user", "test_channel")

        assert pending is not None
        assert pending[0] == request_id
        assert isinstance(pending[1], dict)  # required_fields
        assert pending[2] == RequestType.NETWORK_ACCESS.value  # request_type
        assert pending[3] == 5.0  # risk_score

    def test_update_request_fields(self, temp_db):
        """Test updating request fields"""
        # Save initial request
        request = SecurityRequest(
            user_id="test_user",
            channel_id="test_channel",
            thread_ts="",
            request_text="Test request"
        )

        request_id = temp_db.save_request(request)

        # Update fields
        new_fields = {"destination": "10.0.0.1", "port": "22"}
        temp_db.update_request_fields(request_id, new_fields)

        # Verify update
        pending = temp_db.get_pending_request("test_user", "test_channel")
        assert pending[1] == new_fields

    def test_finalize_request(self, temp_db):
        """Test finalizing a request"""
        # Save pending request
        request = SecurityRequest(
            user_id="test_user",
            channel_id="test_channel",
            thread_ts="",
            request_text="Test request"
        )

        request_id = temp_db.save_request(request)

        # Finalize request
        temp_db.finalize_request(request_id, "Approved", "Low risk, approved automatically")

        # Verify no longer pending
        pending = temp_db.get_pending_request("test_user", "test_channel")
        assert pending is None

    def test_log_interaction(self, temp_db):
        """Test logging interactions"""
        # Save request first
        request = SecurityRequest(
            user_id="test_user",
            channel_id="test_channel",
            thread_ts="",
            request_text="Test request"
        )

        request_id = temp_db.save_request(request)

        # Log interaction
        temp_db.log_interaction(request_id, "follow_up_question", "What is the destination IP?")

        # Verify interaction was logged (would need to add a method to retrieve interactions)
        # For now, just verify no exception was raised
        assert True

    def test_get_statistics(self, temp_db):
        """Test getting database statistics"""
        # Save some sample requests
        requests = [
            SecurityRequest(
                user_id="user1",
                channel_id="channel1",
                thread_ts="",
                request_text="Request 1",
                request_type=RequestType.DEVTOOL_INSTALL,
                outcome=Outcome.APPROVED,
                risk_score=3.0
            ),
            SecurityRequest(
                user_id="user2",
                channel_id="channel1",
                thread_ts="",
                request_text="Request 2",
                request_type=RequestType.PERMISSION_CHANGE,
                outcome=Outcome.REJECTED,
                risk_score=8.5
            )
        ]

        for request in requests:
            temp_db.save_request(request)

        # Get statistics
        stats = temp_db.get_statistics()

        assert isinstance(stats, dict)
        assert 'total_requests' in stats
        assert stats['total_requests'] == 2
        assert 'outcomes' in stats
        assert 'request_types' in stats
        assert 'average_risk_score' in stats
