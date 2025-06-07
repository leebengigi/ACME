import pytest
import pandas as pd
from src.services.classification_service import ClassificationService
from src.services.adaptive_bot_system import AdaptiveBotSystem
from src.database.repository import SecurityRequestRepository
from src.models.data_models import SecurityRequest
from src.models.enums import RequestType, Outcome
import tempfile
import os


class TestIntegration:
    """Integration tests for the complete system"""

    @pytest.fixture
    def complete_system(self, temp_csv_file):
        """Setup complete system for integration testing"""
        # Create temporary database
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()

        # Initialize components
        classification_service = ClassificationService()
        bot_system = AdaptiveBotSystem()
        repository = SecurityRequestRepository(temp_db.name)

        # Load and train with sample data
        training_data = pd.read_csv(temp_csv_file)
        classification_service.train(training_data)
        bot_system.train(training_data)

        yield {
            'classifier': classification_service,
            'bot_system': bot_system,
            'repository': repository,
            'training_data': training_data
        }

        # Cleanup
        os.unlink(temp_db.name)

    def test_end_to_end_workflow(self, complete_system):
        """Test complete end-to-end workflow"""
        classifier = complete_system['classifier']
        bot_system = complete_system['bot_system']
        repository = complete_system['repository']

        # 1. Classification
        request_text = "Install Docker Desktop for development work"
        request_type, confidence = classifier.classify_request(request_text)

        assert request_type == RequestType.DEVTOOL_INSTALL
        assert confidence > 0.5

        # 2. Create security request
        request = SecurityRequest(
            user_id="integration_test",
            channel_id="test_channel",
            thread_ts="",
            request_text=request_text,
            request_type=request_type
        )

        # 3. Process with adaptive system
        processed_request, processing_info = bot_system.process_request(request)

        assert processed_request.risk_score is not None
        assert processed_request.outcome is not None
        assert processed_request.rationale is not None

        # 4. Save to database
        request_id = repository.save_request(processed_request)
        assert request_id > 0

        # 5. Verify database statistics
        stats = repository.get_statistics()
        assert stats['total_requests'] >= 1

    def test_follow_up_workflow(self, complete_system):
        """Test follow-up question workflow"""
        repository = complete_system['repository']

        # 1. Save incomplete request
        request = SecurityRequest(
            user_id="test_user",
            channel_id="test_channel",
            thread_ts="",
            request_text="Need network access",
            request_type=RequestType.NETWORK_ACCESS,
            required_fields={}  # Missing required fields
        )

        request_id = repository.save_request(request)

        # 2. Simulate follow-up response
        follow_up_fields = {
            "destination": "10.0.0.1",
            "port": "443",
            "protocol": "HTTPS",
            "business_justification": "API integration"
        }

        repository.update_request_fields(request_id, follow_up_fields)

        # 3. Retrieve and verify
        pending = repository.get_pending_request("test_user", "test_channel")
        assert pending is not None
        assert pending[1] == follow_up_fields

        # 4. Finalize request
        repository.finalize_request(request_id, "Approved", "Complete information provided")

        # 5. Verify no longer pending
        pending = repository.get_pending_request("test_user", "test_channel")
        assert pending is None

    def test_system_accuracy(self, complete_system):
        """Test overall system accuracy"""
        classifier = complete_system['classifier']
        bot_system = complete_system['bot_system']
        training_data = complete_system['training_data']

        # Test classification accuracy
        test_cases = training_data.head(20)  # Use subset for testing
        correct_classifications = 0

        for _, row in test_cases.iterrows():
            predicted_type, _ = classifier.classify_request(row['request_summary'])
            expected_type = RequestType(row['request_type'])

            if predicted_type == expected_type:
                correct_classifications += 1

        classification_accuracy = correct_classifications / len(test_cases)

        # Should achieve reasonable accuracy
        assert classification_accuracy >= 0.6  # 60% minimum accuracy

        # Test decision accuracy
        correct_decisions = 0

        for _, row in test_cases.iterrows():
            request = SecurityRequest(
                user_id="test",
                channel_id="test",
                thread_ts="",
                request_text=row['request_summary'],
                request_type=RequestType(row['request_type'])
            )

            processed_request, _ = bot_system.process_request(request)
            expected_outcome = Outcome(row['outcome'])

            if processed_request.outcome == expected_outcome:
                correct_decisions += 1

        decision_accuracy = correct_decisions / len(test_cases)

        # Should achieve reasonable decision accuracy
        assert decision_accuracy >= 0.4  # 40% minimum (decisions are more complex)

        print(f"Classification Accuracy: {classification_accuracy:.1%}")
        print(f"Decision Accuracy: {decision_accuracy:.1%}")

