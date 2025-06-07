import pytest
import time
import pandas as pd
from src.services.classification_service import ClassificationService
from src.services.adaptive_bot_system import AdaptiveBotSystem


class TestPerformance:
    """Performance tests for the system"""

    @pytest.fixture
    def large_dataset(self):
        """Create large dataset for performance testing"""
        data = {
            'request_summary': [
                                   'Install Docker Desktop for development',
                                   'Need admin access to production database',
                                   'Open port 443 for HTTPS traffic',
                                   'Export customer data for analysis',
                                   'SSH access to AWS EC2 instance'
                               ] * 200,  # 1000 records
            'request_type': [
                                'devtool_install',
                                'permission_change',
                                'network_access',
                                'data_export',
                                'cloud_resource_access'
                            ] * 200,
            'outcome': [
                           'Approved',
                           'Rejected',
                           'Approved',
                           'Needs More Info',
                           'Needs More Info'
                       ] * 200,
            'security_risk_score': [3.0, 8.5, 6.0, 7.0, 6.5] * 200
        }
        return pd.DataFrame(data)

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_classification_performance(self, benchmark, large_dataset):
        """Test classification performance"""
        classifier = ClassificationService()
        classifier.train(large_dataset)

        def classify_batch():
            results = []
            test_texts = [
                             "Install Docker Desktop for development",
                             "Need admin access to production database",
                             "Open port 443 for HTTPS traffic"
                         ] * 10  # 30 classifications

            for text in test_texts:
                result = classifier.classify_request(text)
                results.append(result)
            return results

        # Benchmark the classification
        results = benchmark(classify_batch)

        # Verify results are reasonable
        assert len(results) == 30
        for result_type, confidence in results:
            assert confidence > 0.0

    @pytest.mark.slow
    def test_training_performance(self, large_dataset):
        """Test training performance"""
        classifier = ClassificationService()

        start_time = time.time()
        classifier.train(large_dataset)
        training_time = time.time() - start_time

        # Training should complete in reasonable time
        assert training_time < 60  # Less than 60 seconds

        print(f"Training completed in {training_time:.2f} seconds")

    @pytest.mark.slow
    def test_adaptive_system_performance(self, large_dataset):
        """Test adaptive system performance"""
        bot_system = AdaptiveBotSystem()

        start_time = time.time()
        bot_system.train(large_dataset)
        training_time = time.time() - start_time

        assert training_time < 120  # Less than 2 minutes

        # Test processing performance
        from src.models.data_models import SecurityRequest
        from src.models.enums import RequestType

        request = SecurityRequest(
            user_id="perf_test",
            channel_id="test",
            thread_ts="",
            request_text="Install Docker Desktop for development",
            request_type=RequestType.DEVTOOL_INSTALL
        )

        start_time = time.time()
        processed_request, _ = bot_system.process_request(request)
        processing_time = time.time() - start_time

        # Processing should be fast
        assert processing_time < 5  # Less than 5 seconds
        assert processed_request.outcome is not None

        print(f"Request processing completed in {processing_time:.3f} seconds")

    @pytest.mark.slow
    def test_memory_usage(self, large_dataset):
        """Test memory usage doesn't grow excessively"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Train multiple systems
        for i in range(3):
            classifier = ClassificationService()
            classifier.train(large_dataset)

            bot_system = AdaptiveBotSystem()
            bot_system.train(large_dataset)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable
        assert memory_increase < 500  # Less than 500MB increase

        print(f"Memory usage increased by {memory_increase:.1f} MB")

