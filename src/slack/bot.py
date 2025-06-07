import sys
from pathlib import Path

# Add project root to Python path for proper module imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import os
import logging
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from config.settings import Settings
from src.services.classification_service import ClassificationService
from src.services.adaptive_bot_system import AdaptiveBotSystem
from src.services.data_service import DataService
from src.database.repository import SecurityRequestRepository
from src.slack.handlers import SlackHandlers
from src.tools.diagnose_features import diagnose_feature_mismatch

logger = logging.getLogger(__name__)


class SecurityBot:
    """
    Main security bot class that handles Slack integration and security request processing.
    
    This bot integrates with Slack to process security requests, using a ML system
    for request classification and risk assessment. It maintains a database of requests
    and provides real-time responses to security-related queries.
    """

    def __init__(self, settings: Settings = None):
        """
        Initialize the security bot with required services and configurations.
        
        Args:
            settings: Configuration settings for the bot. If None, default settings are used.
        """
        self.settings = settings or Settings()

        # Initialize core services for data processing and ML
        self.data_service = DataService(self.settings.HISTORICAL_DATA_PATH)
        self.classification_service = ClassificationService()
        self.ML_system = AdaptiveBotSystem()

        # Initialize database connection for request persistence
        self.repository = SecurityRequestRepository(self.settings.DATABASE_PATH)

        # Initialize Slack app with authentication
        self.slack_app = App(
            token=self.settings.SLACK_BOT_TOKEN,
            signing_secret=self.settings.SLACK_SIGNING_SECRET
        )

        # Setup message handlers with all required services
        self.handlers = SlackHandlers(
            self.slack_app,
            self.classification_service,
            self.ML_system,
            self.repository,
            self.settings
        )

        # Initialize ML systems with historical data
        self._initialize_ML_system()

    def _initialize_ML_system(self):
        """
        Initialize and train the ML system using historical data.
        
        This method:
        1. Loads and normalizes historical security request data
        2. Trains the classification service for request categorization
        3. Trains the adaptive system for risk assessment and decision making
        4. Logs initialization status and database statistics
        """
        try:
            logger.info("Initializing Security Bot System...")

            # Load and prepare historical data for training
            historical_data = self.data_service.load_and_normalize_data()

            # Train classification service for request categorization
            self.classification_service.train(historical_data)

            # Train the complete system for risk assessment and decisions
            self.ML_system.train(historical_data)

            logger.info("Adaptive Security Bot System initialized successfully!")

            # Log current database statistics
            stats = self.repository.get_statistics()
            logger.info(f"Database stats: {stats}")

        except Exception as e:
            logger.error(f"Failed to initialize adaptive system: {e}")
            raise

    def start(self):
        """
        Start the bot in socket mode for real-time communication.
        
        This method initializes the Slack socket mode handler and begins
        listening for incoming messages and events.
        """
        try:
            handler = SocketModeHandler(self.slack_app, self.settings.SLACK_APP_TOKEN)
            logger.info("ACME Security Bot is starting...")
            handler.start()
        except Exception as e:
            logger.error(f"Failed to start bot: {e}")
            raise

    def get_statistics(self):
        """
        Retrieve current statistics about processed security requests.
        
        Returns:
            dict: Statistics about the bot's operation and request processing
        """
        return self.repository.get_statistics()

    def diagnose_feature_mismatch(self, request_text: str, request_type: str):
        """
        Diagnose potential issues with feature extraction for a given request.
        
        This method analyzes how well the system is extracting features from
        the request text and provides detailed diagnostics about feature
        extraction performance.
        
        Args:
            request_text: The text of the security request
            request_type: The type of security request being analyzed
        """
        try:
            risk_assessment = self.adaptive_system.risk_assessment
            diagnosis = diagnose_feature_mismatch(risk_assessment, request_text, request_type)
            
            # Log detailed feature extraction diagnostics
            logger.info("Feature Diagnosis:")
            logger.info(f"- Total features expected: {diagnosis['total_features']}")
            logger.info(f"- Features extracted: {diagnosis['features_extracted']}")
            logger.info(f"- Non-zero features: {diagnosis['non_zero_features']}")
            logger.info("- Top active features:")
            for feature in diagnosis['top_active_features'][:5]:
                logger.info(f"  * {feature['name']}: {feature['value']:.3f}")
        except Exception as e:
            logger.error(f"Feature diagnosis failed: {e}")