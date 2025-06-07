"""
Main module for the ACME Security Bot application.
Handles initialization, configuration, and bot startup.
"""
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.logging_config import setup_logging
from config.settings import Settings
from src.slack.bot import SecurityBot


def main():
    """
    Main entry point for the security bot application.
    
    Initializes the environment, sets up logging, and starts the Slack bot.
    Handles graceful shutdown on keyboard interrupt and error logging.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Create necessary directories for data and logs
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # Setup logging configuration
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting ACME Security Bot")

        # Initialize application settings
        settings = Settings()

        # Create and start the Slack bot instance
        bot = SecurityBot(settings)
        bot.start()

    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Failed to start security bot: {e}")
        raise


if __name__ == "__main__":
    # TODO: Move these to environment variables or secure configuration
    os.environ["SLACK_BOT_TOKEN"] = "xoxb-8993383066320-8963599940390-FpotGSTlfKmkQBBZlg1BTE1p"
    os.environ["SLACK_APP_TOKEN"] = "xapp-1-A08UBGQCDPG-8967316336421-dec00636c8d4954cc15d3d1035ec61ac4af6252468b135652279c38ca48df4d8"
    os.environ["SLACK_SIGNING_SECRET"] = "0771a55d937d2aaf70a3c6c3d500c3d0"
    main()