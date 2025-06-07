"""
Script to setup the database and run initial migrations
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.repository import SecurityRequestRepository
from config.settings import Settings


def main():
    """Setup database"""
    settings = Settings()

    # Create data directory if it doesn't exist
    Path("data").mkdir(exist_ok=True)

    # Initialize repository (this will create tables)
    repo = SecurityRequestRepository(settings.DATABASE_PATH)

    print(f"Database initialized at: {settings.DATABASE_PATH}")

    # Print statistics
    stats = repo.get_statistics()
    print(f"Current statistics: {stats}")


if __name__ == "__main__":
    main()
