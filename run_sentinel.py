"""
SOLAT Sentinel Entry Point

The Sentinel is the headless backend process responsible for:
- Data ingestion via adapters
- Ichimoku signal generation
- Trade execution logic (paper trading)
- Evolutionary scoring and asset ranking
- Database writes (primary writer role)

This script:
1. Initializes the WAL-enabled SQLite database
2. Loads initial assets from the seed file
3. Starts the Sentinel event loop for continuous market scanning
"""

import logging
import sys

from src.database.repository import init_db
from src.core.engine import Sentinel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/sentinel.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main() -> None:
    """
    Main entry point for the SOLAT Sentinel trading engine.

    Flow:
    1. Initialize database with WAL mode
    2. Initialize Sentinel orchestrator
    3. Load initial assets from seed file
    4. Start the continuous event loop
    """
    logger.info("=" * 60)
    logger.info("SOLAT SENTINEL STARTING")
    logger.info("=" * 60)

    try:
        # Step 1: Initialize the database with WAL mode
        logger.info("Initializing database...")
        init_db()
        logger.info("Database initialized in WAL Mode")

        # Step 2: Initialize Sentinel orchestrator
        logger.info("Initializing Sentinel engine...")
        sentinel = Sentinel()

        # Step 3: Load initial assets from seed file
        logger.info("Loading initial assets from seed file...")
        sentinel.initialize_assets()

        # Step 4: Start the continuous event loop
        logger.info("Starting market scanning loop...")
        sentinel.run_loop(interval=60)

    except Exception as e:
        logger.error(f"Fatal error in Sentinel startup: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
