#!/usr/bin/env python3
"""
Automated News Livestream Intelligence System
Main entry point

Usage:
    python main.py [--config CONFIG_PATH] [--channel CHANNEL_NAME]
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_config, setup_logging, get_enabled_channels
from src.pipeline import NewsOrchestrator

logger = logging.getLogger(__name__)


async def main():
    """Main execution function"""

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Automated News Livestream Intelligence System"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config/settings.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--channel", type=str, default=None, help="Specific channel name to process"
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        sys.exit(1)

    # Setup logging
    setup_logging(config)
    logger.info("=" * 60)
    logger.info("Automated News Livestream Intelligence System")
    logger.info("=" * 60)

    # Get channels to process
    enabled_channels = get_enabled_channels(config)

    if not enabled_channels:
        logger.error("No enabled channels found in configuration")
        sys.exit(1)

    # Filter by channel name if specified
    if args.channel:
        enabled_channels = [
            ch for ch in enabled_channels if ch.get("name") == args.channel
        ]
        if not enabled_channels:
            logger.error(f"Channel '{args.channel}' not found or not enabled")
            sys.exit(1)

    channel = enabled_channels[0]
    logger.info(f"Processing channel: {channel.get('name')}")

    # Create orchestrator
    try:
        orchestrator = NewsOrchestrator(config)
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        sys.exit(1)

    # Start processing
    try:
        await orchestrator.start(channel)
    except KeyboardInterrupt:
        logger.info("\nShutdown requested by user")
        await orchestrator.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

    logger.info("System shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
