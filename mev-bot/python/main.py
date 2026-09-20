import asyncio
import logging
from src.core.engine import ArbitrageEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    """Main entry point for Corridor Predator"""
    logger = logging.getLogger(__name__)
    logger.info("启动 Corridor Predator MEV 套利机器人...")

    engine = ArbitrageEngine()

    try:
        # Initialize engine
        await engine.initialize()

        # Start the arbitrage loop
        await engine.start()

    except KeyboardInterrupt:
        logger.info("接收到停止信号...")
    except Exception as e:
        logger.error(f"主循环出错: {e}")
    finally:
        await engine.stop()
        logger.info("机器人已停止")

if __name__ == "__main__":
    asyncio.run(main())
