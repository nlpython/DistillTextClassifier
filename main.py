from loguru import logger

logger.add("file_{time}.log")
logger.warning("Logging started")


