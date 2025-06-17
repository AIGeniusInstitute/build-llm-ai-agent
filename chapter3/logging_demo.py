import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Agent启动")
logger.error("发生错误", exc_info=True)