# logger.py
import os
import sys
from loguru import logger

def config_logger(name: str) -> bool:
    """
    Configures the logger using loguru.

    Args:
        name (str): Name for the log file.

    Returns:
        bool: True if the configuration is successful.
    """
    if not os.path.exists('logs'):
        os.mkdir('logs')

    log_path = f"logs/{name}.log"

    # 기존 로거 설정 제거
    logger.remove()

    # 콘솔 출력 설정
    logger.add(
        sink=sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level}</level> | <level>{message}</level>",
        level="DEBUG",
        colorize=True,
    )

    # 파일 출력 설정
    logger.add(
        sink=log_path,
        rotation="50 MB",
        retention="10 days",
        compression="gz",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
        level="INFO"
    )

    return True
