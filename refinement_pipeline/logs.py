from loguru import logger


def log_forced_run() -> None:
    logger.info("Running refinement because stage was explicitly requested.")
