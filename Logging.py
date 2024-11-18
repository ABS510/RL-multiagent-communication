import logging
from colorama import Fore, Style, init
from datetime import datetime

init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    LOG_COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA + Style.BRIGHT,
    }

    def format(self, record):
        log_color = self.LOG_COLORS.get(record.levelno, "")
        structure = f"%(asctime)s {log_color}%(name)s [%(levelname)s]{Style.RESET_ALL}"
        formatter = logging.Formatter(
            fmt=f"{structure} %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        return formatter.format(record)


def setup_logger(name, log_file=None, level=logging.DEBUG):
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Logs to console
    console_handler = logging.StreamHandler()
    formatter = ColoredFormatter()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Logs to a file
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


# Example usage
if __name__ == "__main__":
    # Initialize logger
    logger = setup_logger(
        "MyApp", "log_file_name.log"
    )  # If set to None, it will not log to a file

    # Log messages at different levels
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
