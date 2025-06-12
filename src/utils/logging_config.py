import logging
import os


def setup_logging(log_file="logs/main_log.log"):
    """
    Configures logging with timestamp, level, and message.
    Matches your desired format.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Remove existing handlers
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",  # <-- NO %(name)s
        datefmt="%Y-%m-%d %H:%M:%S",  # milliseconds
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
