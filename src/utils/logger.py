import logging
import os
from datetime import datetime
from pythonjsonlogger import jsonlogger


def get_logger(name: str, to_file: bool = True, json_format: bool = False) -> logging.Logger:
    """
    Configura y devuelve un logger con formato profesional.

    Args:
        name (str): Nombre del módulo que crea el logger (usualmente __name__).
        to_file (bool): Si True, también escribe en un archivo rotativo.
        json_format (bool): Si True, usa formato JSON (ideal para Jenkins/Docker logs).
    Returns:
        logging.Logger: Logger configurado.
    """
    logger = logging.getLogger(name)

    # Evita duplicar handlers si el logger ya existe
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # === FORMATO ===
    if json_format:
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # === CONSOLA ===
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # === ARCHIVO ===
    if to_file:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(
            log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
