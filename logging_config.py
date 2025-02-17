import logging

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.DEBUG)

# Console handler for outputting logs to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)

# Create formatters and add it to handlers
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_format = logging.Formatter('%(levelname)s - %(message)s')
file_handler.setFormatter(file_format)
console_handler.setFormatter(console_format)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)
