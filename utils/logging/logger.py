import logging 

# Create or retrieve a logger
log = logging.getLogger("main_logger")

# Set the logging level
log.setLevel(logging.DEBUG)

# Create a file handler
file_handler = logging.FileHandler("main.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s  -  %(levelname)s - %(message)s'))
file_handler.setLevel(logging.DEBUG)

# Create a stream handler for output to console
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s  -  %(levelname)s - %(message)s'))
console_handler.setLevel(logging.DEBUG)

# Add the handlers to the logger
log.addHandler(file_handler)
log.addHandler(console_handler)