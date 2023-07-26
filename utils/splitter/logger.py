import logging 

log = logging.getLogger("main_logger")
logging.basicConfig(filename="main.log",
                    level=logging.DEBUG,
                    format='%(asctime)s  -  %(levelname)s - %(message)s')

# Add a handler to write log messages to a file
#file_handler = logging.FileHandler("main.log")
#file_handler.setLevel(logging.DEBUG)
#log.addHandler(file_handler)