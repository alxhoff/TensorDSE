import logging
import json

def ReadJSON(file_path: str):
    with open(file_path) as fin:
        return json.load(fin)
        
def LoggerInit(filename='optimizer.log'):
    logging.basicConfig(filename=filename,
                        level=logging.DEBUG,
                        format='%(asctime)s  -  %(levelname)s - %(message)s')
    return logging.getLogger(__name__)