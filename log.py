import logging
import os


def setup_logger(level=None):

    if level is None:
        if int(os.getenv('AIOPS_LLM_DEBUG', '0').strip()):
            level = logging.DEBUG
        else:
            level = logging.INFO

    _logger = logging.getLogger('aiops_llm_logger')
    _logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(filename)s - %(lineno)d - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    _logger.addHandler(handler)
    return _logger


logger = setup_logger()
