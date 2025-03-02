from logging import DEBUG, Formatter, Logger, StreamHandler, getLogger


def get_logger() -> Logger:
    logger = getLogger(__name__)
    handler = StreamHandler()
    handler.setLevel(DEBUG)
    format = "[%(levelname)s]\t%(asctime)s\t[%(filename)s:%(lineno)d]\t%(message)s"
    handler.setFormatter(Formatter(format))
    logger.setLevel(DEBUG)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
