import logging

logger = logging.getLogger(__name__)

def flatten_images(x):
    logger.info("Flattening images to vectors of shape")
    return x.reshape((x.shape[0], -1))