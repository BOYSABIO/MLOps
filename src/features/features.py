import logging

logger = logging.getLogger(__name__)


def flatten_images(picture):
    """
    Flattens 2D into 1D vectors.
    """
    logger.info("Flattening images to vectors of shape")
    return picture.reshape((picture.shape[0], -1))
