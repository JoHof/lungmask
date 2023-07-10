import logging
import sys

logger = logging.getLogger("lungmask")
logger.setLevel(logging.INFO)
logger.propagate = False
formatter = logging.Formatter(
    fmt="lungmask %(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)
logger.addHandler(handler)
