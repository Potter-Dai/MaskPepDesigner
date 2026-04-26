import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y=%m-%d %H:%M:%S',
)

from . import mmseqs_search, merge_features, make_template_features
