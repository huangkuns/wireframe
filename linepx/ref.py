import os
from pathlib import Path

root_dir = Path(os.getcwd()) / '..'
data_root = root_dir / 'data'

ext = '.pkl'
input_size = 320
data_folder = data_root / 'linepx' / 'processed'

cacheFile = data_root / 'train_cache.ptar'
