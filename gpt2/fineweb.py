"""
This downloads fineweb-edu from Hugging Face dataset, it will preprocess and pretokenize all the data, and
save the data shards to a folder on local disk
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm
