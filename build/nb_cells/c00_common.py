# Cells: notebook title + global configuration + shared utilities

TITLE_MD = r"""# EcoSort: Intelligent Waste Management Assistant — Summative Lab (NNs & Similar Models)

**Part 1:** Dataset exploration and preparation  
**Part 2:** Waste material classification with a CNN (transfer learning)  
**Part 3:** Waste description classification (text)  
**Part 4:** Recycling instruction generation with RAG  
**Part 5:** Integrated waste management assistant  

Authoring environment: Windows | Python 3.12 | CPU only (16 GB RAM)
"""

CONFIG_CODE = r'''# %% [markdown]
# ## 0. Global configuration
# We centralise every path, class name, and hyper-parameter here so the whole
# notebook stays consistent and reproducible.

import os
import sys
import json
import re
import math
import random
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

# Quiet noisy TF startup logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import tensorflow as tf
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    else:
        print('INFO: No GPU detected - running on CPU.')
except Exception as e:
    print('GPU config skipped:', e)
tf.random.set_seed(42)

import torch
torch.manual_seed(42)

# ---------------------------------------------------------------- paths
BASE = os.getcwd()
DATA = os.path.join(BASE, 'data')          # derived data (splits, features)
FIG = os.path.join(DATA, 'figures')        # saved plots
MODELS = os.path.join(BASE, 'models')      # saved model artifacts
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

def find_realwaste_root():
    cands = [
        os.path.join(BASE, 'realwaste', 'realwaste-main', 'RealWaste'),
        os.path.join(BASE, 'RealWaste'),
        os.path.join(BASE, 'data', 'RealWaste'),
    ]
    for c in cands:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError('RealWaste image root not found. Please place the '
                            '"RealWaste" class folders next to the data.')
IMAGE_ROOT = find_realwaste_root()
TEXT_CSV = os.path.join(BASE, 'waste_descriptions.csv')
POLICY_JSON = os.path.join(BASE, 'waste_policy_documents.json')

CLASSES = sorted([d for d in os.listdir(IMAGE_ROOT)
                  if os.path.isdir(os.path.join(IMAGE_ROOT, d))])
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}
print(f'IMAGE_ROOT  : {IMAGE_ROOT}')
print(f'N classes   : {len(CLASSES)}')
print(f'Classes     : {CLASSES}')
print(f'TensorFlow  : {tf.__version__}   torch: {torch.__version__}')
'''

UTILS_CODE = r'''# %% [markdown]
# ### Shared helpers (used everywhere below)

def seed_everything(s=42):
    random.seed(s); np.random.seed(s); tf.random.set_seed(s); torch.manual_seed(s)

def save_fig(fig, name, show=True, dpi=110, tight=True):
    """Save the figure and (for the notebook output) display it."""
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=dpi, bbox_inches='tight' if tight else None)
    if show:
        plt.show()
    plt.close(fig)
    return p

def clf_metrics(y_true, y_pred, labels):
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, classification_report,
                                 confusion_matrix)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'report': classification_report(y_true, y_pred, target_names=labels, zero_division=0, digits=3),
        'confusion': confusion_matrix(y_true, y_pred, labels=range(len(labels))),
    }

def plot_confusion(cm, labels, title='Confusion matrix', cmap='Blues'):
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=labels,
                yticklabels=labels, ax=ax, cbar=False)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    return fig
'''