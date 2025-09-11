# %%
# Standard Library Imports
import os
import sys
import time
import random
import _pickle as pickle
import logging

import yaml
from datetime import datetime

# Third-party Imports
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset  # Ensuring Dataset is imported once
import numpy as np
import pandas as pd
import sklearn.metrics as skm
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter
from MulticoreTSNE import MulticoreTSNE as TSNE
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm.notebook import tqdm
from joblib import parallel_backend  # Added from second block of imports
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, fowlkes_mallows_score, calinski_harabasz_score  # Added adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer  # Added from second block of imports
from sklearn.decomposition import TruncatedSVD  # Added from second block of imports
from torch.utils.data import DataLoader

# Local imports
sys.path.insert(0, '/home/shared/jlian/EHR_image')
from general_model_newCutCPRD.ModelPkg.MLMRaw import BertConfig, BertModel, BertAgePredictor
from general_model_newCutCPRD.ModelPkg.DataProc import *
from general_model_newCutCPRD.pytorch_pretrained_bert import optimizer
from general_model_newCutCPRD.ModelPkg import utils


# Set up logging function as defined previously.
def setup_logging(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

# ================================
# Configuration and Model Setup
# ================================
file_config = {
    'vocab': '',
    'yearVocab':  '',
    'trained_model': '',
    'data_path': '',
}

global_params = {
    'batch_size': 512,
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'output_dir': '',
    'output_name': '',
    'min_visit': 1,
    'max_len_seq': 256,
    'max_age': 110,
    'age_year': False,
    'age_symbol': None,
    'inc_age': True,       # whether to incorporate age info; used for zeroing out age_ids
    'inc_seg': True        # other flags as needed
}

# Load additional vocabulary and configuration objects.
import general_model_newCutCPRD.ModelPkg.utils as utils
YearVocab = utils.load_obj(file_config['yearVocab'])
BertVocab = utils.load_obj(file_config['vocab'])

def extend_vocab_with_numbers(vocab, start=0, end=100):
    # Get the current maximum index value in the vocab
    current_max = max(vocab['token2idx'].values())
    
    # Loop over the desired numeric tokens
    for i in range(start, end + 1):
        token = str(i)  # create token from the number
        if token not in vocab['token2idx']:
            current_max += 1  # assign the next available index
            vocab['token2idx'][token] = current_max
            vocab['idx2token'][current_max] = token
    return vocab

BertVocab = extend_vocab_with_numbers(BertVocab, start=0, end=100)

print('len_vocab', len(BertVocab['token2idx']))

# Assume age vocabulary function is available in utils
ageVocab, _ = utils.age_vocab(max_age=global_params['max_age'], year=global_params['age_year'], symbol=global_params['age_symbol'])

# Set up model configurations and load the pretrained model.
from general_model_newCutCPRD.ModelPkg.MLMRaw import BertConfig, BertModel, BertAgePredictor

model_config = {
    'vocab_size': len(BertVocab['token2idx'].keys()), # number of disease + symbols for word embedding
    'hidden_size': 256, # word embedding and seg embedding hidden size
    'seg_vocab_size': 2, # number of vocab for seg embedding
    'age_vocab_size': len(ageVocab.keys()), # number of vocab for age embedding
    'year_vocab_size': len(YearVocab['token2idx'].keys()), # number of vocab for age embedding
    'max_position_embedding': global_params['max_len_seq'], # maximum number of tokens
    'hidden_dropout_prob': 0.1, # dropout rate
    'num_hidden_layers': 4, # number of multi-head attention layers required
    'num_attention_heads': 4, # number of attention heads
    'attention_probs_dropout_prob': 0.1, # multi-head attention dropout rate
    'intermediate_size': 1024, # the size of the "intermediate" layer in the transformer encoder
    'hidden_act': 'gelu', # The non-linear activation function in the encoder and the pooler "gelu", 'relu', 'swish' are supported
    'initializer_range': 0.02, # parameter weight initializer range,
    'yearOn':False,
    'year_vocab_size': len(YearVocab['token2idx'].keys()),
    'concat_embeddings':False,
}

model = BertModel(BertConfig(model_config))
# Assume toLoad is defined as in your script:
def toLoad(model, filepath, custom=None):
    pretrained_dict = torch.load(filepath, map_location='cpu')
    new_state_dict = {key.replace("bert.", ""): value for key, value in pretrained_dict.items()}
    modeld = model.state_dict()
    if custom is None:
        pretrained_dict = {k: v for k, v in new_state_dict.items() if k in modeld}
    else:
        pretrained_dict = {k: v for k, v in new_state_dict.items() if k in modeld and k not in custom}
    modeld.update(pretrained_dict)
    model.load_state_dict(modeld)
    return model

age_model = BertAgePredictor(model)
age_model.load_state_dict(torch.load(file_config['trained_model']))
age_model.to(global_params['device'])
age_model.eval()

# =====================================
# Inference-Only: Process the Inference Data
# =====================================
# In this example, we assume the full dataset at file_config['data_path'] is used for inference.
df_inference = pd.read_parquet(file_config['data_path'])
# Convert to numeric where needed and compute label as in your original script.
df_inference['label'] = pd.to_numeric(df_inference['label'], errors='coerce')
df_inference['baseline_age'] = pd.to_numeric(df_inference['baseline_age'], errors='coerce')
df_inference.rename(columns={'label': 'phenoage'}, inplace=True)
df_inference['label'] = df_inference['phenoage'] - df_inference['baseline_age']

# Filter based on visit count requirement.
df_inference = df_inference[df_inference['code'].apply(lambda x: list(x).count('SEP') >= global_params['min_visit'])]
df_inference = df_inference.reset_index(drop=True)

# ================================
# Create an Inference Dataset and DataLoader
# ================================
# Assume you have defined a function "create_seq_loader" to generate your dataset.
def create_seq_loader(df):
    # This function should create your dataset using your defined parameters.
    # Here we assume SeqLoaderAge is available.
    from general_model_newCutCPRD.ModelPkg.DataProc import SeqLoaderAge
    return SeqLoaderAge(
        token2idx=BertVocab['token2idx'],
        dataframe=df,
        max_len=global_params['max_len_seq'],
        max_age=global_params['max_age'],
        year=global_params['age_year'],
        age_symbol=global_params['age_symbol'],
        year2idx=YearVocab['token2idx']
    )

# Create the inference dataset and dataloader.
inference_dataset = create_seq_loader(df_inference)
inference_loader = DataLoader(inference_dataset, batch_size=global_params['batch_size'], shuffle=False)

# ================================
# Set up Logging and Output Directory
# ================================
output_subdir = os.path.join(global_params['output_dir'], "CPRD_randinit_bert_small_res_exc_age_exc_seg_exc_concat_time_min_visit1_EFI")
os.makedirs(output_subdir, exist_ok=True)
log_file = os.path.join(output_subdir, "inference_log.txt")
logger = setup_logging(log_file)
logger.info("Starting inference on data from: " + file_config['data_path'])
logger.info(f"Output directory: {output_subdir}")

# ================================
# Run Inference
# ================================
device = global_params['device']
all_preds = []
age_model.eval()
with torch.no_grad():
    for batch in inference_loader:
        # Unpack the batch; the structure should match your dataset.
        age_ids, year_ids, input_ids, posi_ids, segment_ids, attMask, labels = batch
        
        # For inference, set age_ids and year_ids to zeros if specified.
        if not global_params['inc_age']:
            age_ids = torch.zeros_like(age_ids)
        year_ids = torch.zeros_like(year_ids)
        segment_ids = torch.zeros_like(segment_ids)
        
        # Move tensors to the device.
        input_ids = input_ids.to(device)
        age_ids = age_ids.to(device)
        segment_ids = segment_ids.to(device)
        posi_ids = posi_ids.to(device)
        year_ids = year_ids.to(device)
        attMask = attMask.to(device)
        labels = labels.to(device)
        
        # Forward pass. (Labels are provided but not used for prediction.)
        _, age_pred = age_model(input_ids, age_ids, segment_ids, posi_ids, year_ids, attMask, labels)
        all_preds.extend(age_pred.cpu().numpy())

# Verify prediction length matches the input.
assert len(all_preds) == len(df_inference), "Mismatch in number of predictions and input records."

# Add predictions to the DataFrame.
df_inference['predicted'] = all_preds
df_inference['predicted_phenoage'] = df_inference['baseline_age'] + df_inference['predicted']

# Save predictions to CSV (selecting relevant columns).
output_csv = os.path.join(output_subdir, "age_predictions.csv")
df_inference[['patid', 'predicted', 'baseline_age', 'phenoage', 'label']].to_csv(output_csv, index=False)
logger.info(f"Saved predictions to {output_csv}")
print(f"Saved predictions to {output_csv}")
