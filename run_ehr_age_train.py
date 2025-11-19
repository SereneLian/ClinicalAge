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
from general_model_newCutCPRD.ModelPkg.MLMRaw import BertConfig, BertModel, BertAgePredictor
from general_model_newCutCPRD.ModelPkg.DataProc import *
from general_model_newCutCPRD.pytorch_pretrained_bert import optimizer
from general_model_newCutCPRD.ModelPkg import utils

print('starting run....')

# %%
seed = 1234
def set_all_seeds(seed):
    """Set all seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
def toLoad(model, filepath, custom=None):
    pre_bert = filepath

    pretrained_dict = torch.load(pre_bert, map_location='cpu')

    new_state_dict = {}
    for key, value in pretrained_dict.items():
        # Remove the prefix "bert." if present
        new_key = key.replace("bert.", "")
        new_state_dict[new_key] = value

    modeld = model.state_dict()
    # 1. filter out unnecessary keys
    if custom == None:
        pretrained_dict = {k: v for k, v in new_state_dict.items() if k in modeld}
    else:
        pretrained_dict = {k: v for k, v in new_state_dict.items() if k in modeld and k not in custom}
    print(pretrained_dict.keys())
    modeld.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(modeld)
    return model

def setup_logging(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    # Create handlers for console and file.
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

# %%
file_config = {
    'vocab': '',
    # 'yearVocab':  '',
    'pretrained_model': '',
    'data_path': '',
}

global_params = {
    'batch_size': 128,
    'gradient_accumulation_steps': 1,
    'device': 'cuda:0',
    'output_dir':'',
    'output_name': '',
    'save_model': True,
    'max_len_seq': 250,
    'max_age': 110,
    'age_year': False,
    'age_symbol': None,
    'min_visit': 5,
    'inc_age': False,
    'inc_seg': True
}

# YearVocab = utils.load_obj(file_config['yearVocab'])
create_folder(global_params['output_dir'])
BertVocab = utils.load_obj(file_config['vocab'])
print('len_vocab', len(BertVocab['token2idx']))

# ageVocab, _ = utils.age_vocab(max_age=global_params['max_age'], year=global_params['age_year'], symbol=global_params['age_symbol'])

model_config = {
    'vocab_size': len(BertVocab['token2idx'].keys()), # number of disease + symbols for word embedding
    'hidden_size': 150, # word embedding and seg embedding hidden size
    'seg_vocab_size': 2, # number of vocab for seg embedding
    # 'age_vocab_size': len(ageVocab.keys()), # number of vocab for age embedding
    # 'year_vocab_size': len(YearVocab['token2idx'].keys()), # number of vocab for age embedding
    'max_position_embedding': global_params['max_len_seq'], # maximum number of tokens
    'hidden_dropout_prob': 0.1, # dropout rate
    'num_hidden_layers': 6, # number of multi-head attention layers required
    'num_attention_heads': 6, # number of attention heads
    'attention_probs_dropout_prob': 0.1, # multi-head attention dropout rate
    'intermediate_size': 108, # the size of the "intermediate" layer in the transformer encoder
    'hidden_act': 'gelu', # The non-linear activation function in the encoder and the pooler "gelu", 'relu', 'swish' are supported
    'initializer_range': 0.02, # parameter weight initializer range,
    'yearOn':False,
    # 'year_vocab_size': len(YearVocab['token2idx'].keys()),
    'concat_embeddings':False,

}

model = BertModel(BertConfig(model_config))
# model = toLoad(model, file_config['pretrained_model'])
age_model = BertAgePredictor(model)

# %% [markdown]
# process and save data into train/valid/test split (7:1:2) ratio

# %%
processed_data_file = f'ehr_age_processed_df_upsample_res_visit{global_params["min_visit"]}.pkl'
if os.path.exists(processed_data_file):
    with open(processed_data_file, 'rb') as f:
        processed_data = pickle.load(f)
    train_df = processed_data['train']
    valid_df = processed_data['valid']
    test_df = processed_data['test']
    print("Loaded processed DataFrame splits.")
else:
    # Read and shuffle the DataFrame with a fixed random seed.
    df = pd.read_parquet(file_config['data_path'])
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df['baseline_age'] = pd.to_numeric(df['baseline_age'], errors='coerce')
    df.rename(columns={'label': 'phenoage'}, inplace=True)
    df['label'] = df['phenoage'] - df['baseline_age']
    df = df[df['code'].apply(lambda x: list(x).count('SEP') >= global_params['min_visit'] + 1)]
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df_shuffled)
    train_end = int(0.7 * n)
    valid_end = int(0.8 * n)
    
    # Reset indices for each split.
    train_df = df_shuffled.iloc[:train_end].reset_index(drop=True)
    valid_df = df_shuffled.iloc[train_end:valid_end].reset_index(drop=True)
    test_df = df_shuffled.iloc[valid_end:].reset_index(drop=True)
    
    # Specify the age ranges you want to upsample (excluding "30-39")
    upsample_groups = ['60-69', '50-59', '40-49', '70-79']

    # Separate the DataFrame into groups to upsample and those to leave unchanged
    df_to_upsample = train_df[train_df['Age_range'].isin(upsample_groups)]
    df_not_upsample = train_df[~train_df['Age_range'].isin(upsample_groups)]

    # Get the maximum count among the groups to upsample
    group_counts = df_to_upsample['Age_range'].value_counts()
    max_count = group_counts.max()  # For example, 142887 from the "60-69" group

    # Upsample only the specified groups
    upsampled_df = df_to_upsample.groupby('Age_range', group_keys=False).apply(
        lambda x: x.sample(max_count, replace=True) if len(x) < max_count else x
    ).reset_index(drop=True)

    # Combine the upsampled data with the groups you don't want to change (e.g., "30-39")
    train_df = pd.concat([upsampled_df, df_not_upsample]).reset_index(drop=True)
    print(train_df['Age_range'].value_counts())

    processed_data = {'train': train_df, 'valid': valid_df, 'test': test_df}
    with open(processed_data_file, 'wb') as f:
        pickle.dump(processed_data, f)
    print("Processed and saved DataFrame splits.")

# %%
def create_seq_loader(df):
    return SeqLoaderAge(
        token2idx=BertVocab['token2idx'],
        dataframe=df,
        max_len=global_params['max_len_seq'],
        max_age=global_params['max_age'],
        year=global_params['age_year'],
        age_symbol=global_params['age_symbol'],
        # year2idx=YearVocab['token2idx']
    )

splits = {'train': train_df, 'valid': valid_df, 'test': test_df}
datasets = {split: create_seq_loader(df) for split, df in splits.items()}

# Then create dataloaders
train_loader = DataLoader(datasets['train'], batch_size=global_params['batch_size'], shuffle=True)
valid_loader = DataLoader(datasets['valid'], batch_size=global_params['batch_size']*8, shuffle=False)
test_loader  = DataLoader(datasets['test'], batch_size=global_params['batch_size']*8, shuffle=False)

# %%
hyperparams = {
    'lr': 1e-5,
    'batch_size': global_params['batch_size'],
    'num_epochs': 10,
}
# Create a folder name that includes some hyperparameter values.
save_dirname = f"lr{hyperparams['lr']}_bs{hyperparams['batch_size']}_upsample_randinit_res_exc_age_exc_seg_exc_concat_min_visit{global_params['min_visit']}"
output_subdir = os.path.join(global_params['output_dir'], save_dirname)
os.makedirs(output_subdir, exist_ok=True)

# Set up logging to a file in the dynamic folder.
log_file = os.path.join(output_subdir, "log.txt")
logger = setup_logging(log_file)
logger.info("Starting age prediction training with configuration:")
logger.info(f"Hyperparameters: {hyperparams}")
logger.info(f"Output directory: {output_subdir}")

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(age_model.parameters(), lr=hyperparams['lr'])
device = global_params['device'] if torch.cuda.is_available() else 'cpu'
age_model.to(device)

# # -------------------------------
# # 4. Training Setup for Age Prediction
# # -------------------------------

num_epochs = hyperparams['num_epochs']
best_valid_rmse = float('inf')
best_model_state = None

global_step = 0

for epoch in range(num_epochs):
    age_model.train()
    train_losses = []
    for batch in train_loader:
        # Unpack batch
        age_ids, year_ids, input_ids, posi_ids, segment_ids, attMask, labels = batch
        
        # For age prediction we set age_ids and year_ids to zeros.
        if not global_params['inc_age']:
            age_ids = torch.zeros_like(age_ids)
        year_ids = torch.zeros_like(year_ids)
        segment_ids = torch.zeros_like(segment_ids)
        
        # Move tensors to device.
        input_ids = input_ids.to(device)
        age_ids = age_ids.to(device)
        segment_ids = segment_ids.to(device)
        posi_ids = posi_ids.to(device)
        year_ids = year_ids.to(device)
        attMask = attMask.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        loss, age_pred = age_model(input_ids, age_ids, segment_ids, posi_ids, year_ids, attMask, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        global_step += 1
        # Log every 100 steps.
        if global_step % 100 == 0:
            logger.info(f"Epoch {epoch+1} Step {global_step}: Loss = {loss.item():.4f}")
    
    avg_train_loss = np.mean(train_losses)
    
    # Validation loop
    age_model.eval()
    valid_preds = []
    valid_labels = []
    with torch.no_grad():
        for batch in valid_loader:
            age_ids, year_ids, input_ids, posi_ids, segment_ids, attMask, labels = batch
            if not global_params['inc_age']:
                age_ids = torch.zeros_like(age_ids)
            year_ids = torch.zeros_like(year_ids)
            segment_ids = torch.zeros_like(segment_ids)
            
            input_ids = input_ids.to(device)
            age_ids = age_ids.to(device)
            segment_ids = segment_ids.to(device)
            posi_ids = posi_ids.to(device)
            year_ids = year_ids.to(device)
            attMask = attMask.to(device)
            labels = labels.to(device)
            
            _, age_pred = age_model(input_ids, age_ids, segment_ids, posi_ids, year_ids, attMask, labels)
            valid_preds.extend(age_pred.cpu().numpy())
            valid_labels.extend(labels.cpu().numpy())
    
    # Compute validation metrics
    valid_rmse = np.sqrt(skm.mean_squared_error(valid_labels, valid_preds))
    valid_mae = skm.mean_absolute_error(valid_labels, valid_preds)
    
    logger.info(f"Epoch {epoch+1}/{num_epochs} | Avg Train Loss: {avg_train_loss:.4f} | Valid MAE: {valid_mae:.4f} | Valid RMSE: {valid_rmse:.4f}")
    print(f"Epoch {epoch+1}/{num_epochs} | Avg Train Loss: {avg_train_loss:.4f} | Valid MAE: {valid_mae:.4f} | Valid RMSE: {valid_rmse:.4f}")
    
    # Save best model based on lowest RMSE.
    if valid_rmse < best_valid_rmse:
        best_valid_rmse = valid_rmse
        best_model_state = age_model.state_dict()
        save_path = os.path.join(output_subdir, "best_age_model.pt")
        torch.save(best_model_state, save_path)
        logger.info(f"Best model saved at epoch {epoch+1} with Valid RMSE: {valid_rmse:.4f}")
        print(f"Best model saved at epoch {epoch+1} with Valid RMSE: {valid_rmse:.4f}")


# -------------------------------
# 5. Test Evaluation with Confidence Intervals
# -------------------------------
# Load best model state
best_model_path = os.path.join(output_subdir, "best_age_model.pt")
print(f"Loading best model from {best_model_path}")
age_model.load_state_dict(torch.load(best_model_path))
age_model.eval()
test_preds = []
test_labels = []
with torch.no_grad():
    for batch in test_loader:
        age_ids, year_ids, input_ids, posi_ids, segment_ids, attMask, labels = batch

        if not global_params['inc_age']:
            age_ids = torch.zeros_like(age_ids)
        year_ids = torch.zeros_like(year_ids)
        segment_ids = torch.zeros_like(segment_ids)
        
        input_ids = input_ids.to(device)
        age_ids = age_ids.to(device)
        segment_ids = segment_ids.to(device)
        posi_ids = posi_ids.to(device)
        year_ids = year_ids.to(device)
        attMask = attMask.to(device)
        labels = labels.to(device)
        
        _, age_pred = age_model(input_ids, age_ids, segment_ids, posi_ids, year_ids, attMask, labels)
        test_preds.extend(age_pred.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

test_preds = np.array(test_preds)
test_labels = np.array(test_labels)

# Add predictions to the test_df and save to CSV
test_df['predicted'] = test_preds
output_csv_test = os.path.join(output_subdir, "age_predictions_test.csv")
test_df[['patid', 'predicted', 'baseline_age', 'phenoage', 'label']].to_csv(output_csv_test, index=False)
print(f"Saved test predictions to {output_csv_test}")

# Compute the metrics
test_rmse = np.sqrt(skm.mean_squared_error(test_labels, test_preds))
test_mae = skm.mean_absolute_error(test_labels, test_preds)
test_df['predicted_phenoage'] = test_df['baseline_age'] + test_df['predicted']
# Compute the Pearson correlation between predicted_phenoage and phenoage
corr = test_df['predicted_phenoage'].corr(test_df['phenoage'])
logger.info(f"Pearson correlation (baseline_age + predicted vs phenoage): {corr:.4f}")
print(f"Pearson correlation (baseline_age + predicted vs phenoage): {corr:.4f}")
corr = test_df['baseline_age'].corr(test_df['phenoage'])
logger.info(f"Pearson correlation (baseline_age vs phenoage): {corr:.4f}")
print(f"Pearson correlation (baseline_age vs phenoage): {corr:.4f}")

# Define a simple bootstrap function to compute confidence intervals
def bootstrap_metric(metric_fn, y_true, y_pred, n_bootstraps=1000, alpha=0.05):
    bootstrapped_scores = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        indices = np.random.randint(0, n, n)
        score = metric_fn(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    lower = np.percentile(bootstrapped_scores, 100 * alpha/2)
    upper = np.percentile(bootstrapped_scores, 100 * (1 - alpha/2))
    return lower, upper

# Compute confidence intervals for MAE and RMSE
mae_lower, mae_upper = bootstrap_metric(skm.mean_absolute_error, test_labels, test_preds)
rmse_lower, rmse_upper = bootstrap_metric(lambda y_true, y_pred: np.sqrt(skm.mean_squared_error(y_true, y_pred)), 
                                          test_labels, test_preds)

logger.info(f"Test MAE: {test_mae:.4f} (95% CI: [{mae_lower:.4f}, {mae_upper:.4f}]) | "
            f"Test RMSE: {test_rmse:.4f} (95% CI: [{rmse_lower:.4f}, {rmse_upper:.4f}])")
print(f"Test MAE: {test_mae:.4f} (95% CI: [{mae_lower:.4f}, {mae_upper:.4f}])")
print(f"Test RMSE: {test_rmse:.4f} (95% CI: [{rmse_lower:.4f}, {rmse_upper:.4f}])")

df_inference = pd.read_parquet(file_config['data_path'])
df_inference['label'] = pd.to_numeric(df_inference['label'], errors='coerce')
df_inference['baseline_age'] = pd.to_numeric(df_inference['baseline_age'], errors='coerce')
df_inference.rename(columns={'label': 'phenoage'}, inplace=True)
df_inference['label'] = df_inference['phenoage'] - df_inference['baseline_age']
df_inference = df_inference[df_inference['code'].apply(lambda x: list(x).count('SEP') >= global_params['min_visit'])]
df_inference = df_inference.reset_index(drop=True)


# Create an inference dataset using the same create_seq_loader function.
inference_dataset = create_seq_loader(df_inference)
inference_loader = DataLoader(inference_dataset, batch_size=global_params['batch_size'], shuffle=False)

best_model_path = os.path.join(output_subdir, "best_age_model.pt")
print(f"Loading best model from {best_model_path}")
age_model.load_state_dict(torch.load(best_model_path))
age_model.to(device)
age_model.eval()

# Use the best model (already loaded in age_model) to make predictions.
all_preds = []
age_model.eval()
with torch.no_grad():
    for batch in inference_loader:
        age_ids, year_ids, input_ids, posi_ids, segment_ids, attMask, labels = batch
        # For inference, set age_ids and year_ids to zeros.
        if not global_params['inc_age']:
            age_ids = torch.zeros_like(age_ids)
        year_ids = torch.zeros_like(year_ids)
        segment_ids = torch.zeros_like(segment_ids)
        
        input_ids = input_ids.to(device)
        age_ids = age_ids.to(device)
        segment_ids = segment_ids.to(device)
        posi_ids = posi_ids.to(device)
        year_ids = year_ids.to(device)
        attMask = attMask.to(device)
        labels = labels.to(device)  # Move labels to device
        
        # Forward pass (labels are not needed for prediction, but we pass them anyway).
        _, age_pred = age_model(input_ids, age_ids, segment_ids, posi_ids, year_ids, attMask, labels)
        all_preds.extend(age_pred.cpu().numpy())

# Ensure that the number of predictions matches the number of rows.
assert len(all_preds) == len(df_inference), "Number of predictions does not match number of rows."

# Add the predictions as a new column.
df_inference['predicted'] = all_preds

# Save to CSV with only the columns: patid, predicted_age, and phenoage (assumed to be in 'label').
output_csv = os.path.join(output_subdir, "age_predictions.csv")
df_inference[['patid', 'predicted', 'baseline_age', 'phenoage', 'label']].to_csv(output_csv, index=False)
print(f"Saved predictions to {output_csv}")