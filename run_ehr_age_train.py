# %%
# ------------------------------------------------------------------
# 1. IMPORTS
# ------------------------------------------------------------------
import os
import sys
import random
import logging
import pickle
import numpy as np
import pandas as pd
import sklearn.metrics as skm
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler, BertLayerNorm

# %%
# ------------------------------------------------------------------
# 2. UTILITY FUNCTIONS (Integrated from local utils)
# ------------------------------------------------------------------

def set_all_seeds(seed):
    """Set all seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_obj(name):
    """Load a pickle object."""
    if not name.endswith('.pkl'):
        name += '.pkl'
    with open(name, 'rb') as f:
        return pickle.load(f)

def age_vocab(max_age, year=False, symbol=None):
    """Creates vocab for age tokens."""
    age2idx = {}
    idx2age = {}
    if symbol is None:
        symbol = ['PAD', 'UNK']

    for i in range(len(symbol)):
        age2idx[str(symbol[i])] = i
        idx2age[i] = str(symbol[i])

    if year:
        for i in range(max_age):
            age2idx[str(i)] = len(symbol) + i
            idx2age[len(symbol) + i] = str(i)
    else:
        # Your logic: max_age * 12 (months)
        for i in range(max_age * 12):
            age2idx[str(i)] = len(symbol) + i
            idx2age[len(symbol) + i] = str(i)

    return age2idx, idx2age

def seq_padding(tokens, max_len, token2idx=None, symbol='PAD'):
    """Pads sequences to max_len."""
    seq = []
    token_len = len(tokens)
    for i in range(max_len):
        if token2idx is None:
            if i < token_len:
                seq.append(tokens[i])
            else:
                seq.append(symbol)
        else:
            if i < token_len:
                # Get token ID, default to UNK if missing
                # If token2idx is just a dict, use .get
                try:
                    seq.append(token2idx.get(str(tokens[i]), token2idx.get('UNK', 1)))
                except:
                     seq.append(token2idx.get(tokens[i], token2idx.get('UNK', 1)))
            else:
                seq.append(token2idx.get(symbol, 0))
    return seq

def code2index(tokens, token2idx):
    """Converts code tokens to indices."""
    output_tokens = []
    for i, token in enumerate(tokens):
        output_tokens.append(token2idx.get(token, token2idx.get('UNK', 1)))
    return tokens, output_tokens

def position_idx(tokens, symbol='SEP'):
    """Creates position indices based on visits separated by SEP."""
    pos = []
    flag = 0
    for token in tokens:
        if token == symbol:
            pos.append(flag)
            flag += 1
        else:
            pos.append(flag)
    return pos

def index_seg(tokens, symbol='SEP'):
    """Creates segment indices (0/1) flipping at every SEP."""
    flag = 0
    seg = []
    for token in tokens:
        if token == symbol:
            seg.append(flag)
            flag = 1 - flag 
        else:
            seg.append(flag)
    return seg

def setup_logging(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

# %%
# ------------------------------------------------------------------
# 3. DATASET CLASS (Integrated from DataProc)
# ------------------------------------------------------------------

class SeqLoaderAge(Dataset):
    def __init__(self, token2idx, dataframe, max_len, max_age=110, year=False, age_symbol=None, year2idx=None):
        self.vocab = token2idx
        self.max_len = max_len
        self.code = dataframe.code.values
        self.age = dataframe.age.values
        # Handle cases where year might not exist in dataframe
        self.year = dataframe.year.values if 'year' in dataframe.columns else [[] for _ in range(len(dataframe))]
        self.label = dataframe.label.values  # phenoage
        
        self.age2idx, _ = age_vocab(max_age, year, symbol=age_symbol)
        self.year2idx = year2idx

    def __getitem__(self, index):
        # extract data
        age = self.age[index][(-self.max_len+1):]
        code = self.code[index][(-self.max_len+1):]
        
        # Check if year data exists for this index
        if len(self.year[index]) > 0:
            year = self.year[index][(-self.max_len+1):]
        else:
            year = [0] * len(code) # Dummy if no year

        label = self.label[index]
        # convert label to float if necessary
        label = float(label) if isinstance(label, str) else float(label)

        # avoid data cut with first element to be 'SEP'
        if code[0] != 'SEP':
            code = np.append(np.array(['CLS']), code)
            age = np.append(np.array(age[0]), age)
            if len(year) > 0:
                year = np.append(np.array(year[0]), year)
        else:
            code[0] = 'CLS'

        # create mask: 1 for actual tokens, 0 for padding.
        mask = np.ones(self.max_len)
        mask[len(code):] = 0

        # pad sequences
        age = seq_padding(age, self.max_len, token2idx=self.age2idx)
        # If year2idx is None, basic padding
        year = seq_padding(year, self.max_len, token2idx=self.year2idx)

        # convert code tokens to indices
        tokens, code_idx = code2index(code, self.vocab)
        
        # Get tokens list padded for structure generation (pos/seg)
        tokens_padded = seq_padding(tokens, self.max_len, token2idx=None)
        
        position = position_idx(tokens_padded)
        segment = index_seg(tokens_padded)

        # pad code indices
        code_idx = seq_padding(code_idx, self.max_len, token2idx=None, symbol=0) # Assuming 0 is PAD in indices

        return (torch.LongTensor(age),
                torch.LongTensor(year),
                torch.LongTensor(code_idx),
                torch.LongTensor(position),
                torch.LongTensor(segment),
                torch.LongTensor(mask),
                torch.tensor(label, dtype=torch.float32))

    def __len__(self):
        return len(self.code)

# %%
# ------------------------------------------------------------------
# 4. MODEL DEFINITION (Transformers + Custom Embeddings)
# ------------------------------------------------------------------

class EHRBertEmbeddings(nn.Module):
    """
    Construct the embeddings from word, segment, age, position, and optional year.
    Sum = Word + Segment + Age + Position + (Year)
    """
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.age_embeddings = nn.Embedding(config.age_vocab_size, config.hidden_size)
        
        self.yearOn = config.yearOn
        if self.yearOn:
            self.year_embeddings = nn.Embedding(config.year_vocab_size, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Initialize Position Embeddings (Sinusoidal as per your requirement)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self._init_posi_embedding(self.position_embeddings.weight, config.max_position_embeddings, config.hidden_size)

    def _init_posi_embedding(self, weight_tensor, max_position_embedding, hidden_size):
        lookup_table = np.zeros((max_position_embedding, hidden_size), dtype=np.float32)
        for pos in range(max_position_embedding):
            for idx in np.arange(0, hidden_size, step=2):
                lookup_table[pos, idx] = np.sin(pos / (10000 ** (2 * idx / hidden_size)))
            for idx in np.arange(1, hidden_size, step=2):
                lookup_table[pos, idx] = np.cos(pos / (10000 ** (2 * idx / hidden_size)))
        
        with torch.no_grad():
            weight_tensor.copy_(torch.tensor(lookup_table))

    def forward(self, input_ids, age_ids=None, token_type_ids=None, position_ids=None, year_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if age_ids is None:
            age_ids = torch.zeros_like(input_ids)
        if position_ids is None:
            seq_length = input_ids.size(1)
            position_ids = self.position_ids[:, :seq_length]

        word_embed = self.word_embeddings(input_ids)
        segment_embed = self.token_type_embeddings(token_type_ids)
        age_embed = self.age_embeddings(age_ids)
        posi_embed = self.position_embeddings(position_ids)

        embeddings = word_embed + segment_embed + age_embed + posi_embed
        
        if self.yearOn and year_ids is not None:
            year_embed = self.year_embeddings(year_ids)
            embeddings += year_embed

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class EHRBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = EHRBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.init_weights()

    def forward(self, input_ids, age_ids=None, token_type_ids=None, position_ids=None, year_ids=None, attention_mask=None):
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_ids.size(), input_ids.device)

        embedding_output = self.embeddings(
            input_ids=input_ids, 
            age_ids=age_ids, 
            token_type_ids=token_type_ids, 
            position_ids=position_ids,
            year_ids=year_ids
        )
        
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_hidden_states=False
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        return sequence_output, pooled_output

class BertAgePredictor(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = EHRBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.regressor = nn.Linear(config.hidden_size, 1) # Regression output
        self.loss_fct = nn.MSELoss()
        self.init_weights()

    def forward(self, input_ids, age_ids, segment_ids, posi_ids, year_ids, attMask, labels=None):
        _, pooled_output = self.bert(
            input_ids, 
            age_ids=age_ids, 
            token_type_ids=segment_ids, 
            position_ids=posi_ids, 
            year_ids=year_ids,
            attention_mask=attMask
        )
        
        pooled_output = self.dropout(pooled_output)
        logits = self.regressor(pooled_output)
        
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1), labels.view(-1).float())
            
        return loss, logits.view(-1)

# %%
# ------------------------------------------------------------------
# 5. MAIN EXECUTION & CONFIGURATION
# ------------------------------------------------------------------

# --- Global Params ---
global_params = {
    'batch_size': 64,
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'output_dir': './output_age_pred',
    'max_len_seq': 250,
    'max_age': 110,
    'age_year': False,
    'age_symbol': None,
    'min_visit': 5,
    'inc_age': False, # Use Age in embedding
    'inc_year': False,
    'data_path': 'your_data.parquet', # Placeholder
    'vocab_path': 'bert_vocab.pkl',   # Placeholder
    'year_vocab_path': 'year_vocab.pkl' # Placeholder
}

set_all_seeds(1234)
create_folder(global_params['output_dir'])

# --- Load Data (Mocking load for script integrity) ---
# In a real run, ensure files exist. Here we check existence.
if os.path.exists(global_params['vocab_path']):
    BertVocab = load_obj(global_params['vocab_path'])
    print(f"Loaded Vocab. Size: {len(BertVocab['token2idx'])}")
else:
    print("Vocab file not found. Using mock size for demonstration.")
    BertVocab = {'token2idx': {'PAD':0, 'UNK':1, 'CLS':2, 'SEP':3, 'D1':4}}

if global_params['inc_year'] and os.path.exists(global_params['year_vocab_path']):
    YearVocab = load_obj(global_params['year_vocab_path'])
else:
    YearVocab = {'token2idx': {'PAD':0, 'UNK':1}}

# Get Age Vocab size
ageVocab, _ = age_vocab(max_age=global_params['max_age'], year=global_params['age_year'], symbol=global_params['age_symbol'])

# --- PREPARE MODEL CONFIG (STRICTLY FOLLOWING YOUR REQUEST) ---
# Mapping your dict to HuggingFace BertConfig
model_config_dict = {
    'vocab_size': len(BertVocab['token2idx'].keys()), 
    'hidden_size': 256, 
    'seg_vocab_size': 2, 
    'age_vocab_size': len(ageVocab.keys()), 
    'year_vocab_size': len(YearVocab['token2idx'].keys()), 
    'max_position_embedding': global_params['max_len_seq'], 
    'hidden_dropout_prob': 0.1, 
    'num_hidden_layers': 4, 
    'num_attention_heads': 4, 
    'attention_probs_dropout_prob': 0.1, 
    'intermediate_size': 1024, 
    'hidden_act': 'gelu', 
    'initializer_range': 0.02, 
    'yearOn': global_params['inc_year'],
    'concat_embeddings': False,
}

# Initialize HF Config
config = BertConfig(
    vocab_size=model_config_dict['vocab_size'],
    hidden_size=model_config_dict['hidden_size'],
    num_hidden_layers=model_config_dict['num_hidden_layers'],
    num_attention_heads=model_config_dict['num_attention_heads'],
    intermediate_size=model_config_dict['intermediate_size'],
    hidden_act=model_config_dict['hidden_act'],
    hidden_dropout_prob=model_config_dict['hidden_dropout_prob'],
    attention_probs_dropout_prob=model_config_dict['attention_probs_dropout_prob'],
    max_position_embeddings=model_config_dict['max_position_embedding'],
    type_vocab_size=model_config_dict['seg_vocab_size'],
    initializer_range=model_config_dict['initializer_range']
)

# Add custom attributes for our EHR Embeddings
config.age_vocab_size = model_config_dict['age_vocab_size']
config.year_vocab_size = model_config_dict['year_vocab_size']
config.yearOn = model_config_dict['yearOn']

# --- Initialize Model ---
age_model = BertAgePredictor(config)
age_model.to(global_params['device'])

print("Model Configured:")
print(config)

# %%
# ------------------------------------------------------------------
# 6. DATA LOADING & TRAINING LOOP
# ------------------------------------------------------------------

# NOTE: This block assumes 'processed_data_file.pkl' exists containing 
# 'train', 'valid', 'test' DataFrames. 
# If not, it attempts to load from parquet.

data_file_processed = f'ehr_processed_split.pkl'

if os.path.exists(data_file_processed):
    with open(data_file_processed, 'rb') as f:
        processed_data = pickle.load(f)
    train_df, valid_df, test_df = processed_data['train'], processed_data['valid'], processed_data['test']
    print("Loaded processed data splits.")
elif os.path.exists(global_params['data_path']):
    # Fallback: Load raw parquet and split
    df = pd.read_parquet(global_params['data_path'])
    # Basic preprocessing matching your flow
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label'])
    df = df[df['code'].apply(lambda x: len(list(x)) >= global_params['min_visit'])]
    
    # Simple Split
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df)
    train_df = df.iloc[:int(0.7*n)]
    valid_df = df.iloc[int(0.7*n):int(0.8*n)]
    test_df = df.iloc[int(0.8*n):]
else:
    print("No data file found. Creating dummy data for runnable code verification.")
    # Create dummy dataframe structure
    dummy_data = {
        'code': [['D1', 'D1', 'SEP', 'D1'] for _ in range(100)],
        'age': [[40, 40, 40, 41] for _ in range(100)],
        'year': [[2010, 2010, 2010, 2011] for _ in range(100)],
        'label': [5.5 for _ in range(100)]
    }
    train_df = pd.DataFrame(dummy_data)
    valid_df = pd.DataFrame(dummy_data)
    test_df = pd.DataFrame(dummy_data)

# Create Loaders
def get_loader(df, batch_size, shuffle=True):
    ds = SeqLoaderAge(
        token2idx=BertVocab['token2idx'],
        dataframe=df,
        max_len=global_params['max_len_seq'],
        max_age=global_params['max_age'],
        year=global_params['age_year'],
        age_symbol=global_params['age_symbol'],
        year2idx=YearVocab['token2idx']
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

train_loader = get_loader(train_df, global_params['batch_size'], True)
valid_loader = get_loader(valid_df, global_params['batch_size'], False)
test_loader = get_loader(test_df, global_params['batch_size'], False)

# Training Setup
optimizer = torch.optim.AdamW(age_model.parameters(), lr=1e-5)
scaler = torch.cuda.amp.GradScaler() # Mixed Precision
save_path = os.path.join(global_params['output_dir'], "best_model.pt")
logger = setup_logging(os.path.join(global_params['output_dir'], "log.txt"))

best_rmse = float('inf')
num_epochs = 10

print("Starting Training...")

for epoch in range(num_epochs):
    age_model.train()
    train_losses = []
    
    for step, batch in enumerate(train_loader):
        # Unpack
        age_ids, year_ids, input_ids, posi_ids, segment_ids, attMask, labels = [b.to(global_params['device']) for b in batch]
        
        # Handle feature flags
        if not global_params['inc_age']:
            age_ids = torch.zeros_like(age_ids)
        if not global_params['inc_year']:
            year_ids = torch.zeros_like(year_ids)
            
        optimizer.zero_grad()
        
        # Mixed Precision Forward
        with torch.cuda.amp.autocast():
            loss, _ = age_model(input_ids, age_ids, segment_ids, posi_ids, year_ids, attMask, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_losses.append(loss.item())
        
        if step % 50 == 0:
            print(f"Epoch {epoch+1} Step {step} Loss: {loss.item():.4f}")

    # Validation
    age_model.eval()
    preds = []
    truths = []
    with torch.no_grad():
        for batch in valid_loader:
            age_ids, year_ids, input_ids, posi_ids, segment_ids, attMask, labels = [b.to(global_params['device']) for b in batch]
            
            if not global_params['inc_age']: age_ids = torch.zeros_like(age_ids)
            if not global_params['inc_year']: year_ids = torch.zeros_like(year_ids)
            
            _, pred = age_model(input_ids, age_ids, segment_ids, posi_ids, year_ids, attMask, labels)
            preds.extend(pred.cpu().numpy())
            truths.extend(labels.cpu().numpy())
            
    rmse = np.sqrt(skm.mean_squared_error(truths, preds))
    mae = skm.mean_absolute_error(truths, preds)
    
    logger.info(f"Epoch {epoch+1}: Train Loss {np.mean(train_losses):.4f} | Valid RMSE {rmse:.4f} | MAE {mae:.4f}")
    print(f"Epoch {epoch+1}: Valid RMSE {rmse:.4f}")
    
    if rmse < best_rmse:
        best_rmse = rmse
        torch.save(age_model.state_dict(), save_path)
        logger.info("Best model saved.")

print("Training Complete.")