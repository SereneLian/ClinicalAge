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
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler, BertLayerNorm

# %%
# ------------------------------------------------------------------
# 2. UTILITY FUNCTIONS (Integrated)
# ------------------------------------------------------------------

def set_all_seeds(seed):
    """Set all seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

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
        for i in range(max_age * 12):
            age2idx[str(i)] = len(symbol) + i
            idx2age[len(symbol) + i] = str(i)
    return age2idx, idx2age

def seq_padding(tokens, max_len, token2idx=None, symbol='PAD'):
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
                try:
                    seq.append(token2idx.get(str(tokens[i]), token2idx.get('UNK', 1)))
                except:
                    seq.append(token2idx.get(tokens[i], token2idx.get('UNK', 1)))
            else:
                seq.append(token2idx.get(symbol, 0))
    return seq

def code2index(tokens, token2idx):
    output_tokens = []
    for i, token in enumerate(tokens):
        output_tokens.append(token2idx.get(token, token2idx.get('UNK', 1)))
    return tokens, output_tokens

def position_idx(tokens, symbol='SEP'):
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
    flag = 0
    seg = []
    for token in tokens:
        if token == symbol:
            seg.append(flag)
            flag = 1 - flag
        else:
            seg.append(flag)
    return seg

# %%
# ------------------------------------------------------------------
# 3. DATASET CLASS
# ------------------------------------------------------------------

class SeqLoaderAge(Dataset):
    def __init__(self, token2idx, dataframe, max_len, max_age=110, year=False, age_symbol=None, year2idx=None):
        self.vocab = token2idx
        self.max_len = max_len
        self.code = dataframe.code.values
        self.age = dataframe.age.values
        # Handle existence of year column safely
        self.year = dataframe.year.values if 'year' in dataframe.columns else [[] for _ in range(len(dataframe))]
        self.label = dataframe.label.values
        self.baseline_age = dataframe.baseline_age.values if 'baseline_age' in dataframe.columns else np.zeros(len(dataframe))
        self.patid = dataframe.patid.values if 'patid' in dataframe.columns else np.arange(len(dataframe))
        self.phenoage = dataframe.phenoage.values if 'phenoage' in dataframe.columns else np.zeros(len(dataframe))

        self.age2idx, _ = age_vocab(max_age, year, symbol=age_symbol)
        self.year2idx = year2idx

    def __getitem__(self, index):
        age = self.age[index][(-self.max_len+1):]
        code = self.code[index][(-self.max_len+1):]
        
        if len(self.year[index]) > 0:
            year = self.year[index][(-self.max_len+1):]
        else:
            year = [0] * len(code)

        label = float(self.label[index])

        if code[0] != 'SEP':
            code = np.append(np.array(['CLS']), code)
            age = np.append(np.array(age[0]), age)
            if len(year) > 0:
                year = np.append(np.array(year[0]), year)
        else:
            code[0] = 'CLS'

        mask = np.ones(self.max_len)
        mask[len(code):] = 0

        age = seq_padding(age, self.max_len, token2idx=self.age2idx)
        year = seq_padding(year, self.max_len, token2idx=self.year2idx)

        tokens, code_idx = code2index(code, self.vocab)
        tokens_padded = seq_padding(tokens, self.max_len, token2idx=None)
        
        position = position_idx(tokens_padded)
        segment = index_seg(tokens_padded)
        code_idx = seq_padding(code_idx, self.max_len, token2idx=None, symbol=0)

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
# 4. MODEL DEFINITION (Must match Training Architecture)
# ------------------------------------------------------------------

class EHRBertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.age_embeddings = nn.Embedding(config.age_vocab_size, config.hidden_size)
        
        self.yearOn = getattr(config, 'yearOn', False)
        if self.yearOn:
            self.year_embeddings = nn.Embedding(config.year_vocab_size, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
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
        if token_type_ids is None: token_type_ids = torch.zeros_like(input_ids)
        if age_ids is None: age_ids = torch.zeros_like(input_ids)
        if position_ids is None:
            seq_length = input_ids.size(1)
            position_ids = self.position_ids[:, :seq_length]

        embeddings = (self.word_embeddings(input_ids) + 
                      self.token_type_embeddings(token_type_ids) + 
                      self.age_embeddings(age_ids) + 
                      self.position_embeddings(position_ids))
        
        if self.yearOn and year_ids is not None:
            embeddings += self.year_embeddings(year_ids)

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
        embedding_output = self.embeddings(input_ids, age_ids=age_ids, token_type_ids=token_type_ids, position_ids=position_ids, year_ids=year_ids)
        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask, output_hidden_states=False)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output

class BertAgePredictor(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = EHRBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.regressor = nn.Linear(config.hidden_size, 1)
        self.loss_fct = nn.MSELoss()
        self.init_weights()

    def forward(self, input_ids, age_ids, segment_ids, posi_ids, year_ids, attMask, labels=None):
        _, pooled_output = self.bert(input_ids, age_ids=age_ids, token_type_ids=segment_ids, position_ids=posi_ids, year_ids=year_ids, attention_mask=attMask)
        pooled_output = self.dropout(pooled_output)
        logits = self.regressor(pooled_output)
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1), labels.view(-1).float())
        return loss, logits.view(-1)

# %%
# ------------------------------------------------------------------
# 5. MAIN INFERENCE EXECUTION
# ------------------------------------------------------------------

# --- Configuration ---
file_config = {
    'vocab': 'path/to/bert_vocab.pkl',       # UPDATE THIS
    'yearVocab': 'path/to/year_vocab.pkl',   # UPDATE THIS
    'trained_model': 'path/to/best_model.pt',# UPDATE THIS
    'data_path': 'path/to/inference_data.parquet', # UPDATE THIS
}

global_params = {
    'batch_size': 512,
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'output_dir': './inference_output',
    'min_visit': 1,
    'max_len_seq': 256,
    'max_age': 110,
    'age_year': False,
    'age_symbol': None,
    'inc_age': False,  
    'inc_year': False 
}

# Setup output
os.makedirs(global_params['output_dir'], exist_ok=True)
logger = setup_logging(os.path.join(global_params['output_dir'], "inference_log.txt"))

# --- Load Vocabs ---
try:
    BertVocab = load_obj(file_config['vocab'])
    # Extend vocab with numbers 0-100 as per your original logic
    curr_max = max(BertVocab['token2idx'].values())
    for i in range(101):
        t = str(i)
        if t not in BertVocab['token2idx']:
            curr_max += 1
            BertVocab['token2idx'][t] = curr_max
    print(f"Vocab Loaded. Size: {len(BertVocab['token2idx'])}")
except FileNotFoundError:
    logger.warning("Vocab file not found. Inference will likely fail without correct indices.")
    BertVocab = {'token2idx': {'PAD':0, 'UNK':1, 'CLS':2, 'SEP':3}}

YearVocab = load_obj(file_config['yearVocab']) if os.path.exists(file_config['yearVocab']) else {'token2idx': {'PAD':0}}
ageVocab, _ = age_vocab(max_age=global_params['max_age'], year=global_params['age_year'], symbol=global_params['age_symbol'])

# --- Model Config (Matches Training) ---
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
}

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
config.age_vocab_size = model_config_dict['age_vocab_size']
config.year_vocab_size = model_config_dict['year_vocab_size']
config.yearOn = model_config_dict['yearOn']

# --- Initialize & Load Model ---
print("Initializing model...")

age_model = BertAgePredictor(config)
if os.path.exists(file_config['trained_model']):
    print(f"Loading weights from {file_config['trained_model']}")
    # Strict=False allows skipping missing keys if minor version mismatches occur
    age_model.load_state_dict(torch.load(file_config['trained_model'], map_location='cpu'), strict=False)
else:
    logger.error("Trained model file not found!")

age_model.to(global_params['device'])
age_model.eval()

# --- Load Data ---
print("Loading Inference Data...")
if os.path.exists(file_config['data_path']):
    df_inference = pd.read_parquet(file_config['data_path'])
    
    # Preprocessing
    df_inference['label'] = pd.to_numeric(df_inference['label'], errors='coerce')
    df_inference['baseline_age'] = pd.to_numeric(df_inference['baseline_age'], errors='coerce')
    if 'phenoage' not in df_inference.columns:
        df_inference.rename(columns={'label': 'phenoage'}, inplace=True)
    # Re-calculate target label (Label = PhenoAge - BaselineAge)
    df_inference['label'] = df_inference['phenoage'] - df_inference['baseline_age']
    
    # Filter
    df_inference = df_inference[df_inference['code'].apply(lambda x: list(x).count('SEP') >= global_params['min_visit'])]
    df_inference = df_inference.reset_index(drop=True)
    print(f"Data Loaded. Rows: {len(df_inference)}")

    # Create Loader
    inference_dataset = SeqLoaderAge(
        token2idx=BertVocab['token2idx'],
        dataframe=df_inference,
        max_len=global_params['max_len_seq'],
        max_age=global_params['max_age'],
        year=global_params['age_year'],
        age_symbol=global_params['age_symbol'],
        year2idx=YearVocab['token2idx']
    )
    inference_loader = DataLoader(inference_dataset, batch_size=global_params['batch_size'], shuffle=False, num_workers=4)

    # --- Run Inference ---
    print("Starting Inference Loop...")
    all_preds = []
    
    with torch.no_grad():
        for batch in inference_loader:
            age_ids, year_ids, input_ids, posi_ids, segment_ids, attMask, labels = [b.to(global_params['device']) for b in batch]
            
            if not global_params['inc_age']:
                age_ids = torch.zeros_like(age_ids)
            if not global_params['inc_year']:
                year_ids = torch.zeros_like(year_ids)
                
            _, pred = age_model(input_ids, age_ids, segment_ids, posi_ids, year_ids, attMask, labels)
            all_preds.extend(pred.cpu().numpy())
            
    # Save Results
    df_inference['predicted'] = all_preds
    # Predicted PhenoAge = Baseline + Predicted Delta
    df_inference['predicted_phenoage'] = df_inference['baseline_age'] + df_inference['predicted']
    
    output_csv = os.path.join(global_params['output_dir'], "age_predictions.csv")
    
    cols_to_save = ['patid', 'predicted', 'baseline_age', 'phenoage', 'label', 'predicted_phenoage']
    # Filter cols that actually exist
    cols_to_save = [c for c in cols_to_save if c in df_inference.columns]
    
    df_inference[cols_to_save].to_csv(output_csv, index=False)
    logger.info(f"Inference complete. Saved to {output_csv}")
    print(f"Done. Saved to {output_csv}")

else:
    logger.error(f"Data path {file_config['data_path']} does not exist.")