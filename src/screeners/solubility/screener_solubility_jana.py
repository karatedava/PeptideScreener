"""
SCREENER BASED ON SOLUBILITY CLASSIFICATION
"""

from src.screeners.screener import Screener
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
from tqdm import tqdm
import torch
from transformers import BertModel, BertTokenizer
import joblib

class SolubilityScreenerJana(Screener):

    def __init__(self, model_path:Path, device:str='cpu', seq_header:str = 'sequence', thr:float=0.55):
        super().__init__(seq_header=seq_header, device=device)

        self.model = joblib.load(model_path)
        self.threshold = thr

        # Initialize the ProtBERT tokenizer and model
        #self.tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, use_fast=False)
        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        protbert_model = BertModel.from_pretrained("Rostlab/prot_bert")
        self.protbert_model = protbert_model.to(device)

    def run_screening(self, df:pd.DataFrame):

        sequences = df[self.header].to_list()

        embeddings = self.get_protbert_embeddings(sequences)

        probabilities = self.model.predict_proba(embeddings)[:, 1]  # Probabilities for the positive class
        predictions = (probabilities > self.threshold).astype(int)
        df['jana_solubility'] = predictions

        return df
    
    def get_protbert_embeddings(
        self,
        sequences: List[str],
        bs: int = 0,
        max_length: int = 50,
    ) -> np.ndarray:
        """
        Compute ProtBERT embeddings in batches with mean pooling (using attention mask).
        
        Args:
            sequences: List of protein sequences (without spaces)
            batch_size: Number of sequences per batch
            max_length: Maximum token length (truncation point)
        
        Returns:
            np.ndarray: shape (n_sequences, 1024)
        """
        # Preprocess: insert space between amino acids (required for ProtBERT tokenizer)
        processed_seqs = [' '.join(seq.strip()) for seq in sequences]

        if bs == 0:
            if len(processed_seqs) >= 1000:
                bs=64
            elif len(processed_seqs) >= 10000:
                bs=256
            elif len(processed_seqs) >= 100000:
                bs=512
            else:
                bs = 4
        
        embeddings = []
        
        for i in tqdm(range(0, len(processed_seqs), bs), desc="ProtBERT embedding"):
            batch = processed_seqs[i:i + bs]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt',
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.protbert_model(**inputs)
            
            # Shape: (batch_size, seq_len, 50)
            hidden_states = outputs.last_hidden_state
            
            # Attention mask shape: (batch_size, seq_len) → expand to (batch_size, seq_len, 50)
            mask = inputs['attention_mask'].unsqueeze(-1).expand_as(hidden_states).float()
            
            sum_hidden = torch.sum(hidden_states * mask, dim=1)          # (batch_size, 50)
            num_real_tokens = torch.sum(mask, dim=1).clamp(min=1.0)       # avoid div-by-zero
            mean_pooled = sum_hidden / num_real_tokens                    # (batch_size, 50)
            
            # Collect
            embeddings.append(mean_pooled.cpu().numpy())
        
        # Concatenate all batches
        return np.concatenate(embeddings, axis=0)