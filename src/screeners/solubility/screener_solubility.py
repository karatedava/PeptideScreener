"""
SCREENER BASED ON SOLUBILITY CLASSIFICATION
"""

from src.screeners.screener import Screener
import pandas as pd
import numpy as np

from typing import Dict
import re
from pathlib import Path
import joblib
from localcider.sequenceParameters import SequenceParameters
from sparrow import Protein
import xgboost

class SolubilityScreener(Screener):

    def __init__(self, model_path:Path, seq_header:str = 'sequence'):
        super().__init__(seq_header=seq_header)

        self.model = joblib.load(model_path)['model']
    
    def clean_sequence(self, raw: str) -> str:
        """Remove N-terminal Ac-, C-terminal –NH2, and expand (A4) style repeats."""
        seq = CLEAN_REGEXES["acetyl"].sub("", raw)
        seq = CLEAN_REGEXES["amid"].sub("", seq)
        seq = seq.replace("-", "")                         # strip internal dashes
        # expand e.g. (K5) → KKKKK
        while (m := CLEAN_REGEXES["repeat"].search(seq)):
            seq = seq[: m.start()] + m.group(1).upper() * int(m.group(2)) + seq[m.end():]
        seq = seq.strip().upper()
        # if not VALID_SEQ.fullmatch(seq):
        #     sys.exit(f"[ERROR] Non-canonical residue(s) after cleaning: {seq}")
        return seq


    def calc_sequence_features(self, seq: str) -> Dict[str, float]:
        """Return the exact 7 feature values used during model training."""
        sp = SequenceParameters(seq)
        prot = Protein(seq)
        return {
            "fracpol": sum(sp.get_amino_acid_fractions()[x] for x in "QNSTGCH"),
            "dispro":  sp.get_fraction_disorder_promoting(),
            "isopoi":  sp.get_isoelectric_point(),
            "e2e_scaled":        prot.predictor.end_to_end_distance(use_scaled=True),
            "rg_scaled":         prot.predictor.radius_of_gyration(use_scaled=True),
            "scaling_exponent":  prot.predictor.scaling_exponent(),
            "prefactor":         prot.predictor.prefactor(),
        }
    
    def run_screening(self, df:pd.DataFrame):

        sequences = df[self.header].to_list()
        # sequences = [self.clean_sequence(seq) for seq in sequences]
        features = [self.calc_sequence_features(seq) for seq in sequences]
        np.array([self.calc_sequence_features(seq) for seq in sequences])
        features = pd.DataFrame(features)

        # probabilities = np.array(self.model.predict_proba(features)[:, 1], dtype=np.float32)

        Xt = features.copy()

        # Skip imputer, apply scaler + anything else before xgb
        Xt = self.model.named_steps['scaler'].transform(Xt)

        # Predict directly with the XGBClassifier
        probabilities = self.model.named_steps['xgb'].predict_proba(Xt)[:, 1]
        probabilities = np.array(probabilities, dtype=np.float32)


        df['solubility_prob'] = probabilities

        return df

CLEAN_REGEXES = {
    "acetyl": re.compile(r"^\s*Ac-", flags=re.I),
    "amid":   re.compile(r"-NH2\s*$", flags=re.I),
    "repeat": re.compile(r"\(([ACDEFGHIKLMNPQRSTVWY])(\d+)\)", flags=re.I),
    }