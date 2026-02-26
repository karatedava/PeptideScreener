
from src.screeners.screener import Screener
from src.screeners.toxicity.cytotoxicity_screener import CytotoxicityScreener
from src.screeners.solubility.screener_solubility_opt import SolubilityScreener

from src.config import toxicity_clf_path, solubility_clf_path, DEVICE_OPTIONS
from src.utils import get_best_device

import pandas as pd

class ScreenerManager():

    def __init__(self, screeners_dict:dict, seq_header:str='sequence'):

        scr_keys = [key for key in list(screeners_dict.keys()) if screeners_dict[key] == True]
        self.header = seq_header
        self.screener_list = [self._get_screener_(scrk) for scrk in scr_keys]

    def _get_screener_(self, scr_key:str) -> Screener:

        """
        return corresponding screener based on the src_key
        """

        if scr_key == 'toxicity':
            device = get_best_device(DEVICE_OPTIONS)
            screener = CytotoxicityScreener(model_path=toxicity_clf_path, device=device, seq_header=self.header)
        if scr_key == 'solubility':
            screener = SolubilityScreener(model_path=solubility_clf_path, seq_header=self.header)
        
        return screener

    def curate_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Curate the DataFrame by keeping only rows where the sequence column
        (self.header) is not null, not NaN, and not an empty string.
        
        Returns:
            pd.DataFrame: Filtered DataFrame with valid sequences only
        """
        if self.header not in df.columns:
            raise ValueError(f"Column '{self.header}' not found in DataFrame")
        
        valid_mask = (
            df[self.header].notna() & 
            (df[self.header].astype(str).str.strip() != '')
        )
        
        # Return filtered dataframe
        return df[valid_mask].copy()

    def run_complete_screening(self, input_df:pd.DataFrame) -> pd.DataFrame:
        
        df = input_df.copy()

        print('\n CURATING SEQUENCES \n')
        df = self.curate_sequences(df)

        print('\n STARTING SCREENING \n')
        for screener in self.screener_list:
            df = screener.run_screening(df)

        return df