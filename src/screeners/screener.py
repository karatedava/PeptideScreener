from abc import ABC, abstractmethod
import pandas as pd

class Screener(ABC):

    def __init__(self, device:str='cpu', seq_header:str='sequence'):

        self.device = device
        self.header = seq_header
        

    @abstractmethod
    def run_screening(df:pd.DataFrame) -> pd.DataFrame:

        """
        - Run screening on df['sequence'] column
        RETURN: same dataframe with added collumn with predictions
        """