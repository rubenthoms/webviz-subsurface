from typing import List, Dict, Optional, Iterable, Tuple, Sequence

import pandas as pd
import numpy as np

# fmt: off



class SimulationTimeSeriesModel:
    def __init__(self, smry_df: pd.DataFrame, smry_meta_df: Optional[pd.DataFrame] = None) -> None: ...
    
    @property
    #def ensemble_names(self) -> List[str]: ...

    # Queries for meta data
    def vector_names(self) -> List[str]: ...
    def has_vector_named(vector_name: str) -> bool: ...
    def get_filtered_vector_names(self, excludeAllZeroVals: bool, excludeConstantVal: bool) -> List[str]: ...

    def realizations(self) -> List[int]: ...

    def dates(self, realizations: Optional[Sequence[int]]) -> List[str]: ...

    # Data access
    def get_vector(self, vector_name: str, realizations: Optional[Sequence[int]]) -> pd.DataFrame: ...
    def get_vectors(self, vector_names: List[str], realizations: Optional[Sequence[int]]) -> pd.DataFrame: ...

    def get_vectors_for_date(self, date: str, vector_names: Sequence[str]) -> pd.DataFrame: ...
    def get_vectors_for_date(self, date: str, vector_names: Sequence[str], realizations: Optional[Sequence[int]]) -> pd.DataFrame: ...

    # If we were to have this method, what is the definition of "all" vectors
    def get_all_vectors_for_date(self, date: str, realizations: Optional[Sequence[int]]) -> pd.DataFrame: ...


    # Still needs to be discussed
    #   Data type for dates?
    #   Is DataFrame the best data structure for return values?
    #   Query for the existence of a vector by name, eg.: has_vector_named()


# Return values consisting of numpy arrays instead of Pandas DataFrames
# Alernative class name RealizationVectors
#@dataclass
class RealizationData:
    realization: int  
    date_array: np.ndarray[np.str_]
    vector_names: List[str]
    vector_arrays: List[np.ndarray[np.float32]]

# and then
def get_vectors(self, vector_names: List[str], realizations: Optional[Sequence[int]]) -> List[RealizationData]: ...

# fmt: on