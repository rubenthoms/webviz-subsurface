from typing import List, Dict, Optional, Iterable, Tuple, Sequence

import pandas as pd

# fmt: off

class SimulationTimeSeriesModel:
    def __init__(self, smry_df: pd.DataFrame, smry_meta_df: Optional[pd.DataFrame] = None) -> None: ...
    
    @property
    def ensemble_names(self) -> List[str]: ...

    # Queries for available vector names
    # Do we need more or other filters here, eg.: only non-zero, exclude historical vectors, by type
    @property
    def all_vector_names(self) -> List[str]: ...
    @property
    def simulation_vector_names(self) -> List[str]: ...
    @property
    def historical_vector_names(self) -> List[str]: ...

    # Should this kind of functionality be here at all?
    @property
    def vector_groups(self) -> Dict[str, dict]: ...

    @property
    def dates(self) -> List[str]: ...

    def daterange_for_vector(self, ensemble: str, vector_name: str) -> Tuple[str, str]: ...

    def get_vector(self, ensemble: str, vector_name: str) -> pd.DataFrame: ...
    def get_vector_filtered_by_realizations(self, ensemble: str, vector_name: str, realizations: Sequence[int]) -> pd.DataFrame: ...

    def get_vectors_for_date(self, ensemble: str, date: str, vector_names: Sequence[str]) -> pd.DataFrame: ...

    # If we were to have this method, what is the definition of "all" vectors
    def get_all_vectors_for_date(self, ensemble: str, date: str) -> pd.DataFrame: ...

    # Does this method make sense? Probably not
    def get_historical_vector(self, vector_name: str) -> pd.DataFrame: ...

    # To be considered
    #   Data type for dates?
    #   Is DataFrame the best data structure for return values?
    #   Query for the existence of a vector by name, eg.: has_vector_named()
    #   Vectors grouped by type, does this even belong here or rather in separate helper? Or just needed in GUI component
    #   Getter(s) for min/max date(s)
    #   Any need for querying available realizations?
    #   Mapping between simulation and historical vectors, eg.: map_to_hist_vec_name() 

# fmt: on