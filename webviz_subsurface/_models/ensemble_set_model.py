import json
from typing import Union, Optional, List, Callable, Tuple, Dict, Any
import pathlib

import inspect
import tracemalloc


import pandas as pd

from .ensemble_model import EnsembleModel


# !!!!!!!!!!!!!!!!!!
_ensemble_set_model_cache: Dict[str, "EnsembleSetModel"] = {}


class EnsembleSetModel:
    """Class to load and manipulate ensemble sets from given paths to
    ensembles on disk"""

    @staticmethod
    def get_or_create_model(
        ensemble_paths: dict,
        time_index: Optional[Union[list, str]] = None,
        column_keys: Optional[list] = None,
    ) -> "EnsembleSetModel":

        global _ensemble_set_model_cache

        #!!!!!!!!!!!!!!!!!!!!!
        # export PYTHONTRACEMALLOC=1
        tracemalloc.start()

        current0, peak0 = tracemalloc.get_traced_memory()

        stack = inspect.stack()
        the_class = stack[1][0].f_locals["self"].__class__.__name__
        the_method = stack[1][0].f_code.co_name
        print(
            "\nEnsembleSetModel.get_or_create_model() -- called by {}.{}()".format(
                the_class, the_method
            )
        )

        tmp_dict = {
            "ensemble_paths": ensemble_paths,
            "time_index": time_index,
            "column_keys": column_keys,
        }
        key = json.dumps(tmp_dict)
        if key in _ensemble_set_model_cache:
            print(
                "EnsembleSetModel.get_or_create_model() -- returning cached EnsembleSetModel"
            )
            return _ensemble_set_model_cache[key]

        print(
            "EnsembleSetModel.get_or_create_model() -- Creating new EnsembleSetModel!!!!!!!!!!!!!!!!!!!"
        )
        print("EnsembleSetModel.get_or_create_model() -- key", key)
        new_model = EnsembleSetModel(
            ensemble_paths=ensemble_paths,
            time_index=time_index,
            column_keys=column_keys,
        )
        _ensemble_set_model_cache[key] = new_model

        current, peak = tracemalloc.get_traced_memory()

        print(
            f"EnsembleSetModel.get_or_create_model() -- curr memory usage: {current / 10**6}MB;  delta: {(current-current0) / 10**6}MB;  peak: {peak / 10**6}MB"
        )

        return new_model

    def __init__(
        self,
        ensemble_paths: dict,
        time_index: Optional[Union[list, str]],
        column_keys: Optional[list],
        ensemble_set_name: str = "EnsembleSet",
        filter_file: Union[str, None] = "OK",
    ) -> None:
        self._ensemble_paths = ensemble_paths
        self._ensemble_set_name = ensemble_set_name
        self._filter_file = filter_file
        self._webvizstore: List = []
        self._ensembles = [
            EnsembleModel(ens_name, ens_path, filter_file=self._filter_file)
            for ens_name, ens_path in self._ensemble_paths.items()
        ]

        ## Loading here
        self._loaded_smry_df = EnsembleSetModel._get_ensembles_data(
            self._ensembles, "load_smry", time_index=time_index, column_keys=column_keys
        )

        tmp_smry_meta: dict = {}
        for ensemble in self._ensembles:
            tmp_smry_meta.update(
                ensemble.load_smry_meta(column_keys=column_keys).T.to_dict()
            )
        self._loaded_smry_meta_df = pd.DataFrame(tmp_smry_meta).transpose()

    def __repr__(self) -> str:
        return f"EnsembleSetModel: {self._ensemble_paths}"

    @property
    def ens_folders(self) -> dict:
        """Get root folders for ensemble set"""
        return {
            ens: pathlib.Path(ens_path.split("realization")[0])
            for ens, ens_path in self._ensemble_paths.items()
        }

    @staticmethod
    def _get_ensembles_data(
        ensemble_models: List[EnsembleModel], func: str, **kwargs: Any
    ) -> pd.DataFrame:
        """Runs the provided function for each ensemble and concats dataframes"""
        dfs = []
        for ensemble in ensemble_models:
            try:
                dframe = getattr(ensemble, func)(**kwargs)
                dframe.insert(0, "ENSEMBLE", ensemble.ensemble_name)
                dfs.append(dframe)
            except (KeyError, ValueError):
                # Happens if an ensemble is missing some data
                # Warning has already been issued at initialization
                pass
        if dfs:
            return pd.concat(dfs, sort=False)
        raise KeyError(f"No data found for {func} with arguments: {kwargs}")

    def load_parameters(self) -> pd.DataFrame:
        return EnsembleSetModel._get_ensembles_data(self._ensembles, "load_parameters")

    def load_additional_smry_df(
        self,
        time_index: Optional[Union[list, str]] = None,
        column_keys: Optional[list] = None,
    ) -> pd.DataFrame:
        return EnsembleSetModel._get_ensembles_data(
            self._ensembles, "load_smry", time_index=time_index, column_keys=column_keys
        )

    def get_smry_df(self) -> pd.DataFrame:
        return self._loaded_smry_df

    def get_smry_meta_df(self) -> pd.DataFrame:
        return self._loaded_smry_meta_df

    def load_csv(self, csv_file: pathlib.Path) -> pd.DataFrame:
        return EnsembleSetModel._get_ensembles_data(
            self._ensembles, "load_csv", csv_file=csv_file
        )

    @property
    def webvizstore(self) -> List[Tuple[Callable, List[Dict]]]:
        store_functions = []
        for ensemble in self._ensembles:
            store_functions.extend(ensemble.webviz_store)
        return store_functions
