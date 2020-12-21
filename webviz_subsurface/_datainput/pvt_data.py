########################################
#
#  Copyright (C) 2020-     Equinor ASA
#
########################################

import sys
from typing import Dict, List, Any
import warnings

import pandas as pd

# opm and ecl2df are only available for Linux,
# hence, ignore any import exception here to make
# it still possible to use the PvtPlugin on
# machines with other OSes.
#
# NOTE: Functions in this file cannot be used
#       on non-Linux OSes.
try:
    import ecl2df
    from opm.io.ecl import EclFile
except ImportError:
    pass

from webviz_config.common_cache import CACHE
from webviz_config.webviz_store import webvizstore

from .fmu_input import load_ensemble_set, load_csv
from .opm_init_io.pvt_oil import Oil
from .opm_init_io.pvt_gas import Gas
from .opm_init_io.pvt_water import Water


@CACHE.memoize(timeout=CACHE.TIMEOUT)
@webvizstore
def filter_pvt_data_frame(
    data_frame: pd.DataFrame, drop_ensemble_duplicates: bool = False
) -> pd.DataFrame:

    data_frame = data_frame.rename(str.upper, axis="columns").rename(
        columns={
            "TYPE": "KEYWORD",
            "RS": "GOR",
            "RSO": "GOR",
            "R": "GOR",
            "RV": "OGR",
        }
    )
    data_frame = data_frame.fillna(0)
    if "GOR" in data_frame.columns and "OGR" in data_frame.columns:
        data_frame["RATIO"] = data_frame["GOR"] + data_frame["OGR"]
    elif "GOR" in data_frame.columns:
        data_frame["RATIO"] = data_frame["GOR"]
    elif "OGR" in data_frame.columns:
        data_frame["RATIO"] = data_frame["OGR"]

    columns = [
        "ENSEMBLE",
        "REAL",
        "PVTNUM",
        "KEYWORD",
        "RATIO",
        "PRESSURE",
        "VOLUMEFACTOR",
        "VISCOSITY",
    ]

    if not "RATIO" in data_frame.columns:
        data_frame["RATIO"] = 1.0

    data_frame = data_frame[columns]

    stored_data_frames: List[pd.DataFrame] = []
    cleaned_data_frame = data_frame.iloc[0:0]
    columns_subset = data_frame.columns.difference(["REAL", "ENSEMBLE"])

    for ens, ens_data_frame in data_frame.groupby("ENSEMBLE"):
        ens_merged_dataframe = data_frame.iloc[0:0]
        for _rno, realization_data_frame in ens_data_frame.groupby("REAL"):
            if ens_merged_dataframe.empty:
                ens_merged_dataframe = ens_merged_dataframe.append(
                    realization_data_frame
                ).reset_index(drop=True)
            else:
                ens_merged_dataframe = (
                    pd.concat([ens_merged_dataframe, realization_data_frame])
                    .drop_duplicates(
                        subset=columns_subset,
                        keep="first",
                    )
                    .reset_index(drop=True)
                )

        if ens_merged_dataframe["REAL"].nunique() > 1:
            warnings.warn(
                (
                    f"There are variations in PVT between realizations in ensemble {ens}. "
                    "This is currently not supported. Only keeping data for realization "
                    f"{ens_merged_dataframe['REAL'].iloc[0]}."
                ),
                UserWarning,
            )
            ens_merged_dataframe = ens_merged_dataframe[
                ens_merged_dataframe["REAL"] == ens_merged_dataframe["REAL"].iloc[0]
            ]
        if drop_ensemble_duplicates:
            data_frame_stored = False
            for stored_data_frame in stored_data_frames:
                if all(
                    stored_data_frame[columns_subset]
                    == ens_merged_dataframe[columns_subset]
                ):
                    data_frame_stored = True
                    break
            if data_frame_stored:
                continue
            stored_data_frames.append(ens_merged_dataframe)
        cleaned_data_frame = cleaned_data_frame.append(ens_merged_dataframe)

    return cleaned_data_frame


@CACHE.memoize(timeout=CACHE.TIMEOUT)
@webvizstore
def load_pvt_csv(
    ensemble_paths: dict,
    csv_file: str,
    ensemble_set_name: str = "EnsembleSet",
    drop_ensemble_duplicates: bool = False,
) -> pd.DataFrame:
    return filter_pvt_data_frame(
        load_csv(ensemble_paths, csv_file, ensemble_set_name), drop_ensemble_duplicates
    )


@CACHE.memoize(timeout=CACHE.TIMEOUT)
@webvizstore
def load_pvt_dataframe(
    ensemble_paths: Dict[str, str],
    ensemble_set_name: str = "EnsembleSet",
    use_init_file: bool = False,
    drop_ensemble_duplicates: bool = False,
) -> pd.DataFrame:
    # pylint: disable=too-many-statements
    # If ecl2df is not loaded, this machine is probably not
    # running Linux and the modules are not available.
    # To avoid a crash, return an empty DataFrame here.
    if "ecl2df" not in sys.modules:
        print(
            "Your operating system does not support opening and reading"
            " Eclipse files. An empty data frame will be returned and your"
            " plot will therefore not show any data points. Please specify"
            " a relative path to a PVT CSV file in your PvtPlot settings"
            " to display PVT data anyways."
        )
        return pd.DataFrame({})

    def ecl2df_pvt_data_frame(kwargs: Any) -> pd.DataFrame:
        return ecl2df.pvt.df(kwargs["realization"].get_eclfiles())

    def init_to_pvt_data_frame(kwargs: Any) -> pd.DataFrame:
        # pylint: disable-msg=too-many-locals
        # pylint: disable=too-many-nested-blocks
        # pylint: disable=too-many-branches
        ecl_init_file = EclFile(
            kwargs["realization"].get_eclfiles().get_initfile().get_filename()
        )

        oil = Oil.from_ecl_init_file(ecl_init_file)
        gas = Gas.from_ecl_init_file(ecl_init_file)
        water = Water.from_ecl_init_file(ecl_init_file)

        column_pvtnum = []
        column_ratio = []
        column_volume_factor = []
        column_pressure = []
        column_viscosity = []
        column_keyword = []

        ratios: List[float] = []
        pressures: List[float] = []
        (pressure_min, pressure_max) = (0.0, 0.0)

        if oil and not oil.is_dead_oil_const_compr():
            (pressure_min, pressure_max) = oil.range_independent(0)
        elif gas:
            (pressure_min, pressure_max) = gas.range_independent(0)
        else:
            raise NotImplementedError("Missing PVT data")

        for pressure_step in range(0, 20 + 1):
            pressures.append(
                pressure_min + pressure_step / 20.0 * (pressure_max - pressure_min)
            )

        if oil:
            if oil.is_live_oil():
                keyword = "PVTO"
            elif oil.is_dead_oil():
                keyword = "PVDO"
            elif oil.is_dead_oil_const_compr():
                keyword = "PVCDO"
            else:
                raise NotImplementedError(
                    "The PVT property type of oil is not implemented."
                )

            for region_index, region in enumerate(oil.regions()):
                if oil.is_dead_oil_const_compr():
                    column_pvtnum.append([region_index for _ in pressures])
                    column_keyword.append([keyword for _ in pressures])
                    column_ratio.append(ratios)
                    column_pressure.append(pressures)
                    column_volume_factor.append(
                        region.formation_volume_factor(ratios, pressures)
                    )
                    column_viscosity.append(region.viscosity(ratios, pressures))

                else:
                    (ratio, pressure) = (
                        region.get_keys(),
                        region.get_independents(),
                    )
                    column_pvtnum.append([region_index for _ in pressure])
                    column_keyword.append([keyword for _ in pressure])
                    column_ratio.append(ratio)
                    column_pressure.append(pressure)
                    column_volume_factor.append(
                        region.formation_volume_factor(ratio, pressure)
                    )
                    column_viscosity.append(region.viscosity(ratio, pressure))

        if gas:
            if gas.is_wet_gas():
                keyword = "PVTG"
            elif gas.is_dry_gas():
                keyword = "PVDG"
            else:
                raise NotImplementedError(
                    "The PVT property type of gas is not implemented."
                )

            for region_index, region in enumerate(gas.regions()):
                (ratio, pressure) = (
                    region.get_keys(),
                    region.get_independents(),
                )
                column_pvtnum.append([region_index for _ in pressure])
                column_keyword.append([keyword for _ in pressure])
                column_ratio.append(ratio)
                column_pressure.append(pressure)
                column_volume_factor.append(
                    region.formation_volume_factor(ratio, pressure)
                )
                column_viscosity.append(region.viscosity(ratio, pressure))

        if water:
            for region_index, region in enumerate(water.regions()):
                column_pvtnum.append([region_index for _ in pressures])
                column_keyword.append([keyword for _ in pressures])
                column_ratio.append(ratios)
                column_pressure.append(pressures)
                column_volume_factor.append(
                    region.formation_volume_factor(ratios, pressures)
                )
                column_viscosity.append(region.viscosity(ratios, pressures))

        data_frame = pd.DataFrame(
            {
                "PVTNUM": column_pvtnum,
                "KEYWORD": column_keyword,
                "R": column_ratio,
                "PRESSURE": column_pressure,
                "VOLUMEFACTOR": column_volume_factor,
                "VISCOSITY": column_viscosity,
            }
        )

        return data_frame

    return filter_pvt_data_frame(
        load_ensemble_set(ensemble_paths, ensemble_set_name).apply(
            init_to_pvt_data_frame if use_init_file else ecl2df_pvt_data_frame
        ),
        drop_ensemble_duplicates,
    )
