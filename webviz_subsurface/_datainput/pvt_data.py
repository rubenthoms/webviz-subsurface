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
    # pylint: disable=too-many-branches
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
        "PRESSURE_UNIT",
        "VOLUMEFACTOR",
        "VOLUMEFACTOR_UNIT",
        "VISCOSITY",
        "VISCOSITY_UNIT",
    ]

    if not "RATIO" in data_frame.columns:
        raise ValueError(
            "The dataframe must contain a column for the ratio (OGR, GOR, R, RV, RS)."
        )
    if not "VOLUMEFACTOR_UNIT" in data_frame.columns:
        data_frame["VOLUMEFACTOR_UNIT"] = "rm^3/sm^3"
    if not "PRESSURE_UNIT" in data_frame.columns:
        data_frame["PRESSURE_UNIT"] = "bar"
    if not "VISCOSITY_UNIT" in data_frame.columns:
        data_frame["VISCOSITY_UNIT"] = "cP"

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

        # Keep the original unit system
        oil = Oil.from_ecl_init_file(ecl_init_file, True)
        gas = Gas.from_ecl_init_file(ecl_init_file, True)
        water = Water.from_ecl_init_file(ecl_init_file, True)

        column_pvtnum = []
        column_ratio = []
        column_volume_factor = []
        column_volume_factor_unit = []
        column_pressure = []
        column_pressure_unit = []
        column_viscosity = []
        column_viscosity_unit = []
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
            ratios.append(0.0)

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
                    column_pvtnum.extend([region_index for _ in pressures])
                    column_keyword.extend([keyword for _ in pressures])
                    column_ratio.extend(ratios)
                    column_pressure.extend(pressures)
                    column_pressure_unit.extend(
                        [oil.pressure_unit() for _ in pressures]
                    )
                    column_volume_factor.extend(
                        region.formation_volume_factor(ratios, pressures)
                    )
                    column_volume_factor_unit.extend(
                        [oil.formation_volume_factor_unit() for _ in pressures]
                    )
                    column_viscosity.extend(region.viscosity(ratios, pressures))
                    column_viscosity_unit.extend(
                        [oil.viscosity_unit() for _ in pressures]
                    )

                else:
                    (ratio, pressure) = (
                        region.get_keys(),
                        region.get_independents(),
                    )
                    column_pvtnum.extend([region_index for _ in pressure])
                    column_keyword.extend([keyword for _ in pressure])
                    column_ratio.extend(ratio)
                    column_pressure.extend(pressure)
                    column_pressure_unit.extend([oil.pressure_unit() for _ in pressure])
                    column_volume_factor.extend(
                        region.formation_volume_factor(ratio, pressure)
                    )
                    column_volume_factor_unit.extend(
                        [oil.formation_volume_factor_unit() for _ in pressure]
                    )
                    column_viscosity.extend(region.viscosity(ratio, pressure))
                    column_viscosity_unit.extend(
                        [oil.viscosity_unit() for _ in pressure]
                    )

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
                (pressure, ratio) = (
                    region.get_keys(),
                    region.get_independents(),
                )
                column_pvtnum.extend([region_index for _ in pressure])
                column_keyword.extend([keyword for _ in pressure])
                column_ratio.extend(ratio)
                column_pressure.extend(pressure)
                column_pressure_unit.extend([gas.pressure_unit() for _ in pressure])
                column_volume_factor.extend(
                    region.formation_volume_factor(ratio, pressure)
                )
                column_volume_factor_unit.extend(
                    [gas.formation_volume_factor_unit() for _ in pressure]
                )
                column_viscosity.extend(region.viscosity(ratio, pressure))
                column_viscosity_unit.extend([gas.viscosity_unit() for _ in pressure])

        if water:
            for region_index, region in enumerate(water.regions()):
                column_pvtnum.extend([region_index for _ in pressures])
                column_keyword.extend(["PVTW" for _ in pressures])
                column_ratio.extend(ratios)
                column_pressure.extend(pressures)
                column_pressure_unit.extend([water.pressure_unit() for _ in pressures])
                column_volume_factor.extend(
                    region.formation_volume_factor(ratios, pressures)
                )
                column_volume_factor_unit.extend(
                    [water.formation_volume_factor_unit() for _ in pressures]
                )
                column_viscosity.extend(region.viscosity(ratios, pressures))
                column_viscosity_unit.extend(
                    [water.viscosity_unit() for _ in pressures]
                )

        data_frame = pd.DataFrame(
            {
                "PVTNUM": column_pvtnum,
                "KEYWORD": column_keyword,
                "R": column_ratio,
                "PRESSURE": column_pressure,
                "PRESSURE_UNIT": column_pressure_unit,
                "VOLUMEFACTOR": column_volume_factor,
                "VOLUMEFACTOR_UNIT": column_volume_factor_unit,
                "VISCOSITY": column_viscosity,
                "VISCOSITY_UNIT": column_viscosity_unit,
            }
        )

        return data_frame

    return filter_pvt_data_frame(
        load_ensemble_set(ensemble_paths, ensemble_set_name).apply(
            init_to_pvt_data_frame if use_init_file else ecl2df_pvt_data_frame
        ),
        drop_ensemble_duplicates,
    )
