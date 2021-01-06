########################################
#
#  Copyright (C) 2020-     Equinor ASA
#
########################################

from typing import List, Callable, Tuple, Any, Optional
from enum import Enum

from scipy import interpolate
import numpy as np
from opm.io.ecl import EclFile

from ..opm_unit import ConvertUnits, EclUnits, ErtEclUnitEnum


class EclPropertyTableRawData:  # pylint: disable=too-few-public-methods
    """
    A structure for storing read
    INIT file data.
    """

    def __init__(self) -> None:
        self.data: List[float] = []
        self.primary_key: List[int] = []
        self.num_primary = 0
        self.num_rows = 0
        self.num_cols = 0
        self.num_tables = 0


class PvxOBase:
    """A common base class for all fluids.

    Should be inherited by any new fluid in order to have
    a common interface.
    """

    def __init__(self) -> None:
        """Base implementation, raises a NotImplementedError."""
        raise NotImplementedError("You must not create objects of this base class.")

    def get_keys(self) -> List[float]:
        """Base implementation, raises a NotImplementedError."""
        raise NotImplementedError("You must not call any methods of this base class.")

    def get_independents(self) -> List[float]:
        """Base implementation, raises a NotImplementedError."""
        raise NotImplementedError("You must not call any methods of this base class.")

    # pylint: disable=R0201
    def formation_volume_factor(
        self, ratio: List[float], pressure: List[float]
    ) -> List[float]:
        """Base implementation, raises a NotImplementedError."""
        raise NotImplementedError("You must not call any methods of this base class.")

    def viscosity(self, ratio: List[float], pressure: List[float]) -> List[float]:
        """Base implementation, raises a NotImplementedError."""
        raise NotImplementedError("You must not call any methods of this base class.")


def extrap1d(interpolator: interpolate.interp1d) -> Callable[[float], np.ndarray]:
    """Extends the given scipy interpolator by the ability to extrapolate linearly
    in case the given independent is out of range.

    Args:
        interpolator: A scipy interp1d object to be extended by linear extrapolation.

    Returns:
        A callable taking an x value and returns the related interpolated or
        linearly extrapolated y value.
    """
    x_s = interpolator.x
    y_s = interpolator.y

    def pointwise(x: float) -> np.ndarray:
        if x < x_s[0]:
            return y_s[0] + (x - x_s[0]) * (y_s[1] - y_s[0]) / (x_s[1] - x_s[0])
        elif x > x_s[-1]:
            return y_s[-1] + (x - x_s[-1]) * (y_s[-1] - y_s[-2]) / (x_s[-1] - x_s[-2])
        else:
            return interpolator(x)

    def ufunclike(x_s: float) -> np.ndarray:
        return np.ndarray(list(map(pointwise, np.ndarray(x_s))))

    return ufunclike


class PVDx:
    """A base class for dead and dry gas/oil respectively.

    Attributes:
        x: A numpy array holding the independent values.
        y: A two-dimensional numpy array holding the dependent values.
        interpolation: A scipy interp1d object for interpolating the dependent values.
        inter_extrapolation: An extrap1d object for inter- and extrapolating the dependent values.
    """

    def __init__(
        self,
        index_table: int,
        raw: EclPropertyTableRawData,
        convert: ConvertUnits,
    ) -> None:
        """Extracts all values of the table with the given index from raw, converts them according to the
        given convert object and stores them as numpy arrays, x and y respectively.

        Creates an interpolation and an extrapolation object utilising scipy's interp1d and based on it the
        custom tailored extrap1d.

        Raises a ValueError if there is no interpolation interval given, that is when there are fewer than two independents.

        Arguments:
            index_table: The index of the table which values are supposed to be extracted.
            raw: An EclPropertyTableRawData object that was initialised based on an Eclipse INIT file.
            convert: A ConvertUnit object that contains callables for converting units.
        """
        self.x: np.ndarray
        self.y: np.ndarray = np.zeros((0, 0))

        column_stride = raw.num_rows * raw.num_tables * raw.num_primary
        table_stride = raw.num_primary * raw.num_rows

        for index_primary in range(0, raw.num_primary):
            if self.entry_valid(raw.primary_key[index_primary]):
                for index_row in range(0, raw.num_rows):
                    current_stride = (
                        index_table * table_stride
                        + index_primary * raw.num_rows
                        + index_row
                    )

                    if self.entry_valid(raw.data[column_stride * 0 + current_stride]):
                        self.x = np.append(
                            self.x,
                            convert.independent(
                                raw.data[column_stride * 0 + current_stride]
                            ),
                        )

                        for index_column in range(1, raw.num_cols):
                            self.y[index_column] = np.append(
                                self.y[index_column],
                                convert.column[index_column - 1](
                                    raw.data[
                                        column_stride * index_column + current_stride
                                    ]
                                ),
                            )

                    else:
                        break
            else:
                break

        if len(self.x) < 2:
            raise ValueError("No Interpolation Interval of Non-Zero Size.")

        self.interpolation = interpolate.interp1d(self.x, self.y, axis=1)
        self.inter_extrapolation = extrap1d(self.interpolation)

    def get_keys(self) -> List[float]:
        """Returns a list of all primary keys.

        Since this is dry/dead gas/oil, there is no dependency on Rv/Rs.
        Hence, this method returns a list holding only a single float of value 0.0.
        """
        return [
            0.0,
        ]

    def get_independents(self) -> List[float]:
        """Returns a list of all independents.

        In case of gas/oil this returns a list of pressure values.
        """
        return self.x

    @staticmethod
    def entry_valid(x: float) -> bool:
        """Returns True if the given value is valid, i.e. >= 1.0e20, else False."""
        return abs(x) < 1.0e20

    def formation_volume_factor(self, pressure: List[float]) -> List[float]:
        """Returns a list of formation volume factor values corresponding
        to the given list of pressure values.
        """
        # 1 / (1 / B)
        return self.compute_quantity(pressure, lambda p: 1.0 / self.fvf_recip(p))

    def viscosity(self, pressure: List[float]) -> List[float]:
        """Returns a list of viscosity values corresponding to the given list of pressure values."""
        # (1 / B) / (1 / (B * mu)
        return self.compute_quantity(
            pressure, lambda p: self.fvf_recip(p) / self.fvf_mu_recip(p)
        )

    @staticmethod
    def compute_quantity(
        pressures: List[float], evaluate: Callable[[Any], Any]
    ) -> List[float]:
        """Calls the given evaluate function with each of the values
        in the given pressure list and returns a list of results.
        """
        result: List[float] = []

        for pressure in pressures:
            result.append(float(evaluate(pressure)))

        return result

    def fvf_recip(self, point: float) -> float:
        return float(self.inter_extrapolation(point)[0])

    def fvf_mu_recip(self, point: float) -> float:
        return float(self.inter_extrapolation(point)[1])


class PVTx:
    def __init__(
        self,
        index_table: int,
        raw: EclPropertyTableRawData,
        convert: Tuple[
            Callable[
                [
                    float,
                ],
                float,
            ],
            ConvertUnits,
        ],
    ) -> None:
        self.keys: np.ndarray = np.zeros(0)
        self.x: np.ndarray = np.zeros(0)
        self.y: List[np.ndarray] = [np.zeros(0) for _ in range(1, raw.num_cols)]

        column_stride = raw.num_rows * raw.num_tables * raw.num_primary
        table_stride = raw.num_primary * raw.num_rows

        for index_primary in range(0, raw.num_primary):
            if self.entry_valid(raw.primary_key[index_primary]):
                for index_row in range(0, raw.num_rows):
                    current_stride = (
                        index_table * table_stride
                        + index_primary * raw.num_rows
                        + index_row
                    )

                    if self.entry_valid(raw.data[column_stride * 0 + current_stride]):
                        self.keys = np.append(
                            self.keys, convert[0](raw.primary_key[index_primary])
                        )
                        self.x = np.append(
                            self.x,
                            convert[1].independent(
                                raw.data[column_stride * 0 + current_stride]
                            ),
                        )

                        for index_column in range(1, raw.num_cols):
                            self.y[index_column - 1] = np.append(
                                self.y[index_column - 1],
                                convert[1].column[index_column - 1](
                                    raw.data[
                                        column_stride * index_column + current_stride
                                    ]
                                ),
                            )

                    else:
                        break
            else:
                break

        self.interpolants: List[interpolate.interp2d] = []

        for index_column in range(0, raw.num_cols - 1):
            self.interpolants.append(
                interpolate.interp2d(self.keys, self.x, self.y[index_column])
            )

    def get_keys(self) -> List[float]:
        return self.keys

    def get_independents(self) -> List[float]:
        return self.x

    @staticmethod
    def entry_valid(x: float) -> bool:
        # Equivalent to ECLPiecewiseLinearInterpolant.hpp line 293
        # or ECLPvtOil.cpp line 458
        return abs(x) < 1.0e20

    def formation_volume_factor(self, key: List[float], x: List[float]) -> List[float]:
        return self.compute_quantity(
            key,
            x,
            # pylint: disable=unnecessary-lambda
            lambda curve, point: self.interpolants[0](curve, point),
            lambda recip_fvf: 1.0 / recip_fvf,
        )

    def viscosity(self, key: List[float], x: List[float]) -> List[float]:
        return self.compute_quantity(
            key,
            x,
            lambda curve, point: [
                self.interpolants[0](curve, point),
                self.interpolants[1](curve, point),
            ],
            lambda dense_vector: dense_vector[0] / dense_vector[1],
        )

    def compute_quantity(
        self,
        key: List[float],
        x: List[float],
        inner_function: Callable,
        outer_function: Callable,
    ) -> List[float]:
        results: List[float] = []

        num_vals = len(key)

        if len(x) != num_vals:
            raise ValueError(
                "Number of Inner Sampling Points Does Not Match Number of Outer Sampling Points."
            )

        for i in range(0, num_vals):
            quantity = self.evaluate(key[i], x[i], inner_function)

            results.append(float(outer_function(quantity)))

        return results

    @staticmethod
    def evaluate(key: float, x: float, func: Callable) -> np.ndarray:
        return func(key, x)


class InitFileDefinitions:  # pylint: disable=too-few-public-methods
    """
    A namespace for constant definitions for
    reading Eclipse INIT files.
    """

    LOGIHEAD_KW = "LOGIHEAD"
    INTEHEAD_KW = "INTEHEAD"
    INTEHEAD_UNIT_INDEX = 2
    INTEHEAD_PHASE_INDEX = 14
    LOGIHEAD_RS_INDEX = 0
    LOGIHEAD_RV_INDEX = 1
    TABDIMS_IBPVTO_OFFSET_ITEM = 6
    TABDIMS_JBPVTO_OFFSET_ITEM = 7
    TABDIMS_NRPVTO_ITEM = 8
    TABDIMS_NPPVTO_ITEM = 9
    TABDIMS_NTPVTO_ITEM = 10
    TABDIMS_IBPVTW_OFFSET_ITEM = 11
    TABDIMS_NTPVTW_ITEM = 12
    TABDIMS_IBPVTG_OFFSET_ITEM = 13
    TABDIMS_JBPVTG_OFFSET_ITEM = 14
    TABDIMS_NRPVTG_ITEM = 15
    TABDIMS_NPPVTG_ITEM = 16
    TABDIMS_NTPVTG_ITEM = 17
    LOGIHEAD_CONSTANT_OILCOMPR_INDEX = 39 - 1

    TABDIMS_IBDENS_OFFSET_ITEM = 18
    TABDIMS_NTDENS_ITEM = 19


class EclPhaseIndex(Enum):
    Aqua = 0
    Liquid = 1
    Vapour = 2


def is_const_compr_index() -> int:
    return InitFileDefinitions.LOGIHEAD_CONSTANT_OILCOMPR_INDEX


def surface_mass_density(ecl_file: EclFile, phase: EclPhaseIndex) -> List[float]:
    if phase is EclPhaseIndex.Liquid:
        col = 0
    elif phase is EclPhaseIndex.Aqua:
        col = 1
    elif phase is EclPhaseIndex.Vapour:
        col = 2
    else:
        col = -1

    if col == -1:
        raise AttributeError("Phase must be Liquid, Aqua or Vapour.")

    tabdims = ecl_file.__getitem__("TABDIMS")
    tab = ecl_file.__getitem__("TAB")

    start = tabdims[InitFileDefinitions.TABDIMS_IBDENS_OFFSET_ITEM] - 1
    nreg = tabdims[InitFileDefinitions.TABDIMS_NTDENS_ITEM]

    rho = [tab[start + nreg * (col + 0)], tab[start + nreg * (col + 1)]]

    return rho


class MakeInterpolants:
    @staticmethod
    def from_raw_data(
        raw: EclPropertyTableRawData,
        construct: Callable[[int, EclPropertyTableRawData], PvxOBase],
    ) -> List[PvxOBase]:
        interpolants: List[PvxOBase] = []

        for table_index in range(0, raw.num_tables):
            interpolants.append(construct(table_index, raw))

        return interpolants


class Implementation:
    class InvalidArgument(Exception):
        def __init__(self, message: str):
            self.message = message
            super().__init__(message)

    class InvalidType(Exception):
        def __init__(self) -> None:
            super().__init__("Invalid type. Only live oil/wet gas/water supported.")

    def __init__(self, keep_unit_system: bool = False) -> None:
        self._regions: List[PvxOBase] = []
        self.keep_unit_system = keep_unit_system
        self.original_unit_system = ErtEclUnitEnum.ECL_SI_UNITS

    def pvdx_unit_converter(self) -> Optional[ConvertUnits]:
        if self.keep_unit_system:
            return ConvertUnits(
                lambda x: x,
                [
                    lambda x: x,
                    lambda x: x,
                    lambda x: x,
                    lambda x: x,
                ],
            )
        else:
            return None

    def pvtx_unit_converter(
        self,
    ) -> Optional[Tuple[Callable[[float,], float,], ConvertUnits]]:
        if self.keep_unit_system:
            return (
                lambda x: x,
                ConvertUnits(
                    lambda x: x,
                    [
                        lambda x: x,
                        lambda x: x,
                        lambda x: x,
                        lambda x: x,
                    ],
                ),
            )
        else:
            return None

    def pressure_unit(self, latex: bool = False) -> str:
        unit_system = EclUnits.create_unit_system(
            self.original_unit_system
            if self.keep_unit_system
            else ErtEclUnitEnum.ECL_SI_UNITS
        )

        if latex:
            return fr"${unit_system.pressure().symbol}$"
        return f"{unit_system.pressure().symbol}"

    def formation_volume_factor(
        self, region_index: int, ratio: List[float], pressure: List[float]
    ) -> List[float]:
        self.validate_region_index(region_index)

        return self._regions[region_index].formation_volume_factor(ratio, pressure)

    def formation_volume_factor_unit(self, latex: bool = False) -> str:
        raise NotImplementedError("This method cannot be called from the base class.")

    def viscosity(
        self, region_index: int, ratio: List[float], pressure: List[float]
    ) -> List[float]:
        self.validate_region_index(region_index)

        return self._regions[region_index].viscosity(ratio, pressure)

    def viscosity_unit(self, latex: bool = False) -> str:
        raise NotImplementedError("This method cannot be called from the base class.")

    def get_region(self, region_index: int) -> PvxOBase:
        self.validate_region_index(region_index)

        return self._regions[region_index]

    def validate_region_index(self, region_index: int) -> None:
        if region_index >= len(self.regions()):
            if len(self.regions()) == 0:
                raise TypeError(
                    f"No oil PVT interpolant available in region {region_index + 1}."
                )

            raise TypeError(
                f"Region index {region_index + 1} outside valid range 0"
                f", ..., {len(self._regions) - 1}."
            )

    def regions(self) -> List[PvxOBase]:
        return self._regions

    @staticmethod
    def entry_valid(x: float) -> bool:
        # Equivalent to ECLPiecewiseLinearInterpolant.hpp line 293
        # or ECLPvtOil.cpp line 458
        return abs(x) < 1.0e20

    def range_ratio(self, region_index: int) -> Tuple[float, float]:
        region = self.get_region(region_index)
        return (min(region.get_keys()), max(region.get_keys()))

    def range_independent(self, region_index: int) -> Tuple[float, float]:
        region = self.get_region(region_index)
        return (min(region.get_independents()), max(region.get_independents()))
