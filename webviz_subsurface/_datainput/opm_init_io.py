########################################
#
#  Copyright (C) 2020-     Equinor ASA
#
########################################

from enum import Enum
from typing import List, Optional, Union, cast, Callable, Tuple, Any

from scipy import interpolate
import numpy as np

from opm.io.ecl import EclFile
from .opm_unit import UnitSystems, ErtEclUnitEnum, Unit


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


class EclUnits:
    class UnitSystem:
        @staticmethod
        def density() -> float:
            return 0.0

        @staticmethod
        def depth() -> float:
            return 0.0

        @staticmethod
        def pressure() -> float:
            return 0.0

        @staticmethod
        def reservoir_rate() -> float:
            return 0.0

        @staticmethod
        def reservoir_volume() -> float:
            return 0.0

        @staticmethod
        def surface_volume_gas() -> float:
            return 0.0

        @staticmethod
        def surface_volume_liquid() -> float:
            return 0.0

        @staticmethod
        def time() -> float:
            return 0.0

        @staticmethod
        def transmissibility() -> float:
            return 0.0

        @staticmethod
        def viscosity() -> float:
            return 0.0

        def dissolved_gas_oil_ratio(self) -> float:
            return self.surface_volume_gas() / self.surface_volume_liquid()

        def vaporised_oil_gas_ratio(self) -> float:
            return self.surface_volume_liquid() / self.surface_volume_gas()

    class EclMetricUnitSystem(UnitSystem):
        @staticmethod
        def density() -> float:
            return UnitSystems.Metric.Density

        @staticmethod
        def depth() -> float:
            return UnitSystems.Metric.Length

        @staticmethod
        def pressure() -> float:
            return UnitSystems.Metric.Pressure

        @staticmethod
        def reservoir_rate() -> float:
            return UnitSystems.Metric.ReservoirVolume / UnitSystems.Metric.Time

        @staticmethod
        def reservoir_volume() -> float:
            return UnitSystems.Metric.ReservoirVolume

        @staticmethod
        def surface_volume_gas() -> float:
            return UnitSystems.Metric.GasSurfaceVolume

        @staticmethod
        def surface_volume_liquid() -> float:
            return UnitSystems.Metric.LiquidSurfaceVolume

        @staticmethod
        def time() -> float:
            return UnitSystems.Metric.Time

        @staticmethod
        def transmissibility() -> float:
            return UnitSystems.Metric.Transmissibility

        @staticmethod
        def viscosity() -> float:
            return UnitSystems.Metric.Viscosity

    class EclFieldUnitSystem(UnitSystem):
        @staticmethod
        def density() -> float:
            return UnitSystems.Field.Density

        @staticmethod
        def depth() -> float:
            return UnitSystems.Field.Length

        @staticmethod
        def pressure() -> float:
            return UnitSystems.Field.Pressure

        @staticmethod
        def reservoir_rate() -> float:
            return UnitSystems.Field.ReservoirVolume / UnitSystems.Field.Time

        @staticmethod
        def reservoir_volume() -> float:
            return UnitSystems.Field.ReservoirVolume

        @staticmethod
        def surface_volume_gas() -> float:
            return UnitSystems.Field.GasSurfaceVolume

        @staticmethod
        def surface_volume_liquid() -> float:
            return UnitSystems.Field.LiquidSurfaceVolume

        @staticmethod
        def time() -> float:
            return UnitSystems.Field.Time

        @staticmethod
        def transmissibility() -> float:
            return UnitSystems.Field.Transmissibility

        @staticmethod
        def viscosity() -> float:
            return UnitSystems.Field.Viscosity

    class EclLabUnitSystem(UnitSystem):
        @staticmethod
        def density() -> float:
            return UnitSystems.Lab.Density

        @staticmethod
        def depth() -> float:
            return UnitSystems.Lab.Length

        @staticmethod
        def pressure() -> float:
            return UnitSystems.Lab.Pressure

        @staticmethod
        def reservoir_rate() -> float:
            return UnitSystems.Lab.ReservoirVolume / UnitSystems.Lab.Time

        @staticmethod
        def reservoir_volume() -> float:
            return UnitSystems.Lab.ReservoirVolume

        @staticmethod
        def surface_volume_gas() -> float:
            return UnitSystems.Lab.GasSurfaceVolume

        @staticmethod
        def surface_volume_liquid() -> float:
            return UnitSystems.Lab.LiquidSurfaceVolume

        @staticmethod
        def time() -> float:
            return UnitSystems.Lab.Time

        @staticmethod
        def transmissibility() -> float:
            return UnitSystems.Lab.Transmissibility

        @staticmethod
        def viscosity() -> float:
            return UnitSystems.Lab.Viscosity

    class EclPvtMUnitSystem(UnitSystem):
        @staticmethod
        def density() -> float:
            return UnitSystems.PVTM.Density

        @staticmethod
        def depth() -> float:
            return UnitSystems.PVTM.Length

        @staticmethod
        def pressure() -> float:
            return UnitSystems.PVTM.Pressure

        @staticmethod
        def reservoir_rate() -> float:
            return UnitSystems.PVTM.ReservoirVolume / UnitSystems.PVTM.Time

        @staticmethod
        def reservoir_volume() -> float:
            return UnitSystems.PVTM.ReservoirVolume

        @staticmethod
        def surface_volume_gas() -> float:
            return UnitSystems.PVTM.GasSurfaceVolume

        @staticmethod
        def surface_volume_liquid() -> float:
            return UnitSystems.PVTM.LiquidSurfaceVolume

        @staticmethod
        def time() -> float:
            return UnitSystems.PVTM.Time

        @staticmethod
        def transmissibility() -> float:
            return UnitSystems.PVTM.Transmissibility

        @staticmethod
        def viscosity() -> float:
            return UnitSystems.PVTM.Viscosity

    @staticmethod
    def create_unit_system(unit_system: int) -> UnitSystem:
        if unit_system == ErtEclUnitEnum.ECL_METRIC_UNITS:
            return EclUnits.EclMetricUnitSystem()
        elif unit_system == ErtEclUnitEnum.ECL_FIELD_UNITS:
            return EclUnits.EclFieldUnitSystem()
        elif unit_system == ErtEclUnitEnum.ECL_LAB_UNITS:
            return EclUnits.EclLabUnitSystem()
        elif unit_system == ErtEclUnitEnum.ECL_PVT_M_UNITS:
            return EclUnits.EclPvtMUnitSystem()
        else:
            raise ValueError("Unsupported Unit Convention.")


class ConvertUnits:
    def __init__(
        self,
        indepent: Callable[
            [
                float,
            ],
            float,
        ],
        column: List[
            Callable[
                [
                    float,
                ],
                float,
            ]
        ],
    ) -> None:
        self.independent: Callable[
            [
                float,
            ],
            float,
        ] = indepent
        self.column: List[
            Callable[
                [
                    float,
                ],
                float,
            ]
        ] = column


class CreateUnitConverter:
    @staticmethod
    def create_converter_to_SI(
        uscale: float,
    ) -> Callable[[float,], float]:
        return lambda quantity: Unit.Convert.from_(quantity, uscale)

    @staticmethod
    def rs_scale(unit_system: EclUnits.UnitSystem) -> float:
        # Rs = [sVolume(Gas) / sVolume(Liquid)]
        return unit_system.surface_volume_gas() / unit_system.surface_volume_liquid()

    @staticmethod
    def rv_scale(unit_system: EclUnits.UnitSystem) -> float:
        # Rv = [sVolume(Liq) / sVolume(Gas)]
        return unit_system.surface_volume_liquid() / unit_system.surface_volume_gas()

    @staticmethod
    def fvf_scale(unit_system: EclUnits.UnitSystem) -> float:
        # B = [rVolume / sVolume(Liquid)]
        return unit_system.reservoir_volume() / unit_system.surface_volume_liquid()

    @staticmethod
    def fvf_gas_scale(unit_system: EclUnits.UnitSystem) -> float:
        # B = [rVolume / sVolume(Gas)]
        return unit_system.reservoir_volume() / unit_system.surface_volume_gas()

    class ToSI:
        @staticmethod
        def density(
            unit_system: EclUnits.UnitSystem,
        ) -> Callable[[float,], float]:
            return CreateUnitConverter.create_converter_to_SI(unit_system.density())

        @staticmethod
        def pressure(
            unit_system: EclUnits.UnitSystem,
        ) -> Callable[[float,], float]:
            return CreateUnitConverter.create_converter_to_SI(unit_system.pressure())

        @staticmethod
        def compressibility(
            unit_system: EclUnits.UnitSystem,
        ) -> Callable[[float,], float]:
            return CreateUnitConverter.create_converter_to_SI(
                1.0 / unit_system.pressure()
            )

        @staticmethod
        def dissolved_gas_oil_ratio(
            unit_system: EclUnits.UnitSystem,
        ) -> Callable[[float,], float]:
            return CreateUnitConverter.create_converter_to_SI(
                CreateUnitConverter.rs_scale(unit_system)
            )

        @staticmethod
        def vaporised_oil_gas_ratio(
            unit_system: EclUnits.UnitSystem,
        ) -> Callable[[float,], float]:
            return CreateUnitConverter.create_converter_to_SI(
                CreateUnitConverter.rv_scale(unit_system)
            )

        @staticmethod
        def recip_fvf(
            unit_system: EclUnits.UnitSystem,
        ) -> Callable[[float,], float]:
            return CreateUnitConverter.create_converter_to_SI(
                1.0 / CreateUnitConverter.fvf_scale(unit_system)
            )

        @staticmethod
        def recip_fvf_deriv_press(
            unit_system: EclUnits.UnitSystem,
        ) -> Callable[[float,], float]:
            # d(1/B)/dp
            b_scale = CreateUnitConverter.fvf_scale(unit_system)
            p_scale = unit_system.pressure()

            return CreateUnitConverter.create_converter_to_SI(1.0 / (b_scale * p_scale))

        @staticmethod
        def recip_fvf_deriv_vap_oil(
            unit_system: EclUnits.UnitSystem,
        ) -> Callable[[float,], float]:
            # d(1/B)/dRv
            b_scale = CreateUnitConverter.fvf_scale(unit_system)
            rv_scale = CreateUnitConverter.rv_scale(unit_system)

            return CreateUnitConverter.create_converter_to_SI(
                1.0 / (b_scale * rv_scale)
            )

        @staticmethod
        def recip_fvf_visc(
            unit_system: EclUnits.UnitSystem,
        ) -> Callable[[float,], float]:
            b_scale = CreateUnitConverter.fvf_scale(unit_system)
            visc_scale = unit_system.viscosity()

            return CreateUnitConverter.create_converter_to_SI(
                1.0 / (b_scale * visc_scale)
            )

        @staticmethod
        def recip_fvf_visc_deriv_press(
            unit_system: EclUnits.UnitSystem,
        ) -> Callable[[float,], float]:
            # d(1/(B*mu))/dp
            b_scale = CreateUnitConverter.fvf_scale(unit_system)
            p_scale = unit_system.pressure()
            visc_scale = unit_system.viscosity()

            return CreateUnitConverter.create_converter_to_SI(
                1.0 / (b_scale * visc_scale * p_scale)
            )

        @staticmethod
        def recip_fvf_visc_deriv_vap_oil(
            unit_system: EclUnits.UnitSystem,
        ) -> Callable[[float,], float]:
            # d(1/(B*mu))/dRv
            b_scale = CreateUnitConverter.fvf_scale(unit_system)
            visc_scale = unit_system.viscosity()
            rv_scale = CreateUnitConverter.rv_scale(unit_system)

            return CreateUnitConverter.create_converter_to_SI(
                1.0 / (b_scale * visc_scale * rv_scale)
            )

        @staticmethod
        def recip_fvf_gas(
            unit_system: EclUnits.UnitSystem,
        ) -> Callable[[float,], float]:
            return CreateUnitConverter.create_converter_to_SI(
                1.0 / CreateUnitConverter.fvf_gas_scale(unit_system)
            )

        @staticmethod
        def recip_fvf_gas_deriv_press(
            unit_system: EclUnits.UnitSystem,
        ) -> Callable[[float,], float]:
            # d(1/B)/dp
            b_scale = CreateUnitConverter.fvf_gas_scale(unit_system)
            p_scale = unit_system.pressure()

            return CreateUnitConverter.create_converter_to_SI(1.0 / (b_scale * p_scale))

        @staticmethod
        def recip_fvf_gas_deriv_vap_oil(
            unit_system: EclUnits.UnitSystem,
        ) -> Callable[[float,], float]:
            # d(1/B)/dRv
            b_scale = CreateUnitConverter.fvf_gas_scale(unit_system)
            rv_scale = CreateUnitConverter.rv_scale(unit_system)

            return CreateUnitConverter.create_converter_to_SI(
                1.0 / (b_scale * rv_scale)
            )

        @staticmethod
        def recip_fvf_gas_visc(
            unit_system: EclUnits.UnitSystem,
        ) -> Callable[[float,], float]:
            b_scale = CreateUnitConverter.fvf_gas_scale(unit_system)
            visc_scale = unit_system.viscosity()

            return CreateUnitConverter.create_converter_to_SI(
                1.0 / (b_scale * visc_scale)
            )

        @staticmethod
        def recip_fvf_gas_visc_deriv_press(
            unit_system: EclUnits.UnitSystem,
        ) -> Callable[[float,], float]:
            # d(1/(B*mu))/dp
            b_scale = CreateUnitConverter.fvf_gas_scale(unit_system)
            p_scale = unit_system.pressure()
            visc_scale = unit_system.viscosity()

            return CreateUnitConverter.create_converter_to_SI(
                1.0 / (b_scale * visc_scale * p_scale)
            )

        @staticmethod
        def recip_fvf_gas_visc_deriv_vap_oil(
            unit_system: EclUnits.UnitSystem,
        ) -> Callable[[float,], float]:
            # d(1/(B*mu))/dRv
            b_scale = CreateUnitConverter.fvf_gas_scale(unit_system)
            visc_scale = unit_system.viscosity()
            rv_scale = CreateUnitConverter.rv_scale(unit_system)

            return CreateUnitConverter.create_converter_to_SI(
                1.0 / (b_scale * visc_scale * rv_scale)
            )


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


class VariateAndValues:  # pylint: disable=too-few-public-methods
    """
    A structure for holding a variate and
    multiple covariates.
    """

    def __init__(self) -> None:
        self.x = 0.0
        self.y: Union[List[float], List[VariateAndValues]]

    def get_values_as_floats(self) -> List[float]:
        assert len(self.y) > 0 and isinstance(self.y[0], float)
        return cast(List[float], self.y)


class PvxOBase:
    """
    A base class for all fluids.
    """

    def __init__(self, values: List[VariateAndValues]) -> None:
        self.values = values

    def get_values(self) -> List[VariateAndValues]:
        return self.values

    # pylint: disable=R0201
    def formation_volume_factor(
        self, ratio: List[float], pressure: List[float]
    ) -> List[float]:
        """
        Does only return an empty list for now.
        """
        if len(ratio) != len(pressure):
            raise ValueError("Rs / Rv and pressure arguments must be of same size.")
        return []

    def viscosity(self, ratio: List[float], pressure: List[float]) -> List[float]:
        pass


def extrap1d(interpolator: interpolate.interp1d) -> Callable[[float], np.ndarray]:
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
    def __init__(
        self,
        begin: int,
        raw: EclPropertyTableRawData,
        convert: ConvertUnits,
        columns: List[int],
    ) -> None:
        self.x: np.ndarray = np.zeros(raw.num_rows)
        self.y: np.ndarray = np.zeros((raw.num_rows, raw.num_cols))

        column_stride = raw.num_rows * raw.num_tables * raw.num_primary

        for row in range(begin, begin + raw.num_rows):
            if abs(raw.data[row]) < 1.0e20:
                self.x[row - begin] = convert.independent(raw.data[row])

                for col in range(0, len(columns)):
                    self.y[row, col] = convert.column[row](
                        raw.data[begin + column_stride * col + row]
                    )

        if len(self.x) < 2:
            raise ValueError("No Interpolation Interval of Non-Zero Size.")

        self.interpolation = interpolate.interp1d(self.x, self.y, axis=1)
        self.inter_extrapolation = extrap1d(self.interpolation)

    def formation_volume_factor(self, pressure: List[float]) -> List[float]:
        # 1 / (1 / B)
        return self.compute_quantity(pressure, lambda p: 1.0 / self.fvf_recip(p))

    def viscosity(self, pressure: List[float]) -> List[float]:
        # (1 / B) / (1 / (B * mu)
        return self.compute_quantity(
            pressure, lambda p: self.fvf_recip(p) / self.fvf_mu_recip(p)
        )

    @staticmethod
    def compute_quantity(
        pressure: List[float], evaluate: Callable[[Any], Any]
    ) -> List[float]:
        result: List[float] = []

        for p in pressure:
            result.append(evaluate(p))

        return result

    def fvf_recip(self, point: float) -> float:
        return self.inter_extrapolation(point)[0]

    def fvf_mu_recip(self, point: float) -> float:
        return self.inter_extrapolation(point)[1]


class PVTx:
    def __init__(
        self, keys: List[float], property_interpolants: List[interpolate.interp1d]
    ) -> None:
        self.keys = keys
        self.property_interpolants = property_interpolants

        if len(self.keys) != len(self.property_interpolants):
            raise ValueError(
                "Size of Key Table Does not Match Number of Sub-Table Interpolants."
            )

        if len(self.keys) < 2:
            raise ValueError(
                "Mixing-Dependent Property Evaluator Must Have At Least Two Inner Tables"
            )

    def formation_volume_factor(key: List[float], x: List[float]) -> List[float]:
        return self.compute_quantity(
            key, x, lambda curve, point: self.property_interpolants[curve](point)[0]
        )

    @staticmethod
    def compute_quantity(
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

            result.append(quantity)

        return result

    def evaluate(self, key: float, x: float, func: Callable) -> np.ndarray:
        


class LiveOil(PvxOBase):
    pass


class DeadOil(PvxOBase):
    def __init__(
        self,
        indep: int,
        raw: EclPropertyTableRawData,
        convert: ConvertUnits,
        columns: List[int],
    ) -> None:
        self.interpolant = PVDx(indep, raw, convert, columns)

    def formation_volume_factor(
        self, ratio: List[float], pressure: List[float]
    ) -> List[float]:
        return self.interpolant.formation_volume_factor(pressure)

    def viscosity(self, ratio: List[float], pressure: List[float]) -> List[float]:
        return self.interpolant.viscosity(pressure)


class DeadOilConstCompr(PvxOBase):
    def __init__(
        self,
        indep: int,
        raw: EclPropertyTableRawData,
        convert: ConvertUnits,
        columns: List[int],
    ) -> None:
        self.rhos = 0.0

        self.fvf = convert.column[0](raw.data[columns[0]])  # Bo
        self.c_o = convert.column[1](raw.data[columns[1]])  # Co
        self.visc = convert.column[2](raw.data[columns[2]])  # mu_o
        self.c_v = convert.column[3](raw.data[columns[3]])  # Cv

        self.p_o_ref = convert.independent(raw.data[indep])

        if abs(self.p_o_ref) < 1.0e20:
            raise ValueError("Invalid Input PVCDO Table")

    def formation_volume_factor(
        self, ratio: List[float], pressure: List[float]
    ) -> List[float]:
        return self._evaluate(pressure, lambda p: 1.0 / self._recip_fvf(p))

    def viscosity(self, ratio: List[float], pressure: List[float]) -> List[float]:
        return self._evaluate(
            pressure, lambda p: self._recip_fvf(p) / self._recip_fvf_visc(p)
        )

    def _recip_fvf(self, p_o: float) -> float:
        x = self.c_o * (p_o - self.p_o_ref)

        return self._exp(x) / self.fvf

    def surface_mass_density(self) -> float:
        return self.rhos

    def _recip_fvf_visc(self, p_o: float) -> float:
        y = (self.c_o - self.c_v) * (p_o - self.p_o_ref)

        return self._exp(y) / (self.fvf * self.visc)

    @staticmethod
    def _exp(x: float) -> float:
        return 1.0 + x * (1.0 + x / 2.0)

    @staticmethod
    def _evaluate(
        pressures: List[float], calculate: Callable[[Any], Any]
    ) -> List[float]:
        quantities: List[float] = []

        for pressure in pressures:
            quantities.append(calculate(pressure))

        return quantities


class WetGas(PvxOBase):
    pass


class DryGas(PvxOBase):
    pass


class WaterImpl(PvxOBase):
    pass


class MakeInterpolants:
    @staticmethod
    def from_raw_data(
        raw: EclPropertyTableRawData,
        construct: Callable[[int, EclPropertyTableRawData, List[int]], PvxOBase],
    ) -> List[PvxOBase]:
        interpolants: List[PvxOBase] = []

        num_interpolants = raw.num_tables * raw.num_primary
        col_stride = raw.num_rows * num_interpolants

        begin: int = 0
        columns: List[int] = [begin + col_stride]

        for _ in range(1, raw.num_cols):
            columns.append(columns[-1] + col_stride)

        for _ in range(0, num_interpolants):
            interpolants.append(construct(begin, raw, columns))
            begin += raw.num_rows
            columns = [x + raw.num_rows for x in columns]

        return interpolants


class Implementation:
    class InvalidArgument(Exception):
        def __init__(self, message: str):
            self.message = message
            super().__init__(message)

    class InvalidType(Exception):
        def __init__(self) -> None:
            super().__init__("Invalid type. Only live oil/wet gas/water supported.")

    def __init__(self) -> None:
        self.values: List[PvxOBase] = []

    def get_table(self, tab_index: int) -> Optional[PvxOBase]:
        if tab_index in range(0, len(self.values)):
            return self.values[tab_index]

        return None

    def tables(self) -> List[PvxOBase]:
        return self.values

    @staticmethod
    def entry_valid(x: float) -> bool:
        # Equivalent to ECLPiecewiseLinearInterpolant.hpp line 293
        # or ECLPvtOil.cpp line 458
        return abs(x) < 1.0e20


class Oil(Implementation):
    def __init__(
        self,
        raw: EclPropertyTableRawData,
        unit_system: int,
        is_const_compr: bool,
        rhos: List[float],
    ):
        super().__init__()
        self.rhos = rhos
        self.values = self.create_pvt_function(raw, is_const_compr, unit_system)

    @staticmethod
    def _formation_volume_factor(
        unit_system: EclUnits.UnitSystem,
    ) -> Callable[[float,], float]:
        scale = unit_system.reservoir_volume() / unit_system.surface_volume_liquid()

        return lambda x: Unit.Convert.from_(x, scale)

    @staticmethod
    def _viscosity(
        unit_system: EclUnits.UnitSystem,
    ) -> Callable[[float,], float]:
        scale = unit_system.viscosity()

        return lambda x: Unit.Convert.from_(x, scale)

    @staticmethod
    def dead_oil_unit_converter(
        unit_system: Union[int, EclUnits.UnitSystem]
    ) -> ConvertUnits:
        if isinstance(unit_system, int):
            unit_system = EclUnits.create_unit_system(unit_system)

        return ConvertUnits(
            CreateUnitConverter.ToSI.pressure(unit_system),
            [
                CreateUnitConverter.ToSI.recip_fvf(unit_system),
                CreateUnitConverter.ToSI.recip_fvf_visc(unit_system),
                CreateUnitConverter.ToSI.recip_fvf_deriv_press(unit_system),
                CreateUnitConverter.ToSI.recip_fvf_visc_deriv_press(unit_system),
            ],
        )

    @staticmethod
    def pvcdo_unit_converter(
        unit_system: Union[int, EclUnits.UnitSystem]
    ) -> ConvertUnits:
        if isinstance(unit_system, int):
            unit_system = EclUnits.create_unit_system(unit_system)

        return ConvertUnits(
            CreateUnitConverter.ToSI.pressure(unit_system),
            [
                Oil._formation_volume_factor(unit_system),
                CreateUnitConverter.ToSI.compressibility(unit_system),
                Oil._viscosity(unit_system),
                CreateUnitConverter.ToSI.compressibility(unit_system),
            ],
        )

    @staticmethod
    def live_oil_unit_converter(
        unit_system: Union[int, EclUnits.UnitSystem]
    ) -> Tuple[Callable[[float,], float,], ConvertUnits]:
        if isinstance(unit_system, int):
            unit_system = EclUnits.create_unit_system(unit_system)

        return (
            CreateUnitConverter.ToSI.dissolved_gas_oil_ratio(unit_system),
            Oil.dead_oil_unit_converter(unit_system),
        )

    # pylint: disable=unused-argument
    def create_pvt_function(
        self, raw: EclPropertyTableRawData, is_const_compr: bool, unit_system: int
    ) -> List[PvxOBase]:
        if raw.num_primary == 0:
            raise super().InvalidArgument("Oil PVT table without primary lookup key")
        if raw.num_cols != 5:
            raise super().InvalidArgument("PVT table for oil must have five columns")
        if len(raw.primary_key) != (raw.num_primary * raw.num_tables):
            raise super().InvalidArgument(
                "Size mismatch in RS nodes of PVT table for oil"
            )
        if len(raw.data) != (
            raw.num_primary * raw.num_rows * raw.num_cols * raw.num_tables
        ):
            raise super().InvalidArgument(
                "Size mismatch in Condensed table data of PVT table for oil"
            )

        if raw.num_primary == 1:
            return self.create_dead_oil(raw, is_const_compr, unit_system)

        return self.create_live_oil(raw, unit_system)

    def create_live_oil(
        self, raw: EclPropertyTableRawData, unit_system: int
    ) -> List[PvxOBase]:
        # Holding raw.num_tables values
        ret: List[PvxOBase] = []

        column_stride = raw.num_rows * raw.num_tables * raw.num_primary
        table_stride = raw.num_primary * raw.num_rows

        # pylint: disable=too-many-nested-blocks
        for index_table in range(0, raw.num_tables):
            values = []

            # PKey   Inner   C0     C1         C2           C3
            # Rs     Po      1/B    1/(B*mu)   d(1/B)/dPo   d(1/(B*mu))/dPo
            #        :       :      :          :            :

            for index_primary in range(0, raw.num_primary):
                if self.entry_valid(raw.primary_key[index_primary]):
                    outer_value_pair = VariateAndValues()
                    outer_value_pair.x = raw.primary_key[index_primary]
                    # TODO(Ruben): Is there a better way to achieve this?
                    temp_list: List[VariateAndValues] = []
                    outer_value_pair.y = temp_list
                    for index_row in range(0, raw.num_rows):
                        pressure = raw.data[
                            column_stride * 0
                            + index_table * table_stride
                            + index_primary * raw.num_rows
                            + index_row
                        ]
                        if self.entry_valid(pressure):
                            inner_value_pair = VariateAndValues()
                            inner_value_pair.x = pressure
                            inner_value_pair.y = [0.0 for col in range(1, raw.num_cols)]
                            for index_column in range(1, raw.num_cols):
                                inner_value_pair.y[index_column - 1] = raw.data[
                                    column_stride * index_column
                                    + index_table * table_stride
                                    + index_primary * raw.num_rows
                                    + index_row
                                ]
                            outer_value_pair.y.append(inner_value_pair)
                        else:
                            break
                else:
                    break

                values.append(outer_value_pair)
            ret.append(LiveOil(values))

        return ret

    def create_dead_oil(
        self, raw: EclPropertyTableRawData, const_compr: bool, unit_system: int
    ) -> List[PvxOBase]:
        # Holding raw.num_tables values
        ret: List[PvxOBase] = []

        if const_compr:
            cvrt = self.pvcdo_unit_converter(unit_system)

            return MakeInterpolants.from_raw_data(
                raw,
                lambda indep, raw, columns: DeadOilConstCompr(
                    indep, raw, cvrt, columns
                ),
            )

        cvrt = self.dead_oil_unit_converter(unit_system)

        return MakeInterpolants.from_raw_data(
            raw, lambda indep, raw, columns: DeadOil(indep, raw, cvrt, columns)
        )

    @staticmethod
    def from_ecl_init_file(ecl_init_file: EclFile) -> Optional["Oil"]:
        intehead = ecl_init_file.__getitem__(InitFileDefinitions.INTEHEAD_KW)

        logihead = ecl_init_file.__getitem__(InitFileDefinitions.LOGIHEAD_KW)
        is_is_const_compr = bool(logihead[is_const_compr_index()])

        raw = EclPropertyTableRawData()

        tab_dims = ecl_init_file.__getitem__("TABDIMS")
        tab = ecl_init_file.__getitem__("TAB")

        num_rs = tab_dims[InitFileDefinitions.TABDIMS_NRPVTO_ITEM]

        raw.num_rows = tab_dims[InitFileDefinitions.TABDIMS_NPPVTO_ITEM]
        raw.num_cols = 5
        raw.num_tables = tab_dims[InitFileDefinitions.TABDIMS_NTPVTO_ITEM]

        if raw.num_tables == 0:
            return None

        # If there is no Rs index, no gas is dissolved in the oil -> dead oil
        # Otherwise, gas is dissolved and the ratio is given -> live oil
        if logihead[InitFileDefinitions.LOGIHEAD_RS_INDEX]:
            raw.num_primary = num_rs

        else:
            raw.num_primary = 1

        num_tab_elements = raw.num_primary * raw.num_tables
        start = tab_dims[InitFileDefinitions.TABDIMS_JBPVTO_OFFSET_ITEM] - 1
        raw.primary_key = tab[start : start + num_tab_elements]

        num_tab_elements = (
            raw.num_primary * raw.num_rows * raw.num_cols * raw.num_tables
        )
        start = tab_dims[InitFileDefinitions.TABDIMS_IBPVTO_OFFSET_ITEM] - 1
        raw.data = tab[start : start + num_tab_elements]

        rhos = surface_mass_density(ecl_init_file, EclPhaseIndex.Liquid)

        return Oil(
            raw,
            is_is_const_compr,
            intehead[InitFileDefinitions.INTEHEAD_UNIT_INDEX],
            rhos,
        )


class Gas(Implementation):
    def __init__(
        self, raw: EclPropertyTableRawData, unit_system: int, rhos: List[float]
    ):
        super().__init__()
        self.rhos = rhos
        self.values = self.create_pvt_function(raw, unit_system)

    def create_pvt_function(
        self, raw: EclPropertyTableRawData, unit_system: int
    ) -> List[PvxOBase]:
        if raw.num_primary == 0:
            raise super().InvalidArgument("Gas PVT table without primary lookup key")
        if raw.num_cols != 5:
            raise super().InvalidArgument("PVT table for gas must have five columns")
        if len(raw.primary_key) != (raw.num_primary * raw.num_tables):
            raise super().InvalidArgument(
                "Size mismatch in Pressure nodes of PVT table for gas"
            )
        if len(raw.data) != (
            raw.num_primary * raw.num_rows * raw.num_cols * raw.num_tables
        ):
            raise super().InvalidArgument(
                "Size mismatch in Condensed table data of PVT table for gas"
            )

        if raw.num_primary == 1:
            return self.create_dry_gas(raw, unit_system)

        return self.create_wet_gas(raw, unit_system)

    def create_dry_gas(
        self, raw: EclPropertyTableRawData, unit_system: int
    ) -> List[PvxOBase]:
        # Holding raw.num_tables values
        ret: List[PvxOBase] = []

        column_stride = raw.num_rows * raw.num_tables * raw.num_primary
        table_stride = raw.num_primary * raw.num_rows

        # pylint: disable=too-many-nested-blocks
        for index_table in range(0, raw.num_tables):
            values = []

            # Key     C0     C1         C2           C3
            # Pg      1/B    1/(B*mu)   d(1/B)/dRv   d(1/(B*mu))/dRv
            # :       :      :          NaN          NaN

            for index_primary in range(0, raw.num_primary):
                if self.entry_valid(raw.primary_key[index_primary]):
                    outer_value_pair = VariateAndValues()
                    outer_value_pair.x = raw.primary_key[index_primary]
                    # TODO(Ruben): Is there a better way to achieve this?
                    temp_list: List[VariateAndValues] = []
                    outer_value_pair.y = temp_list
                    for index_row in range(0, raw.num_rows):
                        pressure = raw.data[
                            column_stride * 0
                            + index_table * table_stride
                            + index_primary * raw.num_rows
                            + index_row
                        ]
                        if self.entry_valid(pressure):
                            inner_value_pair = VariateAndValues()
                            inner_value_pair.x = pressure
                            inner_value_pair.y = [0.0 for col in range(1, raw.num_cols)]
                            for index_column in range(1, raw.num_cols):
                                inner_value_pair.y[index_column - 1] = raw.data[
                                    column_stride * index_column
                                    + index_table * table_stride
                                    + index_primary * raw.num_rows
                                    + index_row
                                ]
                            outer_value_pair.y.append(inner_value_pair)
                        else:
                            break
                else:
                    break

                values.append(outer_value_pair)

            ret.append(DryGas(values))

        return ret

    def create_wet_gas(
        self, raw: EclPropertyTableRawData, unit_system: int
    ) -> List[PvxOBase]:
        # Holding raw.num_tables values
        ret: List[PvxOBase] = []

        column_stride = raw.num_rows * raw.num_tables * raw.num_primary
        table_stride = raw.num_primary * raw.num_rows

        # pylint: disable=too-many-nested-blocks
        for index_table in range(0, raw.num_tables):
            values = []

            # PKey   Inner   C0     C1         C2           C3
            # Pg     Rv      1/B    1/(B*mu)   d(1/B)/dRv   d(1/(B*mu))/dRv
            #        :       :      :          :            :

            for index_primary in range(0, raw.num_primary):
                if self.entry_valid(raw.primary_key[index_primary]):
                    outer_value_pair = VariateAndValues()
                    outer_value_pair.x = raw.primary_key[index_primary]
                    # TODO(Ruben): Is there a better way to achieve this?
                    temp_list: List[VariateAndValues] = []
                    outer_value_pair.y = temp_list
                    for index_row in range(0, raw.num_rows):
                        r_v = raw.data[
                            column_stride * 0
                            + index_table * table_stride
                            + index_primary * raw.num_rows
                            + index_row
                        ]
                        if self.entry_valid(r_v):
                            inner_value_pair = VariateAndValues()
                            inner_value_pair.x = r_v
                            inner_value_pair.y = [0.0 for col in range(1, raw.num_cols)]
                            for index_column in range(1, raw.num_cols):
                                inner_value_pair.y[index_column - 1] = raw.data[
                                    column_stride * index_column
                                    + index_table * table_stride
                                    + index_primary * raw.num_rows
                                    + index_row
                                ]
                            outer_value_pair.y.append(inner_value_pair)
                        else:
                            break
                else:
                    break

                values.append(outer_value_pair)

            ret.append(WetGas(values))

        return ret

    @staticmethod
    def from_ecl_init_file(ecl_init_file: EclFile) -> Optional["Gas"]:
        intehead = ecl_init_file.__getitem__(InitFileDefinitions.INTEHEAD_KW)
        intehead_phase = intehead[InitFileDefinitions.INTEHEAD_PHASE_INDEX]

        if (intehead_phase & (1 << 2)) == 0:
            return None

        raw = EclPropertyTableRawData()

        tab_dims = ecl_init_file.__getitem__("TABDIMS")
        tab = ecl_init_file.__getitem__("TAB")

        num_rv = tab_dims[InitFileDefinitions.TABDIMS_NRPVTG_ITEM]
        num_pg = tab_dims[InitFileDefinitions.TABDIMS_NPPVTG_ITEM]

        raw.num_cols = 5
        raw.num_tables = tab_dims[InitFileDefinitions.TABDIMS_NTPVTG_ITEM]

        if raw.num_tables == 0:
            return None

        logihead = ecl_init_file.__getitem__(InitFileDefinitions.LOGIHEAD_KW)

        if logihead[InitFileDefinitions.LOGIHEAD_RV_INDEX]:
            raw.num_primary = num_pg
            raw.num_rows = num_rv

        else:
            raw.num_primary = 1
            raw.num_rows = num_pg

        num_tab_elements = raw.num_primary * raw.num_tables
        start = tab_dims[InitFileDefinitions.TABDIMS_JBPVTG_OFFSET_ITEM] - 1
        raw.primary_key = tab[start : start + num_tab_elements]

        num_tab_elements = (
            raw.num_primary * raw.num_rows * raw.num_cols * raw.num_tables
        )
        start = tab_dims[InitFileDefinitions.TABDIMS_IBPVTG_OFFSET_ITEM] - 1
        raw.data = tab[start : start + num_tab_elements]

        rhos = surface_mass_density(ecl_init_file, EclPhaseIndex.Vapour)

        return Gas(raw, intehead[InitFileDefinitions.INTEHEAD_UNIT_INDEX], rhos)


class Water(Implementation):
    def __init__(
        self, raw: EclPropertyTableRawData, unit_system: int, rhos: List[float]
    ):
        super().__init__()
        self.rhos = rhos
        self.values = self.create_water(raw, unit_system)

    def create_water(
        self, raw: EclPropertyTableRawData, unit_system: int
    ) -> List[PvxOBase]:
        # Holding raw.num_tables values
        ret: List[PvxOBase] = []

        column_stride = raw.num_rows * raw.num_tables * raw.num_primary
        table_stride = raw.num_primary * raw.num_rows

        for index_table in range(0, raw.num_tables):
            values = []

            index_primary = 0
            outer_value_pair = VariateAndValues()
            outer_value_pair.x = 0
            # TODO(Ruben): Is there a better way to achieve this?
            temp_list: List[VariateAndValues] = []
            outer_value_pair.y = temp_list
            for index_row in range(0, raw.num_rows):
                pressure = raw.data[
                    column_stride * 0
                    + index_table * table_stride
                    + index_primary * raw.num_rows
                    + index_row
                ]
                if self.entry_valid(pressure):
                    inner_value_pair = VariateAndValues()
                    inner_value_pair.x = pressure
                    inner_value_pair.y = [0.0 for col in range(1, raw.num_cols)]
                    for index_column in range(1, raw.num_cols):
                        inner_value_pair.y[index_column - 1] = raw.data[
                            column_stride * index_column
                            + index_table * table_stride
                            + index_primary * raw.num_rows
                            + index_row
                        ]
                    outer_value_pair.y.append(inner_value_pair)
                else:
                    break

                values.append(outer_value_pair)

            ret.append(WaterImpl(values))

        return ret

    @staticmethod
    def from_ecl_init_file(ecl_init_file: EclFile) -> Optional["Water"]:
        intehead = ecl_init_file.__getitem__(InitFileDefinitions.INTEHEAD_KW)
        intehead_phase = intehead[InitFileDefinitions.INTEHEAD_PHASE_INDEX]

        if (intehead_phase & (1 << 2)) == 0:
            return None

        raw = EclPropertyTableRawData()

        tab_dims = ecl_init_file.__getitem__("TABDIMS")
        tab = ecl_init_file.__getitem__("TAB")

        raw.num_primary = 1
        raw.num_rows = 1
        raw.num_cols = 5
        raw.num_tables = tab_dims[InitFileDefinitions.TABDIMS_NTPVTG_ITEM]

        if raw.num_tables == 0:
            return None

        num_tab_elements = (
            raw.num_primary * raw.num_rows * raw.num_cols * raw.num_tables
        )
        start = tab_dims[InitFileDefinitions.TABDIMS_IBPVTW_OFFSET_ITEM] - 1
        raw.data = tab[start : start + num_tab_elements]

        rhos = surface_mass_density(ecl_init_file, EclPhaseIndex.Aqua)

        return Water(raw, intehead[InitFileDefinitions.INTEHEAD_UNIT_INDEX], rhos)
