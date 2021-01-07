########################################
#
#  Copyright (C) 2020-     Equinor ASA
#
########################################

from typing import Tuple, Callable, List, Any, Union, Optional

from opm.io.ecl import EclFile

from ..opm_unit import ConvertUnits, EclUnits, Unit, CreateUnitConverter, ErtEclUnitEnum
from .pvt_common import (
    is_const_compr_index,
    surface_mass_density,
    InitFileDefinitions,
    EclPhaseIndex,
    PVTx,
    PVDx,
    PvxOBase,
    EclPropertyTableRawData,
    FluidImplementation,
)


class LiveOil(PvxOBase):
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
        PvxOBase.__init__(self)
        self.interpolant = PVTx(index_table, raw, convert)

    def formation_volume_factor(
        self, ratio: List[float], pressure: List[float]
    ) -> List[float]:
        return self.interpolant.formation_volume_factor(ratio, pressure)

    def viscosity(self, ratio: List[float], pressure: List[float]) -> List[float]:
        return self.interpolant.viscosity(ratio, pressure)

    def get_keys(self) -> List[float]:
        return self.interpolant.get_keys()

    def get_independents(self) -> List[float]:
        return self.interpolant.get_independents()


class DeadOil(PvxOBase):
    def __init__(
        self,
        table_index: int,
        raw: EclPropertyTableRawData,
        convert: ConvertUnits,
    ) -> None:
        PvxOBase.__init__(self)
        self.interpolant = PVDx(table_index, raw, convert)

    def formation_volume_factor(
        self, ratio: List[float], pressure: List[float]
    ) -> List[float]:
        return self.interpolant.formation_volume_factor(pressure)

    def viscosity(self, ratio: List[float], pressure: List[float]) -> List[float]:
        return self.interpolant.viscosity(pressure)

    def get_keys(self) -> List[float]:
        return self.interpolant.get_keys()

    def get_independents(self) -> List[float]:
        return self.interpolant.get_independents()


class DeadOilConstCompr(PvxOBase):
    def __init__(
        self,
        index_table: int,
        raw: EclPropertyTableRawData,
        convert: ConvertUnits,
    ) -> None:
        PvxOBase.__init__(self)

        self.rhos = 0.0

        column_stride = raw.num_rows * raw.num_tables * raw.num_primary
        table_stride = raw.num_primary * raw.num_rows
        current_stride = index_table * table_stride

        self.fvf = convert.column[0](raw.data[0 * column_stride + current_stride])  # Bo
        self.c_o = convert.column[1](raw.data[1 * column_stride + current_stride])  # Co
        self.visc = convert.column[2](
            raw.data[2 * column_stride + current_stride]
        )  # mu_o
        self.c_v = convert.column[3](raw.data[3 * column_stride + current_stride])  # Cv

        self.p_o_ref = convert.independent(raw.data[current_stride])

        if abs(self.p_o_ref) < 1.0e20:
            raise ValueError("Invalid Input PVCDO Table")

    def formation_volume_factor(
        self, ratio: List[float], pressure: List[float]
    ) -> List[float]:
        return self.__evaluate(pressure, lambda p: 1.0 / self.__recip_fvf(p))

    def viscosity(self, ratio: List[float], pressure: List[float]) -> List[float]:
        return self.__evaluate(
            pressure, lambda p: self.__recip_fvf(p) / self.__recip_fvf_visc(p)
        )

    def __recip_fvf(self, p_o: float) -> float:
        x = self.c_o * (p_o - self.p_o_ref)

        return self.__exp(x) / self.fvf

    def surface_mass_density(self) -> float:
        return self.rhos

    def __recip_fvf_visc(self, p_o: float) -> float:
        y = (self.c_o - self.c_v) * (p_o - self.p_o_ref)

        return self.__exp(y) / (self.fvf * self.visc)

    @staticmethod
    def __exp(x: float) -> float:
        return 1.0 + x * (1.0 + x / 2.0)

    @staticmethod
    def __evaluate(
        pressures: List[float], calculate: Callable[[Any], Any]
    ) -> List[float]:
        quantities: List[float] = []

        for pressure in pressures:
            quantities.append(calculate(pressure))

        return quantities

    def get_keys(self) -> List[float]:
        return [
            0.0,
        ]

    def get_independents(self) -> List[float]:
        return [
            self.p_o_ref,
        ]


class Oil(FluidImplementation):
    def __init__(
        self,
        raw: EclPropertyTableRawData,
        unit_system: int,
        is_const_compr: bool,
        rhos: List[float],
        keep_unit_system: bool = False,
    ):
        super().__init__(keep_unit_system)
        self.rhos = rhos
        self.original_unit_system = ErtEclUnitEnum(unit_system)
        self.create_pvt_function(raw, is_const_compr, unit_system)

    def formation_volume_factor_unit(self, latex: bool = False) -> str:
        unit_system = EclUnits.create_unit_system(
            self.original_unit_system
            if self.keep_unit_system
            else ErtEclUnitEnum.ECL_SI_UNITS
        )

        if latex:
            return fr"${{r{unit_system.reservoir_volume().symbol}}}/{{s{unit_system.surface_volume_liquid().symbol}}}$"
        return f"r{unit_system.reservoir_volume().symbol}/s{unit_system.surface_volume_liquid().symbol}"

    def viscosity_unit(self, latex: bool = False) -> str:
        unit_system = EclUnits.create_unit_system(
            self.original_unit_system
            if self.keep_unit_system
            else ErtEclUnitEnum.ECL_SI_UNITS
        )

        if latex:
            return fr"${unit_system.viscosity().symbol}$"
        return f"{unit_system.viscosity().symbol}"

    @staticmethod
    def _formation_volume_factor(
        unit_system: EclUnits.UnitSystem,
    ) -> Callable[[float,], float]:
        scale = (
            unit_system.reservoir_volume().value
            / unit_system.surface_volume_liquid().value
        )

        return lambda x: Unit.Convert.from_(x, scale)

    @staticmethod
    def _viscosity(
        unit_system: EclUnits.UnitSystem,
    ) -> Callable[[float,], float]:
        scale = unit_system.viscosity().value

        return lambda x: Unit.Convert.from_(x, scale)

    def dead_oil_unit_converter(
        self, unit_system: Union[int, EclUnits.UnitSystem]
    ) -> ConvertUnits:
        if not isinstance(unit_system, EclUnits.UnitSystem):
            unit_system = EclUnits.create_unit_system(unit_system)

        return super().pvdx_unit_converter() or ConvertUnits(
            CreateUnitConverter.ToSI.pressure(unit_system),
            [
                CreateUnitConverter.ToSI.recip_fvf(unit_system),
                CreateUnitConverter.ToSI.recip_fvf_visc(unit_system),
                CreateUnitConverter.ToSI.recip_fvf_deriv_press(unit_system),
                CreateUnitConverter.ToSI.recip_fvf_visc_deriv_press(unit_system),
            ],
        )

    def pvcdo_unit_converter(
        self, unit_system: Union[int, EclUnits.UnitSystem]
    ) -> ConvertUnits:
        if not isinstance(unit_system, EclUnits.UnitSystem):
            unit_system = EclUnits.create_unit_system(unit_system)

        return super().pvdx_unit_converter() or ConvertUnits(
            CreateUnitConverter.ToSI.pressure(unit_system),
            [
                Oil._formation_volume_factor(unit_system),
                CreateUnitConverter.ToSI.compressibility(unit_system),
                Oil._viscosity(unit_system),
                CreateUnitConverter.ToSI.compressibility(unit_system),
            ],
        )

    def live_oil_unit_converter(
        self, unit_system: Union[int, EclUnits.UnitSystem]
    ) -> Tuple[Callable[[float,], float,], ConvertUnits]:
        if not isinstance(unit_system, EclUnits.UnitSystem):
            unit_system = EclUnits.create_unit_system(unit_system)

        return super().pvtx_unit_converter() or (
            CreateUnitConverter.ToSI.dissolved_gas_oil_ratio(unit_system),
            self.dead_oil_unit_converter(unit_system),
        )

    # pylint: disable=unused-argument
    def create_pvt_function(
        self, raw: EclPropertyTableRawData, is_const_compr: bool, unit_system: int
    ) -> None:
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
            self.create_dead_oil(raw, is_const_compr, unit_system)

        self.create_live_oil(raw, unit_system)

    def create_live_oil(self, raw: EclPropertyTableRawData, unit_system: int) -> None:
        cvrt = self.live_oil_unit_converter(unit_system)

        self._regions = self.make_interpolants_from_raw_data(
            # PKey   Inner   C0     C1         C2           C3
            # Rs     Po      1/B    1/(B*mu)   d(1/B)/dPo   d(1/(B*mu))/dPo
            #        :       :      :          :            :
            raw,
            lambda table_index, raw: LiveOil(table_index, raw, cvrt),
        )

    def create_dead_oil(
        self, raw: EclPropertyTableRawData, const_compr: bool, unit_system: int
    ) -> None:
        if const_compr:
            cvrt = self.pvcdo_unit_converter(unit_system)

            self._regions = self.make_interpolants_from_raw_data(
                raw,
                lambda table_index, raw: DeadOilConstCompr(table_index, raw, cvrt),
            )

        cvrt = self.dead_oil_unit_converter(unit_system)

        self._regions = self.make_interpolants_from_raw_data(
            raw, lambda table_index, raw: DeadOil(table_index, raw, cvrt)
        )

    def is_live_oil(self) -> bool:
        if len(self._regions) > 0:
            return isinstance(self._regions[0], LiveOil)
        return False

    def is_dead_oil(self) -> bool:
        if len(self._regions) > 0:
            return isinstance(self._regions[0], DeadOil)
        return False

    def is_dead_oil_const_compr(self) -> bool:
        if len(self._regions) > 0:
            return isinstance(self._regions[0], DeadOilConstCompr)
        return False

    @staticmethod
    def from_ecl_init_file(
        ecl_init_file: EclFile, keep_unit_system: bool = False
    ) -> Optional["Oil"]:
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
            intehead[InitFileDefinitions.INTEHEAD_UNIT_INDEX],
            is_is_const_compr,
            rhos,
            keep_unit_system,
        )
