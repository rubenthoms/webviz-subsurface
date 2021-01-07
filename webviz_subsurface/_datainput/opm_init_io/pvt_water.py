########################################
#
#  Copyright (C) 2020-     Equinor ASA
#
########################################

from typing import Callable, List, Any, Union, Optional

from opm.io.ecl import EclFile

from ..opm_unit import ConvertUnits, EclUnits, CreateUnitConverter, ErtEclUnitEnum
from .pvt_common import (
    surface_mass_density,
    InitFileDefinitions,
    EclPhaseIndex,
    PvxOBase,
    EclPropertyTableRawData,
    FluidImplementation,
)


class WaterImpl(PvxOBase):
    def __init__(
        self,
        index_table: int,
        raw: EclPropertyTableRawData,
        convert: ConvertUnits,
    ) -> None:
        PvxOBase.__init__(self)

        column_stride = raw.num_rows * raw.num_tables * raw.num_primary
        table_stride = raw.num_primary * raw.num_rows
        current_stride = index_table * table_stride

        # [ Pw, 1/Bw, Cw, 1/(Bw*mu_w), Cw - Cv ]
        self.__pw_ref = convert.independent(raw.data[current_stride])
        self.__recip_fvf = convert.column[0](
            raw.data[current_stride + 1 * column_stride]
        )
        self.__c_w = convert.column[1](raw.data[current_stride + 2 * column_stride])
        self.__recip_fvf_visc = convert.column[2](
            raw.data[current_stride + 3 * column_stride]
        )
        self.__diff_cw_cv = convert.column[3](
            raw.data[current_stride + 4 * column_stride]
        )

    def recip_fvf(self, p_w: float) -> float:
        x = self.__c_w * (p_w - self.__pw_ref)

        return self.__recip_fvf * self.__exp(x)

    def recip_fvf_visc(self, p_w: float) -> float:
        y = self.__diff_cw_cv * (p_w - self.__pw_ref)

        return self.__recip_fvf_visc * self.__exp(y)

    @staticmethod
    def __evaluate(
        pressures: List[float], calculate: Callable[[Any], Any]
    ) -> List[float]:
        quantities: List[float] = []

        for pressure in pressures:
            quantities.append(calculate(pressure))

        return quantities

    @staticmethod
    def __exp(x: float) -> float:
        return 1.0 + x * (1.0 + x / 2.0)

    def formation_volume_factor(
        self, ratio: List[float], pressure: List[float]
    ) -> List[float]:
        return self.__evaluate(pressure, lambda p: 1.0 / self.recip_fvf(p))

    def viscosity(self, ratio: List[float], pressure: List[float]) -> List[float]:
        return self.__evaluate(
            pressure, lambda p: self.recip_fvf(p) / self.recip_fvf_visc(p)
        )

    def get_keys(self) -> List[float]:
        return [
            0.0,
        ]

    def get_independents(self) -> List[float]:
        return [
            self.__pw_ref,
        ]


class Water(FluidImplementation):
    def __init__(
        self,
        raw: EclPropertyTableRawData,
        unit_system: int,
        rhos: List[float],
        keep_unit_system: bool = False,
    ):
        super().__init__(keep_unit_system)
        self.rhos = rhos
        self.original_unit_system = ErtEclUnitEnum(unit_system)
        self.create_water(raw, unit_system, rhos)

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

    def water_unit_converter(
        self, unit_system: Union[int, EclUnits.UnitSystem]
    ) -> ConvertUnits:
        if not isinstance(unit_system, EclUnits.UnitSystem):
            unit_system = EclUnits.create_unit_system(unit_system)

        return super().pvdx_unit_converter() or ConvertUnits(
            CreateUnitConverter.ToSI.pressure(unit_system),
            [
                CreateUnitConverter.ToSI.recip_fvf(unit_system),
                CreateUnitConverter.ToSI.compressibility(unit_system),
                CreateUnitConverter.ToSI.recip_fvf_visc(unit_system),
                CreateUnitConverter.ToSI.compressibility(unit_system),
            ],
        )

    def create_water(
        self, raw: EclPropertyTableRawData, unit_system: int, rhos: List[float]
    ) -> None:
        # Holding raw.num_tables values
        ret: List[PvxOBase] = []

        cvrt = self.water_unit_converter(unit_system)

        ret = self.make_interpolants_from_raw_data(
            raw,
            lambda table_index, raw: WaterImpl(table_index, raw, cvrt),
        )

        assert len(rhos) == len(ret)

        self._regions = ret

    @staticmethod
    def from_ecl_init_file(
        ecl_init_file: EclFile, keep_unit_system: bool = False
    ) -> Optional["Water"]:
        intehead = ecl_init_file.__getitem__(InitFileDefinitions.INTEHEAD_KW)
        intehead_phase = intehead[InitFileDefinitions.INTEHEAD_PHASE_INDEX]

        if (intehead_phase & (1 << 2)) == 0:
            return None

        raw = EclPropertyTableRawData()

        tab_dims = ecl_init_file.__getitem__("TABDIMS")
        tab = ecl_init_file.__getitem__("TAB")

        raw.num_primary = 1  # Single record per region
        raw.num_rows = 1  # Single record per region
        raw.num_cols = 5  # [ Pw, 1/B, Cw, 1/(B*mu), Cw - Cv ]
        raw.num_tables = tab_dims[InitFileDefinitions.TABDIMS_NTPVTW_ITEM]

        if raw.num_tables == 0:
            return None

        num_tab_elements = (
            raw.num_primary * raw.num_rows * raw.num_cols * raw.num_tables
        )
        start = tab_dims[InitFileDefinitions.TABDIMS_IBPVTW_OFFSET_ITEM] - 1
        raw.data = tab[start : start + num_tab_elements]

        rhos = surface_mass_density(ecl_init_file, EclPhaseIndex.Aqua)

        return Water(
            raw,
            intehead[InitFileDefinitions.INTEHEAD_UNIT_INDEX],
            rhos,
            keep_unit_system,
        )
