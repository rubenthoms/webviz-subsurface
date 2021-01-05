########################################
#
#  Copyright (C) 2020-     Equinor ASA
#
########################################

from typing import Tuple, Callable, List, Union, Optional

from opm.io.ecl import EclFile

from ..opm_unit import ConvertUnits, EclUnits, CreateUnitConverter, ErtEclUnitEnum
from .pvt_common import (
    surface_mass_density,
    InitFileDefinitions,
    EclPhaseIndex,
    PVTx,
    PVDx,
    PvxOBase,
    EclPropertyTableRawData,
    Implementation,
    MakeInterpolants,
)


class WetGas(PvxOBase):
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
        # Remember:
        # PKey   Inner   C0     C1         C2           C3
        # Pg     Rv      1/B    1/(B*mu)   d(1/B)/dRv   d(1/(B*mu))/dRv
        #        :       :      :          :            :
        return self.interpolant.formation_volume_factor(pressure, ratio)

    def viscosity(self, ratio: List[float], pressure: List[float]) -> List[float]:
        # Remember:
        # PKey   Inner   C0     C1         C2           C3
        # Pg     Rv      1/B    1/(B*mu)   d(1/B)/dRv   d(1/(B*mu))/dRv
        #        :       :      :          :            :
        return self.interpolant.viscosity(pressure, ratio)

    def get_keys(self) -> List[float]:
        return self.interpolant.get_keys()

    def get_independents(self) -> List[float]:
        return self.interpolant.get_independents()


class DryGas(PvxOBase):
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


class Gas(Implementation):
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
        self.create_pvt_function(raw, unit_system)

    def formation_volume_factor_unit(self) -> str:
        unit_system = EclUnits.create_unit_system(
            self.original_unit_system
            if self.keep_unit_system
            else ErtEclUnitEnum.ECL_SI_UNITS
        )

        return fr"${{r{unit_system.reservoir_volume().symbol}}}/{{s{unit_system.surface_volume_gas().symbol}}}$"

    def viscosity_unit(self) -> str:
        unit_system = EclUnits.create_unit_system(
            self.original_unit_system
            if self.keep_unit_system
            else ErtEclUnitEnum.ECL_SI_UNITS
        )

        return fr"${unit_system.viscosity().symbol}$"

    def create_pvt_function(
        self, raw: EclPropertyTableRawData, unit_system: int
    ) -> None:
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
            self.create_dry_gas(raw, unit_system)

        self.create_wet_gas(raw, unit_system)

    def dry_gas_unit_converter(
        self, unit_system: Union[int, EclUnits.UnitSystem]
    ) -> ConvertUnits:
        if not isinstance(unit_system, EclUnits.UnitSystem):
            unit_system = EclUnits.create_unit_system(unit_system)

        return super().pvdx_unit_converter() or ConvertUnits(
            CreateUnitConverter.ToSI.pressure(unit_system),
            [
                CreateUnitConverter.ToSI.recip_fvf_gas(unit_system),
                CreateUnitConverter.ToSI.recip_fvf_gas_visc(unit_system),
                CreateUnitConverter.ToSI.recip_fvf_gas_deriv_press(unit_system),
                CreateUnitConverter.ToSI.recip_fvf_gas_visc_deriv_press(unit_system),
            ],
        )

    def wet_gas_unit_converter(
        self, unit_system: Union[int, EclUnits.UnitSystem]
    ) -> Tuple[Callable[[float,], float,], ConvertUnits]:
        if not isinstance(unit_system, EclUnits.UnitSystem):
            unit_system = EclUnits.create_unit_system(unit_system)

        return super().pvtx_unit_converter() or (
            CreateUnitConverter.ToSI.pressure(unit_system),
            ConvertUnits(
                CreateUnitConverter.ToSI.vaporised_oil_gas_ratio(unit_system),
                [
                    CreateUnitConverter.ToSI.recip_fvf_gas(unit_system),
                    CreateUnitConverter.ToSI.recip_fvf_gas_visc(unit_system),
                    CreateUnitConverter.ToSI.recip_fvf_gas_deriv_vap_oil(unit_system),
                    CreateUnitConverter.ToSI.recip_fvf_gas_visc_deriv_vap_oil(
                        unit_system
                    ),
                ],
            ),
        )

    def create_dry_gas(self, raw: EclPropertyTableRawData, unit_system: int) -> None:
        cvrt = self.dry_gas_unit_converter(unit_system)

        self._regions = MakeInterpolants.from_raw_data(
            # Inner   C0     C1         C2           C3
            # Pg      1/B    1/(B*mu)   d(1/B)/dRv   d(1/(B*mu))/dRv
            #         :       :         :            :
            raw,
            lambda table_index, raw: DryGas(table_index, raw, cvrt),
        )

    def create_wet_gas(self, raw: EclPropertyTableRawData, unit_system: int) -> None:
        cvrt = self.wet_gas_unit_converter(unit_system)

        self._regions = MakeInterpolants.from_raw_data(
            # PKey   Inner   C0     C1         C2           C3
            # Pg     Rv      1/B    1/(B*mu)   d(1/B)/dRv   d(1/(B*mu))/dRv
            #        :       :      :          :            :
            raw,
            lambda table_index, raw: WetGas(table_index, raw, cvrt),
        )

    def is_wet_gas(self) -> bool:
        if len(self._regions) > 0:
            return isinstance(self._regions[0], WetGas)
        return False

    def is_dry_gas(self) -> bool:
        if len(self._regions) > 0:
            return isinstance(self._regions[0], DryGas)
        return False

    @staticmethod
    def from_ecl_init_file(
        ecl_init_file: EclFile, keep_unit_system: bool = False
    ) -> Optional["Gas"]:
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

        return Gas(
            raw,
            intehead[InitFileDefinitions.INTEHEAD_UNIT_INDEX],
            rhos,
            keep_unit_system,
        )
