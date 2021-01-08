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
    """Class holding a PVT interpolant and access methods for PVT data for water"""

    def __init__(
        self,
        index_table: int,
        raw: EclPropertyTableRawData,
        convert: ConvertUnits,
    ) -> None:
        # pylint: disable=super-init-not-called
        """Initializes a Water object.

        Creates an interpolant for Water for the Eclipse data given
        by the raw data in the table with the given index.

        Args:
            index_table: Index of the PVT table
            raw: Eclipse raw data
            convert: Tuple holding a callable and a ConvertUnits object for unit conversions

        """
        column_stride = raw.num_rows * raw.num_tables * raw.num_primary
        table_stride = raw.num_primary * raw.num_rows
        current_stride = index_table * table_stride

        # [ Pw, 1/Bw, Cw, 1/(Bw*mu_w), Cw - Cv ]
        self.__pw_ref = convert.independent(raw.data[current_stride])
        self.__recip_fvf_ref = convert.column[0](
            raw.data[current_stride + 1 * column_stride]
        )
        self.__c_w_ref = convert.column[1](raw.data[current_stride + 2 * column_stride])
        self.__recip_fvf_visc_ref = convert.column[2](
            raw.data[current_stride + 3 * column_stride]
        )
        self.__diff_cw_cv_ref = convert.column[3](
            raw.data[current_stride + 4 * column_stride]
        )

    def __recip_fvf(self, p_w: float) -> float:
        """Computes the reciproke of the formation volume factor for the given water pressure.

        Args:
            p_w: Water pressure

        Returns:
            Reciproke of the formation volume factor

        """

        # See Eclipse Reference Manual, p. 1762
        # B_w(P) = (B_w(P_ref)) / (1 + x + (x^2 / 2))
        # x = C * (P - P_ref)
        # NOTE: Don't forget that all values in INIT files are stored as reciprokes!
        # 1 / B_w(P) = 1 / B_w(P_ref) * (1 + x + x^2 / 2)

        x = self.__c_w_ref * (p_w - self.__pw_ref)

        return self.__recip_fvf_ref * self.__compute_polynomial(x)

    def __recip_fvf_visc(self, p_w: float) -> float:
        """Computes the reciproke of the product of formation volume factor
        and viscosity for the given water pressure.

        Args:
            p_w: Water pressure

        Returns:
            Reciproke of the product of formation volume factor and viscosity

        """

        # See Eclipse Reference Manual, p. 1762f.
        # mu_w(P) * B_w(P) = B_w(P_ref) * mu_w(P_ref) / (1 + y + (y^2 / 2))
        # y = (C - C_v) * (P - P_ref)
        # NOTE: Don't forget that all values in INIT files are stored as reciprokes!
        # 1 / (mu_w(P) * B_w(P)) = 1 / (B_w(P_ref) * mu_w(P_ref)) * (1 + y + (y^2 / 2))

        y = self.__diff_cw_cv_ref * (p_w - self.__pw_ref)

        return self.__recip_fvf_visc_ref * self.__compute_polynomial(y)

    @staticmethod
    def __evaluate(
        pressures: List[float], calculate: Callable[[Any], Any]
    ) -> List[float]:
        """Calls the calculate method with each of the values
        in the pressures list and returns the results.

        Args:
            pressures: List of pressure values
            calculate: Method to be called with each of the pressure values

        Returns:
            List of result values

        """
        quantities: List[float] = []

        for pressure in pressures:
            quantities.append(calculate(pressure))

        return quantities

    @staticmethod
    def __compute_polynomial(x: float) -> float:
        """Internal helper function.

        Computes the polynomial denominator of Eq. 3.155 and
        Eq. 3.156, Eclipse Reference Manual (p. 1762).

        Args:
            x: Expression to use (see manual)

        Returns:
            Polynomial denominator of equations

        """
        return 1.0 + x * (1.0 + x / 2.0)

    def formation_volume_factor(
        self, ratio: List[float], pressure: List[float]
    ) -> List[float]:
        """Computes a list of formation volume factor values
        for the given pressure values.

        Args:
            ratio: Dummy argument, only to conform to interface of base class
            pressure: List of pressure values

        Returns:
            A list of all formation volume factor values for the given pressure values.

        """
        return self.__evaluate(pressure, lambda p: 1.0 / self.__recip_fvf(p))

    def viscosity(self, ratio: List[float], pressure: List[float]) -> List[float]:
        """Computes a list of viscosity values for the given pressure values.

        Args:
            ratio: Dummy argument, only to conform to interface of base class
            pressure: List of pressure values

        Returns:
            A list of viscosity values corresponding
            to the given list of pressure values.

        """
        return self.__evaluate(
            pressure, lambda p: self.__recip_fvf(p) / self.__recip_fvf_visc(p)
        )

    def get_keys(self) -> List[float]:
        """Returns a list of all primary keys.

        Since this is water, there is no dependency on any ratio.
        Hence, this method returns a list holding a single float of value 0.0.

        """
        return [
            0.0,
        ]

    def get_independents(self) -> List[float]:
        """Returns a list of all independent pressure values (Pw).

        Since this is water, this does return with only one single pressure value,
        the reference pressure.
        """
        return [
            self.__pw_ref,
        ]


class Water(FluidImplementation):
    """Class for storing PVT tables for water

    Holds a list of regions (one per PVT table).

    Attributes:
        surface_mass_densities: List of surface mass densities
        keep_unit_system: True if the original unit system was kept
        original_unit_system: An ErtEclUnitEnum representing the original unit system
    """

    def __init__(
        self,
        raw: EclPropertyTableRawData,
        unit_system: int,
        surface_mass_densities: List[float],
        keep_unit_system: bool = False,
    ):
        """Initializes a Water object.

        Args:
            raw: Eclipse raw data object
            unit_system: The original unit system
            surface_mass_densities: List of surface mass densities
            keep_unit_system:
                True if the original unit system shall be kept,
                False if units shall be converted to SI units.

        """
        super().__init__(keep_unit_system)
        self.surface_mass_densities = surface_mass_densities
        self.original_unit_system = ErtEclUnitEnum(unit_system)
        self.create_water(raw, unit_system, surface_mass_densities)

    def formation_volume_factor_unit(self, latex: bool = False) -> str:
        """Creates and returns a string containing the unit symbol of the formation volume factor.

        Args:
            latex: True if the unit symbol shall be returned as LaTeX, False if not.

        Returns:
            A string containing the unit symbol of the formation volume factor.

        """
        unit_system = EclUnits.create_unit_system(
            self.original_unit_system
            if self.keep_unit_system
            else ErtEclUnitEnum.ECL_SI_UNITS
        )

        if latex:
            return (
                fr"${{r{unit_system.reservoir_volume().symbol}}}"
                fr"/{{s{unit_system.surface_volume_liquid().symbol}}}$"
            )
        return (
            f"r{unit_system.reservoir_volume().symbol}"
            f"/s{unit_system.surface_volume_liquid().symbol}"
        )

    def viscosity_unit(self, latex: bool = False) -> str:
        """Creates and returns a string containing the unit symbol of the viscosity.

        Args:
            latex: True if the unit symbol shall be returned as LaTeX, False if not.

        Returns:
            A string containing the unit symbol of the viscosity.

        """
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
        """Creates a ConvertUnits object for unit conversions for water.

        Args:
            unit_system:
                Either an integer or an enum
                describing the unit system the units are stored in

        Returns:
            ConvertUnits object for unit conversions.

        """
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
        self,
        raw: EclPropertyTableRawData,
        unit_system: int,
        surface_mass_densities: List[float],
    ) -> None:
        """Creates interpolants for water from the given raw Eclipse data and uses
        a water unit converter based on the given unit system.

        Args:
            raw: Eclipse raw data object
            unit_system: Integer representation of the unit system used in Eclipse data

        """
        # Holding raw.num_tables values
        ret: List[PvxOBase] = []

        cvrt = self.water_unit_converter(unit_system)

        ret = self.make_interpolants_from_raw_data(
            raw,
            lambda table_index, raw: WaterImpl(table_index, raw, cvrt),
        )

        assert len(surface_mass_densities) == len(ret)

        self._regions = ret

    @staticmethod
    def from_ecl_init_file(
        ecl_init_file: EclFile, keep_unit_system: bool = False
    ) -> Optional["Water"]:
        """Reads the given Eclipse file and creates a Water object from its data.

        Args:
            ecl_init_file: Eclipse INIT file
            keep_unit_system:
                Set to True if the unit system used in the Eclipse file
                shall be kept, False if SI shall be used.

        Returns:
            A Water object or None if the data in the Eclipse file was invalid

        """
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

        surface_mass_densities = surface_mass_density(ecl_init_file, EclPhaseIndex.Aqua)

        return Water(
            raw,
            intehead[InitFileDefinitions.INTEHEAD_UNIT_INDEX],
            surface_mass_densities,
            keep_unit_system,
        )
