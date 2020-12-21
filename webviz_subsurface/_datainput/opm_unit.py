########################################
#
#  Copyright (C) 2020-     Equinor ASA
#
########################################

from enum import IntEnum
from typing import List, Callable


class ErtEclUnitEnum(IntEnum):
    ECL_METRIC_UNITS = 1
    ECL_FIELD_UNITS = 2
    ECL_LAB_UNITS = 3
    ECL_PVT_M_UNITS = 4


class Prefix:
    micro = 1.0e-6
    milli = 1.0e-3
    centi = 1.0e-2
    deci = 1.0e-1
    kilo = 1.0e3
    mega = 1.0e6
    giga = 1.0e9


class Unit:

    # Common powers
    @staticmethod
    def square(value: float) -> float:
        return value * value

    @staticmethod
    def cubic(value: float) -> float:
        return value * value * value

    #############################
    # Basic units and conversions
    #############################

    # Length
    meter = 1.0
    inch = 2.54 * Prefix.centi * meter
    feet = 12 * inch

    # Time
    second = 1.0
    minute = 60.0 * second
    hour = 60.0 * minute
    day = 24.0 * hour
    year = 365.0 * day

    # Volume
    gallon = 231.0 * cubic.__func__(inch)  # type: ignore[attr-defined]
    stb = 42.0 * gallon
    liter = 1.0 * cubic.__func__(Prefix.deci * meter)  # type: ignore[attr-defined]

    # Mass
    kilogram = 1.0
    gram = 1.0e-3 * kilogram
    pound = 0.45359237 * kilogram

    # Energy
    joule = 1.0
    btu = 1054.3503 * joule

    # Standardised constant
    gravity = 9.80665 * meter / square.__func__(second)  # type: ignore[attr-defined]

    ###############################
    # Derived units and conversions
    ###############################

    # Force
    Newton = kilogram * meter / square.__func__(second)  # type: ignore[attr-defined]
    dyne = 1.0e-5 * Newton
    lbf = pound * gravity

    # Pressure
    Pascal = Newton / square.__func__(meter)  # type: ignore[attr-defined]
    barsa = 100000.0 * Pascal
    atm = 101325.0 * Pascal
    psia = lbf / square.__func__(inch)  # type: ignore[attr-defined]

    # Temperature
    deg_celsius = 1.0
    deg_celsius_offset = 273.15

    deg_fahrenheit = 5.0 / 9.0
    deg_fahrenheit_offset = 255.37

    # Viscosity
    Pas = Pascal * second
    Poise = Prefix.deci * Pas

    # Permeability
    p_grad = atm / (Prefix.centi * meter)
    area = square.__func__(Prefix.centi * meter)  # type: ignore[attr-defined]
    flux = cubic.__func__(Prefix.centi * meter) / second  # type: ignore[attr-defined]
    velocity = flux / area
    visc = Prefix.centi * Poise
    darcy = (velocity * visc) / p_grad

    class Convert:
        @staticmethod
        def from_(value: float, unit: float) -> float:
            return value * unit

        @staticmethod
        def to_(value: float, unit: float) -> float:
            return value / unit


class UnitSystems:
    class Metric:
        Pressure = Unit.barsa
        Temperature = Unit.deg_celsius
        TemperatureOffset = Unit.deg_celsius_offset
        AbsoluteTemperature = Unit.deg_celsius
        Length = Unit.meter
        Time = Unit.day
        Mass = Unit.kilogram
        Permeability = Prefix.milli * Unit.darcy
        Transmissibility = (
            Prefix.centi * Unit.Poise * Unit.cubic(Unit.meter) / (Unit.day * Unit.barsa)
        )
        LiquidSurfaceVolume = Unit.cubic(Unit.meter)
        GasSurfaceVolume = Unit.cubic(Unit.meter)
        ReservoirVolume = Unit.cubic(Unit.meter)
        GasDissolutionFactor = GasSurfaceVolume / LiquidSurfaceVolume
        OilDissolutionFactor = LiquidSurfaceVolume / GasSurfaceVolume
        Density = Unit.kilogram / Unit.cubic(Unit.meter)
        PolymerDensity = Unit.kilogram / Unit.cubic(Unit.meter)
        Salinity = Unit.kilogram / Unit.cubic(Unit.meter)
        Viscosity = Prefix.centi * Unit.Poise
        Timestep = Unit.day
        SurfaceTension = Unit.dyne / (Prefix.centi * Unit.meter)
        Energy = Prefix.kilo * Unit.joule

    class Field:
        Pressure = Unit.psia
        Temperature = Unit.deg_fahrenheit
        TemperatureOffset = Unit.deg_fahrenheit_offset
        AbsoluteTemperature = Unit.deg_fahrenheit
        Length = Unit.feet
        Time = Unit.day
        Mass = Unit.pound
        Permeability = Prefix.milli * Unit.darcy
        Transmissibility = Prefix.centi * Unit.Poise * Unit.stb / (Unit.day * Unit.psia)
        LiquidSurfaceVolume = Unit.stb
        GasSurfaceVolume = Unit.cubic(Unit.feet)
        ReservoirVolume = Unit.stb
        GasDissolutionFactor = GasSurfaceVolume / LiquidSurfaceVolume
        OilDissolutionFactor = LiquidSurfaceVolume / GasSurfaceVolume
        Density = Unit.pound / Unit.cubic(Unit.feet)
        PolymerDensity = Unit.pound / Unit.stb
        Salinity = Unit.pound / Unit.stb
        Viscosity = Prefix.centi * Unit.Poise
        Timestep = Unit.day
        SurfaceTension = Unit.dyne / (Prefix.centi * Unit.meter)
        Energy = Unit.btu

    class Lab:
        Pressure = Unit.atm
        Temperature = Unit.deg_celsius
        TemperatureOffset = Unit.deg_celsius_offset
        AbsoluteTemperature = Unit.deg_celsius
        Length = Prefix.centi * Unit.meter
        Time = Unit.hour
        Mass = Unit.gram
        Permeability = Prefix.milli * Unit.darcy
        Transmissibility = (
            Prefix.centi
            * Unit.Poise
            * Unit.cubic(Prefix.centi * Unit.meter)
            / (Unit.hour * Unit.atm)
        )
        LiquidSurfaceVolume = Unit.cubic(Prefix.centi * Unit.meter)
        GasSurfaceVolume = Unit.cubic(Prefix.centi * Unit.meter)
        ReservoirVolume = Unit.cubic(Prefix.centi * Unit.meter)
        GasDissolutionFactor = GasSurfaceVolume / LiquidSurfaceVolume
        OilDissolutionFactor = LiquidSurfaceVolume / GasSurfaceVolume
        Density = Unit.gram / Unit.cubic(Prefix.centi * Unit.meter)
        PolymerDensity = Unit.gram / Unit.cubic(Prefix.centi * Unit.meter)
        Salinity = Unit.gram / Unit.cubic(Prefix.centi * Unit.meter)
        Viscosity = Prefix.centi * Unit.Poise
        Timestep = Unit.hour
        SurfaceTension = Unit.dyne / (Prefix.centi * Unit.meter)
        Energy = Unit.joule

    class PVTM:
        Pressure = Unit.atm
        Temperature = Unit.deg_celsius
        TemperatureOffset = Unit.deg_celsius_offset
        AbsoluteTemperature = Unit.deg_celsius
        Length = Unit.meter
        Time = Unit.day
        Mass = Unit.kilogram
        Permeability = Prefix.milli * Unit.darcy
        Transmissibility = (
            Prefix.centi * Unit.Poise * Unit.cubic(Unit.meter) / (Unit.day * Unit.atm)
        )
        LiquidSurfaceVolume = Unit.cubic(Unit.meter)
        GasSurfaceVolume = Unit.cubic(Unit.meter)
        ReservoirVolume = Unit.cubic(Unit.meter)
        GasDissolutionFactor = GasSurfaceVolume / LiquidSurfaceVolume
        OilDissolutionFactor = LiquidSurfaceVolume / GasSurfaceVolume
        Density = Unit.kilogram / Unit.cubic(Unit.meter)
        PolymerDensity = Unit.kilogram / Unit.cubic(Unit.meter)
        Salinity = Unit.kilogram / Unit.cubic(Unit.meter)
        Viscosity = Prefix.centi * Unit.Poise
        Timestep = Unit.day
        SurfaceTension = Unit.dyne / (Prefix.centi * Unit.meter)
        Energy = Prefix.kilo * Unit.joule


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
            raise ValueError(f"Unsupported Unit Convention: {unit_system}")


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
    # pylint: disable=invalid-name
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
