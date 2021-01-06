########################################
#
#  Copyright (C) 2020-     Equinor ASA
#
########################################

from enum import IntEnum
from typing import List, Callable, Union, Dict


class ErtEclUnitEnum(IntEnum):
    """ An enum for the different unit systems """

    ECL_SI_UNITS = 0
    ECL_METRIC_UNITS = 1
    ECL_FIELD_UNITS = 2
    ECL_LAB_UNITS = 3
    ECL_PVT_M_UNITS = 4


def unit_system_name(unit_system: ErtEclUnitEnum) -> str:
    if unit_system == ErtEclUnitEnum.ECL_SI_UNITS:
        return "SI"
    if unit_system == ErtEclUnitEnum.ECL_METRIC_UNITS:
        return "METRIC"
    if unit_system == ErtEclUnitEnum.ECL_FIELD_UNITS:
        return "FIELD"
    if unit_system == ErtEclUnitEnum.ECL_LAB_UNITS:
        return "LAB"
    if unit_system == ErtEclUnitEnum.ECL_PVT_M_UNITS:
        return "PVT"
    return "UNKNOWN"


class UnitBase:
    pass


# pylint: disable=too-few-public-methods
class Prefix:
    class Base:
        def __init__(self, value: float, symbol: str) -> None:
            self.value = value
            self.symbol = symbol

        def __mul__(self, other):  # type: ignore[no-untyped-def]
            if issubclass(other.__class__, UnitBase):
                return other.__class__(
                    self.value * other.value, f"{self.symbol}{other.symbol}"
                )
            elif isinstance(other, float):
                return self.value * other
            else:
                raise TypeError("Can only be multiplied with a unit or a float.")

        def __add__(self, other):  # type: ignore[no-untyped-def]
            raise NotImplementedError(
                "Prefixes can only be multiplied with either Units or floats."
            )

        def __sub__(self, other):  # type: ignore[no-untyped-def]
            raise NotImplementedError(
                "Prefixes can only be multiplied with either Units or floats."
            )

        def __truediv__(self, other):  # type: ignore[no-untyped-def]
            raise NotImplementedError(
                "Prefixes can only be multiplied with either Units or floats."
            )

    micro = Base(1.0e-6, "\u00B5")
    milli = Base(1.0e-3, "m")
    centi = Base(1.0e-2, "c")
    deci = Base(1.0e-1, "d")
    kilo = Base(1.0e3, "k")
    mega = Base(1.0e6, "M")
    giga = Base(1.0e9, "G")


class Unit:

    # Base class for units
    class Base(UnitBase):
        def __init__(self, value: Union[float, "Base"], symbol: str) -> None:  # type: ignore[name-defined]
            if isinstance(value, float):
                self.value = value
                self.symbol = symbol
            else:
                self.value = value.value
                self.symbol = symbol

            self.tidy_symbol()

        def tidy_symbol(self) -> None:
            self.symbol = "".join(self.symbol.split())
            numerator: Dict[str, int] = {}
            denominator: Dict[str, int] = {}

            in_parantheses = False
            in_denominator = False
            power = False
            current_power = ""
            current_symbol = ""
            i = 0

            for char in self.symbol:
                if char in ["*", "/", "(", ")"]:
                    if current_symbol != "":
                        if not in_denominator:
                            if current_symbol not in numerator:
                                numerator[current_symbol] = 0
                            numerator[current_symbol] += int(
                                current_power if current_power != "" else 1
                            )
                        else:
                            if current_symbol not in denominator:
                                denominator[current_symbol] = 0
                            denominator[current_symbol] += int(
                                current_power if current_power != "" else 1
                            )

                if char in ["*", "/", "(", ")"]:
                    if char == "*":
                        if not in_parantheses:
                            in_denominator = False
                    elif char == "/":
                        in_denominator = True
                    elif char == "(":
                        in_parantheses = True
                    elif char == ")":
                        in_parantheses = False

                    current_symbol = ""
                    current_power = ""
                    power = False
                elif char == "^":
                    power = True
                else:
                    if power:
                        current_power += char
                    else:
                        current_symbol += char

                if i == len(self.symbol) - 1:
                    if current_symbol != "":
                        if not in_denominator:
                            if current_symbol not in numerator:
                                numerator[current_symbol] = 0
                            numerator[current_symbol] += int(
                                current_power if current_power != "" else 1
                            )
                        else:
                            if current_symbol not in denominator:
                                denominator[current_symbol] = 0
                            denominator[current_symbol] += int(
                                current_power if current_power != "" else 1
                            )
                i += 1

            self.symbol = ""

            for num, num_power in numerator.items():
                for denom, denom_power in denominator.items():
                    if num == denom:
                        denominator[num] = max(0, denom_power - num_power)
                        numerator[num] = max(0, num_power - denom_power)
                        break

            for num, num_power in numerator.items():
                if num_power > 0:
                    if self.symbol == "":
                        self.symbol += num
                    else:
                        self.symbol += f"*{num}"
                    if num_power > 1:
                        self.symbol += f"^{num_power}"

            if len(denominator) > 0:
                self.symbol += "/("

                for denom, denom_power in denominator.items():
                    if denom_power > 0:
                        if self.symbol[-1] == "(":
                            self.symbol += denom
                        else:
                            self.symbol += f"*{denom}"
                        if denom_power > 1:
                            self.symbol += f"^{denom_power}"

                self.symbol += ")"

        def __mul__(self, other):  # type: ignore[no-untyped-def]
            if type(other) is type(self):
                return self.__class__(
                    self.value * other.value, f"{self.symbol}*{other.symbol}"
                )
            elif isinstance(other, Prefix):
                return self.__class__(
                    self.value * other.value, f"{self.symbol}*{other.symbol}"
                )
            elif isinstance(other, float):
                return self.__class__(self.value * other, self.symbol)
            elif isinstance(other, int):
                return self.__class__(self.value * float(other), self.symbol)
            else:
                raise TypeError(
                    "You can only multiply this unit with another unit, a float or an integer."
                )

        def __truediv__(self, other):  # type: ignore[no-untyped-def]
            if type(other) is type(self):
                return self.__class__(
                    self.value / other.value, f"{self.symbol}/({other.symbol})"
                )
            elif isinstance(other, float):
                return self.__class__(self.value / other, self.symbol)
            elif isinstance(other, int):
                return self.__class__(self.value / float(other), self.symbol)
            else:
                raise TypeError(
                    "You can only divide this unit by another unit, a float or an integer."
                )

    # Common powers
    @staticmethod
    def square(value: Base) -> Base:
        return value * value

    @staticmethod
    def cubic(value: Base) -> Base:
        return value * value * value

    #############################
    # Basic units and conversions
    #############################

    meter = Base(1.0, "m")

    inch = Base(Prefix.centi * meter * 2.54, "inch")

    feet = Base(inch * 12.0, "ft")

    # Time
    second = Base(1.0, "s")
    minute = Base(second * 60.0, "m")
    hour = Base(minute * 60.0, "h")
    day = Base(hour.value * 24.0, "d")
    year = Base(day.value * 365.0, "a")

    # Volume
    gallon = Base(cubic.__func__(inch) * 231.0, "gal")  # type: ignore[attr-defined]
    stb = Base(gallon * 42.0, "stb")
    liter = Base(cubic.__func__(Prefix.deci * meter.value), "l")  # type: ignore[attr-defined]

    # Mass
    kilogram = Base(1.0, "kg")
    gram = Base(kilogram * 1.0e-3, "g")
    pound = Base(kilogram * 0.45359237, "lb")

    # Energy
    joule = Base(1.0, "J")
    btu = Base(joule * 1054.3503, "Btu")

    # Standardised constant
    gravity = (meter * 9.80665) / square.__func__(second)  # type: ignore[attr-defined]

    ###############################
    # Derived units and conversions
    ###############################

    # Force
    Newton = Base(kilogram * meter / square.__func__(second), "N")  # type: ignore[attr-defined]
    dyne = Base(Newton * 1.0e-5, "dyn")
    lbf = Base(pound * gravity, "lbf")

    # Pressure
    Pascal = Base(Newton / square.__func__(meter), "Pa")  # type: ignore[attr-defined]
    barsa = Base(Pascal * 100000.0, "bar")
    atm = Base(Pascal * 101325.0, "atm")
    psia = Base(lbf / square.__func__(inch), "psi")  # type: ignore[attr-defined]

    # Temperature
    deg_kelvin = Base(1.0, "K")

    deg_celsius = Base(1.0, "\u2103")
    deg_celsius_offset = 273.15

    deg_fahrenheit = Base(5.0 / 9.0, "\u2109")
    deg_fahrenheit_offset = 255.37

    # Viscosity
    Pas = Pascal * second
    Poise = Base(Prefix.deci * Pas, "P")

    # Permeability
    p_grad = atm / (Prefix.centi * meter)
    area = square.__func__(Prefix.centi * meter)  # type: ignore[attr-defined]
    # pylint: disable=line-too-long
    flux = cubic.__func__(Prefix.centi * meter) / second  # type: ignore[attr-defined]
    # pylint: enable=line-too-long
    velocity = flux / area
    visc = Base(Prefix.centi * Poise, "cP")
    darcy = Base((velocity * visc) / p_grad, "D")

    class Convert:
        @staticmethod
        def from_(value: float, unit: float) -> float:
            return value * unit

        @staticmethod
        def to_(value: float, unit: float) -> float:
            return value / unit


class UnitSystems:
    class SI:
        Pressure = Unit.Pascal
        Temperature = Unit.deg_kelvin
        TemperatureOffset = 0.0
        AbsoluteTemperature = Unit.deg_kelvin
        Length = Unit.meter
        Time = Unit.second
        Mass = Unit.kilogram
        Permeability = Unit.square(Unit.meter)
        Transmissibility = Unit.cubic(Unit.meter)
        LiquidSurfaceVolume = Unit.cubic(Unit.meter)
        GasSurfaceVolume = Unit.cubic(Unit.meter)
        ReservoirVolume = Unit.cubic(Unit.meter)
        GasDissolutionFactor = GasSurfaceVolume / LiquidSurfaceVolume
        OilDissolutionFactor = LiquidSurfaceVolume / GasSurfaceVolume
        Density = Unit.kilogram / Unit.cubic(Unit.meter)
        PolymerDensity = Unit.kilogram / Unit.cubic(Unit.meter)
        Salinity = Unit.kilogram / Unit.cubic(Unit.meter)
        Viscosity = Unit.Pascal * Unit.second
        Timestep = Unit.second
        SurfaceTension = Unit.Newton / Unit.meter
        Energy = Unit.joule

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


class UnitSystemUnits:
    class Metric:
        Pressure = Unit


class EclUnits:
    class UnitSystem:
        @staticmethod
        def density() -> Unit.Base:
            raise NotImplementedError("The base class cannot be called directly.")

        @staticmethod
        def depth() -> Unit.Base:
            raise NotImplementedError("The base class cannot be called directly.")

        @staticmethod
        def pressure() -> Unit.Base:
            raise NotImplementedError("The base class cannot be called directly.")

        @staticmethod
        def reservoir_rate() -> Unit.Base:
            raise NotImplementedError("The base class cannot be called directly.")

        @staticmethod
        def reservoir_volume() -> Unit.Base:
            raise NotImplementedError("The base class cannot be called directly.")

        @staticmethod
        def surface_volume_gas() -> Unit.Base:
            raise NotImplementedError("The base class cannot be called directly.")

        @staticmethod
        def surface_volume_liquid() -> Unit.Base:
            raise NotImplementedError("The base class cannot be called directly.")

        @staticmethod
        def time() -> Unit.Base:
            raise NotImplementedError("The base class cannot be called directly.")

        @staticmethod
        def transmissibility() -> Unit.Base:
            raise NotImplementedError("The base class cannot be called directly.")

        @staticmethod
        def viscosity() -> Unit.Base:
            raise NotImplementedError("The base class cannot be called directly.")

        def dissolved_gas_oil_ratio(self) -> Unit.Base:
            return self.surface_volume_gas() / self.surface_volume_liquid()

        def vaporised_oil_gas_ratio(self) -> Unit.Base:
            return self.surface_volume_liquid() / self.surface_volume_gas()

    class EclSIUnitSystem(UnitSystem):
        @staticmethod
        def density() -> Unit.Base:
            return UnitSystems.SI.Density

        @staticmethod
        def depth() -> Unit.Base:
            return UnitSystems.SI.Length

        @staticmethod
        def pressure() -> Unit.Base:
            return UnitSystems.SI.Pressure

        @staticmethod
        def reservoir_rate() -> Unit.Base:
            return UnitSystems.SI.ReservoirVolume / UnitSystems.SI.Time

        @staticmethod
        def reservoir_volume() -> Unit.Base:
            return UnitSystems.SI.ReservoirVolume

        @staticmethod
        def surface_volume_gas() -> Unit.Base:
            return UnitSystems.SI.GasSurfaceVolume

        @staticmethod
        def surface_volume_liquid() -> Unit.Base:
            return UnitSystems.SI.LiquidSurfaceVolume

        @staticmethod
        def time() -> Unit.Base:
            return UnitSystems.SI.Time

        @staticmethod
        def transmissibility() -> Unit.Base:
            return UnitSystems.SI.Transmissibility

        @staticmethod
        def viscosity() -> Unit.Base:
            return UnitSystems.SI.Viscosity

    class EclMetricUnitSystem(UnitSystem):
        @staticmethod
        def density() -> Unit.Base:
            return UnitSystems.Metric.Density

        @staticmethod
        def depth() -> Unit.Base:
            return UnitSystems.Metric.Length

        @staticmethod
        def pressure() -> Unit.Base:
            return UnitSystems.Metric.Pressure

        @staticmethod
        def reservoir_rate() -> Unit.Base:
            return UnitSystems.Metric.ReservoirVolume / UnitSystems.Metric.Time

        @staticmethod
        def reservoir_volume() -> Unit.Base:
            return UnitSystems.Metric.ReservoirVolume

        @staticmethod
        def surface_volume_gas() -> Unit.Base:
            return UnitSystems.Metric.GasSurfaceVolume

        @staticmethod
        def surface_volume_liquid() -> Unit.Base:
            return UnitSystems.Metric.LiquidSurfaceVolume

        @staticmethod
        def time() -> Unit.Base:
            return UnitSystems.Metric.Time

        @staticmethod
        def transmissibility() -> Unit.Base:
            return UnitSystems.Metric.Transmissibility

        @staticmethod
        def viscosity() -> Unit.Base:
            return UnitSystems.Metric.Viscosity

    class EclFieldUnitSystem(UnitSystem):
        @staticmethod
        def density() -> Unit.Base:
            return UnitSystems.Field.Density

        @staticmethod
        def depth() -> Unit.Base:
            return UnitSystems.Field.Length

        @staticmethod
        def pressure() -> Unit.Base:
            return UnitSystems.Field.Pressure

        @staticmethod
        def reservoir_rate() -> Unit.Base:
            return UnitSystems.Field.ReservoirVolume / UnitSystems.Field.Time

        @staticmethod
        def reservoir_volume() -> Unit.Base:
            return UnitSystems.Field.ReservoirVolume

        @staticmethod
        def surface_volume_gas() -> Unit.Base:
            return UnitSystems.Field.GasSurfaceVolume

        @staticmethod
        def surface_volume_liquid() -> Unit.Base:
            return UnitSystems.Field.LiquidSurfaceVolume

        @staticmethod
        def time() -> Unit.Base:
            return UnitSystems.Field.Time

        @staticmethod
        def transmissibility() -> Unit.Base:
            return UnitSystems.Field.Transmissibility

        @staticmethod
        def viscosity() -> Unit.Base:
            return UnitSystems.Field.Viscosity

    class EclLabUnitSystem(UnitSystem):
        @staticmethod
        def density() -> Unit.Base:
            return UnitSystems.Lab.Density

        @staticmethod
        def depth() -> Unit.Base:
            return UnitSystems.Lab.Length

        @staticmethod
        def pressure() -> Unit.Base:
            return UnitSystems.Lab.Pressure

        @staticmethod
        def reservoir_rate() -> Unit.Base:
            return UnitSystems.Lab.ReservoirVolume / UnitSystems.Lab.Time

        @staticmethod
        def reservoir_volume() -> Unit.Base:
            return UnitSystems.Lab.ReservoirVolume

        @staticmethod
        def surface_volume_gas() -> Unit.Base:
            return UnitSystems.Lab.GasSurfaceVolume

        @staticmethod
        def surface_volume_liquid() -> Unit.Base:
            return UnitSystems.Lab.LiquidSurfaceVolume

        @staticmethod
        def time() -> Unit.Base:
            return UnitSystems.Lab.Time

        @staticmethod
        def transmissibility() -> Unit.Base:
            return UnitSystems.Lab.Transmissibility

        @staticmethod
        def viscosity() -> Unit.Base:
            return UnitSystems.Lab.Viscosity

    class EclPvtMUnitSystem(UnitSystem):
        @staticmethod
        def density() -> Unit.Base:
            return UnitSystems.PVTM.Density

        @staticmethod
        def depth() -> Unit.Base:
            return UnitSystems.PVTM.Length

        @staticmethod
        def pressure() -> Unit.Base:
            return UnitSystems.PVTM.Pressure

        @staticmethod
        def reservoir_rate() -> Unit.Base:
            return UnitSystems.PVTM.ReservoirVolume / UnitSystems.PVTM.Time

        @staticmethod
        def reservoir_volume() -> Unit.Base:
            return UnitSystems.PVTM.ReservoirVolume

        @staticmethod
        def surface_volume_gas() -> Unit.Base:
            return UnitSystems.PVTM.GasSurfaceVolume

        @staticmethod
        def surface_volume_liquid() -> Unit.Base:
            return UnitSystems.PVTM.LiquidSurfaceVolume

        @staticmethod
        def time() -> Unit.Base:
            return UnitSystems.PVTM.Time

        @staticmethod
        def transmissibility() -> Unit.Base:
            return UnitSystems.PVTM.Transmissibility

        @staticmethod
        def viscosity() -> Unit.Base:
            return UnitSystems.PVTM.Viscosity

    @staticmethod
    def create_unit_system(unit_system: int) -> UnitSystem:
        if unit_system == ErtEclUnitEnum.ECL_SI_UNITS:
            return EclUnits.EclSIUnitSystem()
        if unit_system == ErtEclUnitEnum.ECL_METRIC_UNITS:
            return EclUnits.EclMetricUnitSystem()
        if unit_system == ErtEclUnitEnum.ECL_FIELD_UNITS:
            return EclUnits.EclFieldUnitSystem()
        if unit_system == ErtEclUnitEnum.ECL_LAB_UNITS:
            return EclUnits.EclLabUnitSystem()
        if unit_system == ErtEclUnitEnum.ECL_PVT_M_UNITS:
            return EclUnits.EclPvtMUnitSystem()
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
        """Converts quantity from its measurement units to SI units.

        Example:
        quantity = 100 bar
        [quantity](METRIC) = bar
        [quantity](SI) = Pa
        uscale = METRIC.pressure() = 100 000 Pa/bar
        returns 100 bar * 100 000 Pa/bar = 10 000 000 Pa
        """
        return lambda quantity: Unit.Convert.from_(quantity, uscale)

    @staticmethod
    def rs_scale(unit_system: EclUnits.UnitSystem) -> float:
        # Rs = [sVolume(Gas) / sVolume(Liquid)]
        return (
            unit_system.surface_volume_gas().value
            / unit_system.surface_volume_liquid().value
        )

    @staticmethod
    def rv_scale(unit_system: EclUnits.UnitSystem) -> float:
        # Rv = [sVolume(Liq) / sVolume(Gas)]
        return (
            unit_system.surface_volume_liquid().value
            / unit_system.surface_volume_gas().value
        )

    @staticmethod
    def fvf_scale(unit_system: EclUnits.UnitSystem) -> float:
        # B = [rVolume / sVolume(Liquid)]
        return (
            unit_system.reservoir_volume().value
            / unit_system.surface_volume_liquid().value
        )

    @staticmethod
    def fvf_gas_scale(unit_system: EclUnits.UnitSystem) -> float:
        # B = [rVolume / sVolume(Gas)]
        return (
            unit_system.reservoir_volume().value
            / unit_system.surface_volume_gas().value
        )

    class ToSI:
        @staticmethod
        def density(
            unit_system: EclUnits.UnitSystem,
        ) -> Callable[[float,], float]:
            return CreateUnitConverter.create_converter_to_SI(
                unit_system.density().value
            )

        @staticmethod
        def pressure(
            unit_system: EclUnits.UnitSystem,
        ) -> Callable[[float,], float]:
            return CreateUnitConverter.create_converter_to_SI(
                unit_system.pressure().value
            )

        @staticmethod
        def compressibility(
            unit_system: EclUnits.UnitSystem,
        ) -> Callable[[float,], float]:
            return CreateUnitConverter.create_converter_to_SI(
                1.0 / unit_system.pressure().value
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
            p_scale = unit_system.pressure().value

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
            visc_scale = unit_system.viscosity().value

            return CreateUnitConverter.create_converter_to_SI(
                1.0 / (b_scale * visc_scale)
            )

        @staticmethod
        def recip_fvf_visc_deriv_press(
            unit_system: EclUnits.UnitSystem,
        ) -> Callable[[float,], float]:
            # d(1/(B*mu))/dp
            b_scale = CreateUnitConverter.fvf_scale(unit_system)
            p_scale = unit_system.pressure().value
            visc_scale = unit_system.viscosity().value

            return CreateUnitConverter.create_converter_to_SI(
                1.0 / (b_scale * visc_scale * p_scale)
            )

        @staticmethod
        def recip_fvf_visc_deriv_vap_oil(
            unit_system: EclUnits.UnitSystem,
        ) -> Callable[[float,], float]:
            # d(1/(B*mu))/dRv
            b_scale = CreateUnitConverter.fvf_scale(unit_system)
            visc_scale = unit_system.viscosity().value
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
            p_scale = unit_system.pressure().value

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
            visc_scale = unit_system.viscosity().value

            return CreateUnitConverter.create_converter_to_SI(
                1.0 / (b_scale * visc_scale)
            )

        @staticmethod
        def recip_fvf_gas_visc_deriv_press(
            unit_system: EclUnits.UnitSystem,
        ) -> Callable[[float,], float]:
            # d(1/(B*mu))/dp
            b_scale = CreateUnitConverter.fvf_gas_scale(unit_system)
            p_scale = unit_system.pressure().value
            visc_scale = unit_system.viscosity().value

            return CreateUnitConverter.create_converter_to_SI(
                1.0 / (b_scale * visc_scale * p_scale)
            )

        @staticmethod
        def recip_fvf_gas_visc_deriv_vap_oil(
            unit_system: EclUnits.UnitSystem,
        ) -> Callable[[float,], float]:
            # d(1/(B*mu))/dRv
            b_scale = CreateUnitConverter.fvf_gas_scale(unit_system)
            visc_scale = unit_system.viscosity().value
            rv_scale = CreateUnitConverter.rv_scale(unit_system)

            return CreateUnitConverter.create_converter_to_SI(
                1.0 / (b_scale * visc_scale * rv_scale)
            )
