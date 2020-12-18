from enum import IntEnum


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
    gallon = 231.0 * cubic.__func__(inch)
    stb = 42.0 * gallon
    liter = 1.0 * cubic.__func__(Prefix.deci * meter)

    # Mass
    kilogram = 1.0
    gram = 1.0e-3 * kilogram
    pound = 0.45359237 * kilogram

    # Energy
    joule = 1.0
    btu = 1054.3503 * joule

    # Standardised constant
    gravity = 9.80665 * meter / square.__func__(second)

    ###############################
    # Derived units and conversions
    ###############################

    # Force
    Newton = kilogram * meter / square.__func__(second)
    dyne = 1.0e-5 * Newton
    lbf = pound * gravity

    # Pressure
    Pascal = Newton / square.__func__(meter)
    barsa = 100000.0 * Pascal
    atm = 101325.0 * Pascal
    psia = lbf / square.__func__(inch)

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
    area = square.__func__(Prefix.centi * meter)
    flux = cubic.__func__(Prefix.centi * meter) / second
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