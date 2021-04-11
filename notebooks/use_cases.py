import hermes
from hermes.constellations.Telesat import Telesat_00053
from hermes.objects import Satellite

from astropy import units as u, time

from optimization_problems.constraints import Requirements


# Use cases
def _iotm2m():
    use_case = {}

    # Define satellite
    satellite = Satellite.circular(hermes.objects.Earth.poli_body, 600 * u.km, inc=97.8 * u.deg, raan=22.5 * u.deg,
                                   arglat=0 * u.deg)
    use_case['satellite'] = satellite

    # Define requirements
    requirements = Requirements()
    # do something
    use_case['requirements'] = requirements

    return use_case


IoTM2M_mission = _iotm2m()


def _eo():
    use_case = {}

    # Define satellite
    satellite = Satellite.circular(hermes.objects.Earth.poli_body, 500 * u.km, inc=98.0 * u.deg, raan=0.0 * u.deg,
                                   arglat=0 * u.deg)
    use_case['satellite'] = satellite

    # Define requirements
    requirements = Requirements()
    # do something
    use_case['requirements'] = requirements

    return use_case


EO_mission = _eo()


# Target definitions
def _telesat():
    target = {}

    # Define constellation
    constellation = Telesat_00053
    target['constellation'] = constellation

    # Define system parameters
    target['frequency'] = 20e9
    target['GT_dBK'] = 13.2

    return target


Telesat_target = _telesat()


def _O3b():
    target = {}

    # Define constellation
    constellation = Telesat_00053
    target['constellation'] = constellation

    # Define system parameters
    target['frequency'] = 28.4e9
    target['GT_dBK'] = 13.2

    return target


O3b_target = _O3b()
