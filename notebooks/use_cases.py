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
    use_case['T_orbit_s'] = satellite.period.to(u.s).value
    use_case['T_sim_s'] = use_case['T_orbit_s'] * 3

    # Define targets
    use_case['T_bitorbit_target'] = 120 * 8 * 1e9 # Throughput per orbit [bit]
    use_case['L_sorbit_target'] = 0
    use_case['E_Jorbit_target'] = 50 * use_case['T_orbit_s'] * 0.145 # Energy consumption per orbit [J]
    use_case['P_sorbit_target'] = use_case['T_sim_s']

    # Define constraints
    perc_margin = 0.35 # percentage above and below target values
    use_case['T_bitorbit_min'] = use_case['T_bitorbit_target'] * (1 - perc_margin)
    use_case['T_bitorbit_max'] = use_case['T_bitorbit_target'] * (1 + perc_margin)

    use_case['E_Jorbit_max'] = use_case['E_Jorbit_target'] * 2
    use_case['L_sorbit_max'] = (1.5 * u.h).to(u.s).value

    requirements = Requirements()
    requirements.min_throughput = use_case['T_bitorbit_min'] * (use_case['T_sim_s'] / use_case['T_orbit_s'])
    requirements.max_throughput = use_case['T_bitorbit_max'] * (use_case['T_sim_s'] / use_case['T_orbit_s'])
    requirements.max_energy = use_case['E_Jorbit_max'] * (use_case['T_sim_s'] / use_case['T_orbit_s'])
    requirements.max_latency = use_case['L_sorbit_max']

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
    target['GT_dBK'] = 7.0

    return target


O3b_target = _O3b()
