from astropy import time
from astropy import units as u

from hermes.constellations.SpaceX_00087_constellation import SpaceX_00087
from hermes.objects import Satellite, Earth, Constellation, SatSet
from hermes.simulation import Scenario

from mayavi import mlab

## Main script

# Default scenario values
fig = None

J2015 = time.Time('J2015', scale='tt')

start = time.Time('2019-09-01 10:00:00.000', scale='tt')
stop = time.Time('2019-09-08 10:00:00.000', scale='tt')
step = 1 * u.s

# Default epoch is J2000
sat1 = Satellite.circular(Earth.poli_body, 500 * u.km, inc=90 * u.deg, raan=135 * u.deg, arglat=0 * u.deg)
sat1.set_color('#00ffff')

# Override variables when animating.
animate = True
record = True
if animate:
    fig = mlab.figure(size=(1200, 1200), bgcolor=(1.0, 1.0, 1.0))
    step = 10 * u.s
    stop = start + 0.4*sat1.period
    sat1 = Satellite.circular(Earth.poli_body, 500 * u.km, inc=90 * u.deg, raan=0 * u.deg, arglat=0 * u.deg)
    sat1.set_color('#00ffff')

scenario = Scenario(Earth, start, stop, step, figure=fig)

constellation = SpaceX_00087

# Add objects
#scenario.add_satellite(sat1)
scenario.add_satellite(constellation)

# Add analysis
#an = AccessAnalysis(scenario, sat1, constellation)
# an.csv_name = "C:/Users/jjvanthof/git/mmWaveISL/MATLAB/SpaceX_%d_7day.csv" % sat1.raan.to(u.deg).value
#scenario.add_analysis(an)

scenario.initialize()

if fig is None:
    scenario.step()  # do one step to let numba compile
    for step in scenario.simulate():
        pass

    print("Saving data...")
    scenario.stop()

    input("Press any key to quit...")

    import sys

    sys.exit()

# animation
if not (fig is None):
    scenario.draw_scenario()

    point = time.Time('2019-09-01 10:14:00.000', scale='tt')
    scenario.step_to(point, True)

    mlab.draw()
    mlab.savefig('spacex.png')

    # scenario.draw_frame()
    scenario.step()  # do one step to let numba compile
    fig.scene.movie_maker.record = record
    scenario.animate()
    mlab.show()
