
import matplotlib.pylab as plt
from lammps import lammps, reaction

if __name__ == '__main__':

	# Create a dictionary of simulation parameters
	params = {
			  'units': 'real',
			  'dim': 3,
			  'boundary': ('p','p','p'),
			  'dt': 1.0, 
			  'temp': 300.0, 
			  'relax': 75.0, 
			  'cutoff': 1.0,
			  'ensemble': 'nvt', 
			  'nSS': 2, 
			  'idSS': [1,2],
			  'box': (0,40,0,80,0,40),
			  'Natoms': [10000, 5000],
			  'print': ('time', 'atoms', 'ke'),
			  'mass': [12.00, 1.01],
			  'runs': 1,
			  'rxnSteps': 1000,
			  'prob': 0.5,
			  'freq': 100,
			  'thermalize': 1000
			  }

	# Create an instance of the Reaction class
	Rxn = reaction(**params)

	# Define the domain, create atoms, and initialize masses, velocities, etc.
	Rxn.initialize()

	# Specify the force field based on the LJ cut-off
	Rxn.setupPhysics(ftype='lj/cut')

	# Print output specifies in 'print' ever 100 steps
	Rxn.printSetup(freq=params['freq'])

	# Setup integration method
	Rxn.setupIntegrate(name='integrator')
	
	# Monitor temperature as a function of time
	Rxn.monitor(name='globTemp', group='all', var='temp')

	# An equilibration / thermalization run
	Rxn.integrate(steps=params['thermalize'])

	# Extract and plot temperature as a function of time
	temp = Rxn.extract('globTemp', 0, 1, params['thermalize'])
	plt.plot(temp)
	plt.show()

	# Write 'all' coordinates to 'traj.xyz' file every 'freq' steps
	Rxn.dumpSetup(sel='all', freq=params['freq'], traj='traj.xyz')

	# Run an ensemble of short MD runs
	for _ in range(params['runs']):
		Rxn.integrate(steps = params['rxnSteps'])