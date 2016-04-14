
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
			  'relax': 10.0, 
			  'cutoff': 1.0,
			  'ensemble': 'nvt', 
			  'nSS': 2,  # number of components / subsystems
			  'idSS': [1,2],
			  'box': (0,40,0,80,0,40), # simulation box size
			  'Natoms': [10000, 5000],
			  'print': ('time', 'atoms', 'ke'), # print the time, atom number, and kinetic energy
			  'mass': [12.00, 1.01],
			  'totalSteps': 1000,
			  'rxnSteps': 100,
			  'prob': 0.5, # probability of reaction taking place ~ purely heuristic for now
			  'freq': 100, # frequency of saving/printing output
			  'thermalize': 1000 # number of steps to thermalize the system before commencing production run
			  }

	# Create an instance of the Reaction class
	Rxn = reaction(**params)

	# Define the domain, create atoms, and initialize masses, velocities, etc.
	Rxn.initialize()

	# Specify the force field based on the LJ cut-off
	Rxn.setupPhysics(ftype='lj/cut')

	# Print output specified in 'print' every 'freq' steps
	Rxn.printSetup(freq=params['freq'])

	# Setup integration method
	Rxn.setupIntegrate(name='integrator')
	
	# An equilibration / thermalization run
	Rxn.integrate(steps=params['thermalize'])

	# Monitor temperature as a function of time
	Rxn.monitor(name='globTemp', group='all', var='temp')

	# Write 'all' coordinates to 'traj.xyz' file every 'freq' steps
	Rxn.dumpSetup(sel='all', freq=params['freq'], traj='traj.xyz')

	coords = np.zeros((Rxn.lmp.get_natoms(),3))

	# Run an ensemble of short MD runs
	for _ in range(params['totalSteps'] / params['rxnSteps']):
		Rxn.integrate(steps = params['rxnSteps'])
                coords = Rxn.extractCoords(coords) # computes the average coords
		Rxn.computeRxn(params['prob'], params['cutoff'])

	# Plot temperature vs time, then save the figure as a pdf
	plt.rc('text', usetex=True)
	plt.plot(Rxn.vars)
	plt.xlabel(r"Time (fs) $\times$ {}".format(params['rxnSteps']))
	plt.ylabel("Temp (K)")
	plt.savefig("Temp.pdf")

