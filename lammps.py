# ----------------------------------------------------------------------
#   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
#   http://lammps.sandia.gov, Sandia National Laboratories
#   Steve Plimpton, sjplimp@sandia.gov
#
#   Copyright (2003) Sandia Corporation.  Under the terms of Contract
#   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
#   certain rights in this software.  This software is distributed under 
#   the GNU General Public License.
#
#   See the README file in the top-level LAMMPS directory.
# -------------------------------------------------------------------------

# Reactive MD class that uses the python wrapper for LAMMPS library via ctypes

import sys,traceback,types
from ctypes import *
from os.path import dirname,abspath,join
from inspect import getsourcefile

import numpy as np
import itertools
import logging
from numpy.linalg import norm
from scipy import spatial

logging.basicConfig(filename='reaction.log', format='%(asctime)s:%(levelname)s: %(message)s', level=logging.DEBUG)

class lammps:
  
  # detect if Python is using version of mpi4py that can pass a communicator
  
  has_mpi4py_v2 = False
  try:
    from mpi4py import MPI
    from mpi4py import __version__ as mpi4py_version
    if mpi4py_version.split('.')[0] == '2':
      has_mpi4py_v2 = True
  except:
    pass

  # create instance of LAMMPS
  
  def __init__(self,name="",cmdargs=None,ptr=None,comm=None):

    # determine module location
    
    modpath = dirname(abspath(getsourcefile(lambda:0)))

    # load liblammps.so unless name is given.
    # e.g. if name = "g++", load liblammps_g++.so
    # try loading the LAMMPS shared object from the location
    # of lammps.py with an absolute path (so that LD_LIBRARY_PATH
    # does not need to be set for regular installations.
    # fall back to loading with a relative path, which typically
    # requires LD_LIBRARY_PATH to be set appropriately.

    try:
      if not name: self.lib = CDLL(join(modpath,"liblammps.so"),RTLD_GLOBAL)
      else: self.lib = CDLL(join(modpath,"liblammps_%s.so" % name),RTLD_GLOBAL)
    except:
      if not name: self.lib = CDLL("liblammps.so",RTLD_GLOBAL)
      else: self.lib = CDLL("liblammps_%s.so" % name,RTLD_GLOBAL)

    # if no ptr provided, create an instance of LAMMPS
    #   don't know how to pass an MPI communicator from PyPar
    #   but we can pass an MPI communicator from mpi4py v2.0.0 and later
    #   no_mpi call lets LAMMPS use MPI_COMM_WORLD
    #   cargs = array of C strings from args
    # if ptr, then are embedding Python in LAMMPS input script
    #   ptr is the desired instance of LAMMPS
    #   just convert it to ctypes ptr and store in self.lmp
    
    if not ptr:
      # with mpi4py v2, can pass MPI communicator to LAMMPS
      # need to adjust for type of MPI communicator object
      # allow for int (like MPICH) or void* (like OpenMPI)
      
      if lammps.has_mpi4py_v2 and comm != None:
        if lammps.MPI._sizeof(lammps.MPI.Comm) == sizeof(c_int):
          MPI_Comm = c_int
        else:
          MPI_Comm = c_void_p

        narg = 0
        cargs = 0
        if cmdargs:
          cmdargs.insert(0,"lammps.py")
          narg = len(cmdargs)
          cargs = (c_char_p*narg)(*cmdargs)
          self.lib.lammps_open.argtypes = [c_int, c_char_p*narg, \
                                           MPI_Comm, c_void_p()]
        else:
          self.lib.lammps_open.argtypes = [c_int, c_int, \
                                           MPI_Comm, c_void_p()]

        self.lib.lammps_open.restype = None
        self.opened = 1
        self.lmp = c_void_p()
        comm_ptr = lammps.MPI._addressof(comm)
        comm_val = MPI_Comm.from_address(comm_ptr)
        self.lib.lammps_open(narg,cargs,comm_val,byref(self.lmp))

      else:
        self.opened = 1
        if cmdargs:
          cmdargs.insert(0,"lammps.py")
          narg = len(cmdargs)
          cargs = (c_char_p*narg)(*cmdargs)
          self.lmp = c_void_p()
          self.lib.lammps_open_no_mpi(narg,cargs,byref(self.lmp))
        else:
          self.lmp = c_void_p()
          self.lib.lammps_open_no_mpi(0,None,byref(self.lmp))
          # could use just this if LAMMPS lib interface supported it
          # self.lmp = self.lib.lammps_open_no_mpi(0,None)
          
    else:
      self.opened = 0
      # magic to convert ptr to ctypes ptr
      pythonapi.PyCObject_AsVoidPtr.restype = c_void_p
      pythonapi.PyCObject_AsVoidPtr.argtypes = [py_object]
      self.lmp = c_void_p(pythonapi.PyCObject_AsVoidPtr(ptr))

  def __del__(self):
    if self.lmp and self.opened: self.lib.lammps_close(self.lmp)

  def close(self):
    if self.opened: self.lib.lammps_close(self.lmp)
    self.lmp = None

  def version(self):
    return self.lib.lammps_version(self.lmp)

  def file(self,file):
    self.lib.lammps_file(self.lmp,file)

  def command(self,cmd):
    self.lib.lammps_command(self.lmp,cmd)

  def extract_global(self,name,type):
    if type == 0:
      self.lib.lammps_extract_global.restype = POINTER(c_int)
    elif type == 1:
      self.lib.lammps_extract_global.restype = POINTER(c_double)
    else: return None
    ptr = self.lib.lammps_extract_global(self.lmp,name)
    return ptr[0]

  def extract_atom(self,name,type):
    if type == 0:
      self.lib.lammps_extract_atom.restype = POINTER(c_int)
    elif type == 1:
      self.lib.lammps_extract_atom.restype = POINTER(POINTER(c_int))
    elif type == 2:
      self.lib.lammps_extract_atom.restype = POINTER(c_double)
    elif type == 3:
      self.lib.lammps_extract_atom.restype = POINTER(POINTER(c_double))
    else: return None
    ptr = self.lib.lammps_extract_atom(self.lmp,name)
    return ptr

  def extract_compute(self,id,style,type):
    if type == 0:
      if style > 0: return None
      self.lib.lammps_extract_compute.restype = POINTER(c_double)
      ptr = self.lib.lammps_extract_compute(self.lmp,id,style,type)
      return ptr[0]
    if type == 1:
      self.lib.lammps_extract_compute.restype = POINTER(c_double)
      ptr = self.lib.lammps_extract_compute(self.lmp,id,style,type)
      return ptr
    if type == 2:
      self.lib.lammps_extract_compute.restype = POINTER(POINTER(c_double))
      ptr = self.lib.lammps_extract_compute(self.lmp,id,style,type)
      return ptr
    return None

  # in case of global datum, free memory for 1 double via lammps_free()
  # double was allocated by library interface function
  
  def extract_fix(self,id,style,type,i=0,j=0):
    if style == 0:
      self.lib.lammps_extract_fix.restype = POINTER(c_double)
      ptr = self.lib.lammps_extract_fix(self.lmp,id,style,type,i,j)
      result = ptr[0]
      self.lib.lammps_free(ptr)
      return result
    elif (style == 1) or (style == 2):
      if type == 1:
        self.lib.lammps_extract_fix.restype = POINTER(c_double)
      elif type == 2:
        self.lib.lammps_extract_fix.restype = POINTER(POINTER(c_double))
      else:
        return None
      ptr = self.lib.lammps_extract_fix(self.lmp,id,style,type,i,j)
      return ptr
    else:
      return None

  # free memory for 1 double or 1 vector of doubles via lammps_free()
  # for vector, must copy nlocal returned values to local c_double vector
  # memory was allocated by library interface function
  
  def extract_variable(self,name,group,type):
    if type == 0:
      self.lib.lammps_extract_variable.restype = POINTER(c_double)
      ptr = self.lib.lammps_extract_variable(self.lmp,name,group)
      result = ptr[0]
      self.lib.lammps_free(ptr)
      return result
    if type == 1:
      self.lib.lammps_extract_global.restype = POINTER(c_int)
      nlocalptr = self.lib.lammps_extract_global(self.lmp,"nlocal")
      nlocal = nlocalptr[0]
      result = (c_double*nlocal)()
      self.lib.lammps_extract_variable.restype = POINTER(c_double)
      ptr = self.lib.lammps_extract_variable(self.lmp,name,group)
      for i in xrange(nlocal): result[i] = ptr[i]
      self.lib.lammps_free(ptr)
      return result
    return None

  # set variable value
  # value is converted to string
  # returns 0 for success, -1 if failed
  
  def set_variable(self,name,value):
    return self.lib.lammps_set_variable(self.lmp,name,str(value))

  # return total number of atoms in system
  
  def get_natoms(self):
    return self.lib.lammps_get_natoms(self.lmp)

  # return vector of atom properties gathered across procs, ordered by atom ID

  def gather_atoms(self,name,type,count):
    natoms = self.lib.lammps_get_natoms(self.lmp)
    if type == 0:
      data = ((count*natoms)*c_int)()
      self.lib.lammps_gather_atoms(self.lmp,name,type,count,data)
    elif type == 1:
      data = ((count*natoms)*c_double)()
      self.lib.lammps_gather_atoms(self.lmp,name,type,count,data)
    else: return None
    return data

  # scatter vector of atom properties across procs, ordered by atom ID
  # assume vector is of correct type and length, as created by gather_atoms()

  def scatter_atoms(self,name,type,count,data):
    self.lib.lammps_scatter_atoms(self.lmp,name,type,count,data)

class reaction:
  """
  This class implement a unimolecular reaction: species(1) -> species(2) using a combination of MD and MC algorithms 
  """
  def __init__(self, units, dim, **pargs):
    """ Initialize some settings and specifications 
    @ lmp: lammps object
    @ units: unit system (si, cgs, etc.)
    @ dim: dimensions of the problems (2 or 3)
    """
    logging.info('Initializing LAMMPS object')

    self.lmp = lammps()
    self.pargs = pargs
    self.monitorList = []
    self.vars = []

    if pargs['ensemble'] == 'nvt':
      self.pargs['ensembleArgs'] =  (pargs['temp'], pargs['temp'], pargs['relax'])

    elif pargs['ensemble'] == 'npt':
      self.pargs['ensembleArgs'] = (pargs['temp'], pargs['temp'], pargs['relax'], pargs['press'], pargs['press'], pargs['prelax'])

    logging.info('Setting up problem dimensions and boundaries')

    self.lmp.command('units {}'.format(units))
    self.lmp.command('dimension {}'.format(dim))
    self.lmp.command('atom_style atomic')
    self.lmp.command('atom_modify map array') # array is faster than hash in looking up atomic IDs, but the former takes more memory
    self.lmp.command('boundary {} {} {}'.format(*pargs['boundary']))
    self.lmp.command('newton off')
    self.lmp.command('processors * * *') # let LAMMPS handle DD

  def createDomain(self):
    """ Define the domain of the simulation
    @ nsys: number of subsystems
    @ pos: 6 x 1 tuple that defines the boundaries of the box 
    """
    logging.info('Creating domain')
    self.lmp.command('lattice none 1.0')

    self.lmp.command('region box block {} {} {} {} {} {}'.format(*self.pargs['box']))
    self.lmp.command('create_box {} box'.format(self.pargs['nSS']))

  def createAtoms(self, region = 'NULL'):
    """ Create atoms in a pre-defined region
    @ N: max total number of particles to be inserted
    @ density: initial density of the particles
    @ vel: 3 x 1 tuple of initial velocities of all particles
    @ args: dictionary of params
    """
    logging.info('Creating atoms')

    for idSS in self.pargs['idSS']:
      seed = np.random.randint(1,10**6)
      self.lmp.command('create_atoms {} random {} {} {}'.format(idSS, self.pargs['Natoms'][idSS-1], seed, region))

  def deleteAtoms(self, group):
    """ Delete atoms belonging to specific group
    """
    logging.info('Deleting atoms in group {}'.format(group))
    self.lmp.command('delete_atoms group {}'.format(group))

  def createGroup(self, group = None):
    """ Create groups of atoms 
    """
    logging.info('Creating atom group {}'.format(group))

    if group is None:
      for idSS in self.pargs['idSS']:
        self.lmp.command('group group{} type {}'.format(idSS, idSS))

  def setupNeighbor(self):
    """
    """
    logging.info('Setting up nearest neighbor searching parameters')
    self.lmp.command('neighbor 1 multi')
    self.lmp.command('neigh_modify delay 0')

  def createProperty(self, var, prop, type, valueProp, valueType):
    """
    Material and interaction properties required
    """
    logging.info('Creating proprety')
    self.lmp.command('fix {} all property/global {} {} {} {}'.format(var, prop, type, valueProp, valueType))

  def setupPhysics(self, ftype = 'lj/cut', rc = 2.5):
    """
    Specify the interation forces
    """
    logging.info('Setting up interaction parameters')
    self.lmp.command('pair_style {} {}'.format(ftype, rc))
    indices = []

    for idSS in self.pargs['idSS']:
      indices.append( (self.pargs['idSS'][idSS-1], self.pargs['idSS'][idSS-1]) ) 

    crossIndices = itertools.combinations(self.pargs['idSS'], 2)

    for rt in crossIndices:
      indices.append(rt)

    print indices

    for ind in indices:
      # pair_coeff epsilon sigma cutoff (LJ) [cutoff (Coulomb)]
      self.lmp.command('pair_coeff {} {} 0.356359487256 0.46024 {}'.format(ind[0], ind[1], rc))

  def initializeVelocity(self):
    """
    """
    logging.info('Initializing atomic velocities')

    seed = np.random.randint(1,10**6)
    self.lmp.command('velocity all create {} {}'.format(self.pargs['temp'] * 0.3, seed))

  def setupMass(self):
    """
    """

    for idSS in self.pargs['idSS']:
      logging.info('Setting up masses for species {}'.format(idSS))
      self.lmp.command('mass {} {}'.format(idSS, self.pargs['mass'][idSS-1]))

  def initialize(self):
    """
    """
    self.createDomain()
    self.createAtoms()
    self.setupMass()
    self.createGroup()
    self.initializeVelocity()
    self.setupNeighbor()

  def setupIntegrate(self, name, dt = None):
    """
    Specify how Newton's eqs are integrated in time. 
    @ name: name of the fixed simulation ensemble applied to all atoms
    @ dt: timestep
    @ ensemble: ensemble type (nvt, nve, or npt)
    @ args: tuple args for npt or nvt simulations
    """

    logging.info('Setting up integration scheme parameters')

    if self.pargs['ensemble'] is 'nvt':
      logging.info('Running NVT simulation ...')
      command = 'fix {} all {} temp {} {} {}'

    elif self.pargs['ensemble'] is 'npt':
      logging.info('Running NPT simulation ...')
      command = 'fix {} all {} temp/rescale {} {} {} iso {} {} {}'
    else:
      raise ValueError

    #self.lmp.command(command.format(name, self.pargs['ensemble'], *self.pargs['ensembleArgs']))

    self.lmp.command('fix 3 all temp/rescale 100 300.0 300.1 0.02 0.5')

    if dt is None:
      self.lmp.command('timestep {}'.format(self.pargs['dt']))

  def integrate(self, steps):
    """
    Run simulation in time
    """
    logging.info('Integrating the system for {} steps'.format(steps))

    self.lmp.command('run {}'.format(steps))

    for tup in self.monitorList:
      self.lmp.command('compute {} {} {}'.format(*tup))
      print 'compute {} {} {}'.format(*tup)
      self.vars.append(self.lmp.extract_compute('thermo_temp', 0, 0))
      self.lmp.command('uncompute {}'.format(tup[0]))

  def printSetup(self, freq):
    """
    Specify which variables to write to file, and their format
    """
    logging.info('Setting up printing options')

    self.lmp.command('thermo_style custom' + ' %s '*len(self.pargs['print']) % self.pargs['print'])
    self.lmp.command('thermo {}'.format(freq))
    self.lmp.command('thermo_modify norm no lost ignore')

  def dumpSetup(self, sel, freq, traj):
    """
    """
    logging.info('Setting up trajectory i/o')
    traj, trajFormat = traj.split('.')

    self.lmp.command('dump dump {} xyz {} {}.{}'.format(sel, freq, traj, trajFormat))

  def extractCoords(self, coords):
    """
    Extracts atomic positions from a certian frame and adds it to coords
    """
    # Extract coordinates from lammps
    self.lmp.command('variable x atom x')
    x = Rxn.lmp.extract_variable("x", "group1", 1)

    self.lmp.command('variable y atom y')
    y = Rxn.lmp.extract_variable("y", "group1", 1)

    self.lmp.command('variable z atom z')
    z = Rxn.lmp.extract_variable("z", "group1", 1)

    for i in range(Rxn.lmp.get_natoms()):
      coords[i,:] += x[i], y[i], z[i]

    self.lmp.command('variable x delete')
    self.lmp.command('variable y delete')
    self.lmp.command('variable z delete')

    return coords

  def computeRxn(self, prob, rc):
    """ 
    Computes reaction probability for species '1' based on each atom's coordination number
    with respect to the surrounding species '2'.
    """
    logging.info('Computing reaction probability')
    
    # Do NNS
    # tree = spatial.KDTree(coords)
    # dist, indices = tree.query(x=tree.data, k=2) 

    # Compute reaction probability
    #for distance in dist:
    # if distance <= rc:
    
    nAtoms = self.lmp.get_natoms()

    # this means the two atoms might react 
    # compute reaction probability here
    if np.random.rand() >= prob:
      # delete atom belonging to group1
      self.lmp.command('delete_atoms overlap {} group1 group2'.format(rc))

      nReact = nAtoms - self.lmp.get_natoms()
      logging.info('A total of {} atoms have reacted'.format(nReact))

      seed = np.random.randint(1,10**6)
      self.lmp.command('create_atoms {} random {} {} {}'.format(2, nReact, seed, 'NULL'))

  def monitor(self, name, group, var):
    """
    """
    self.monitorList.append((name, group, var))

  def __del__(self):
    """ Destructor
    """
    self.lmp.close()
