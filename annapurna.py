#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Filip Stefaniak, Janusz M. Bujnicki,
# AnnapuRNA: a scoring function for predicting RNA-small molecule interactions,
# bioRxiv 2020.09.08.287136; doi: https://doi.org/10.1101/2020.09.08.287136
# https://github.com/filipspl/AnnapuRNA

import sys, getopt
import argparse
from Bio import PDB
import numpy as np
import os, errno, time
import subprocess # for execution shell command
import pandas as pd
import math	# for rmsd
import shutil   # copying file

import resource		# memory usage

from Bio.PDB.PDBExceptions import PDBConstructionWarning 	# handle biopython warnings
import warnings							# handle biopython warnings

from pdb_parser_lib import *	# yapdb_parser classes

import openbabel		# for calculation of Energy for full atom representation pf the molecule
import pybel			# for clustering

#---------------------------------------------------------#

### Fine tuning ###

chunksize = 2000000		# adjust according to the available RAM memory

# H2O cluster address and port
# use 127.0.0.1 for the localhost
# for non-local ports the model directory must be visible by the h2o instance

h2o_ip = "127.0.0.1"

modelDir = os.path.dirname(os.path.realpath(__file__)) + "/models/"

# averageStructure = True
averageStructure = False # default

#---------------------------------------------------------#

# DOCS:
# H2O and python: https://h2o-release.s3.amazonaws.com/h2o/rel-shannon/26/docs-website/h2o-py/docs/h2o.html
# 		  https://h2o-release.s3.amazonaws.com/h2o/rel-turing/10/docs-website/h2o-py/docs/h2o.html

#---------------------------------------------------------#

def parse_options():
    parser = argparse.ArgumentParser(description='AnnapuRNA: Coarese Grained Scoring Function for RNA-Ligand Complexes', add_help=False )

    group = parser.add_argument_group('required arguments')

    group.add_argument("-r", "--rna", dest="rnaFile", required=True,
                      help="RNA pdb file (full atom)")
    group.add_argument("-l", "--ligand", dest="ligandFile", required=True,
                      help="ligand file (sdf, mol2, mol, pdb or any other understood by OpenBabel)")
    group.add_argument("-m", "--model", dest="modelName", required=True, choices=['DL_basic', 'DL_modern', 'kNN_basic', 'kNN_modern', 'RF_modern', 'NB_modern', 'ALL'], action='append', default=[],
                      help="prediction model to be used; may be used multiple times for scoring with different models. Use --info for more information on available models. Use ALL to use all available models.")

    group = parser.add_argument_group('optional arguments')

    # WARNING adjust default values to the benchmark best results.
    group.add_argument("--clustering_method", dest="ClusteringMethod", default=False, choices=[False, 'AD','SR', 'AP'],
                      help="Clustering method. AD = AutoDock-like; SR = SimRNA-like; AP = Affinity Propagation.")

    group.add_argument("--cluster_fraction", dest="ClusteringFraction", default=0, type=float,
                      help="Docking poses clustering. Select this fraction of top scoring poses. 0-1. 0 = do not cluster results")
    group.add_argument("--cluster_cutoff", dest="ClusteringCutoff", default=0, type=float,
                      help="Docking poses clustering. Use this RMSD cutoff for clustering. 0 = do not use the RMSD cutoff")



    group.add_argument("-e", "--weight_ligand_energy", dest="EnergyWeight", default=0.1, type=float,
                      help="weight for a ligand's energy term. Default: 0.1. 0 (zero) = do not use the energy term.")

    group.add_argument("-w", "--weight_distance", dest="useWeightDistance", default=False, choices=[False, 'L-J','linear', '1/x', 'exp', 'x^2', 'log'],
                      help="weight probabilities by distance depending function. False = don't weight by distance (default)")
    group.add_argument("-d", "--distance_cutoff", dest="useDistanceCutoff", default=False, type=float,
                      help="use distance cutoff. 0-10 Å. Default: 10 Å.")
    group.add_argument("-t", "--transform_proba", dest="doTransformProba", default=False, choices=[False, 'PMF'],
                      help="transform calculated probabilities. Default: false")


    group.add_argument("-o", "--output", dest="outputFilename", default="./annapurna-stats",
                      help="output filename prefix")
    group.add_argument("-p", "--port", dest="h2oPort", default="30000",
                      help="port of H2O server")

    group.add_argument("-g", "--groupby", dest="groupByName", action='store_true',
                      help="in addition, output scores with a single best score for each compound.")
    group.add_argument("--merge", dest="mergeOutputs", action='store_true',
                      help="merge prediction from multiple models into a single file.")
    group.add_argument("-O", "--overwrite", dest="overwriteResults", action='store_true',
                      help="overwrite the output file, if exists")
    group.add_argument("-s", "--skip_statistics", dest="skipStatistics", action='store_true',
                      help="skip collecting statistics if the intermediate file exists; may be used for applying other available models to aleready collected data.")
    group.add_argument("--dont_clean", dest="skipCleaning", action='store_true',
                      help="don't perform structure cleaning.")
    group.add_argument("--dont_simrnate", dest="skipSimrnaing", action='store_true',
                      help="don't perform coarse graining of the input structure. Assumes the RNA is already in SimRNA format.")


    group.add_argument("-i", "--info", action=_getMoreInfo,
                      help="display information on available models and exit.")

    group.add_argument('-V', '--verbose', dest='beVerbose', action='store_true')
    #group.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
    group.add_argument("-h", "--help", action="help", help="show this help message and exit.")

    return parser.parse_args()


## helper application for argparse to show model information ##
class _getMoreInfo(argparse.Action):
    def __init__(self,
                 option_strings,
                 dest=argparse.SUPPRESS,
                 default=argparse.SUPPRESS,
                 help=None):
        super(_getMoreInfo, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        modelInfo()
        parser.exit()

## model definition to get the proper engine h2o|scipy ##
models_engines = {'DL_basic':'h2o',
                  'DL_modern':'h2o',
                  'NB_modern':'h2o',
                  'kNN_basic':'scipy',
                  'kNN_modern':'scipy',
                  'RF_modern':'scipy'
                  }

## model information for help ##
def modelInfo():
  '''Print information about available models'''
  print "Available models:"
  print "------------------------------"
  print " ALL:		Use all available models."
  print " DL_basic:	Deep Learning Neural Network basing on the basic PDB dataset. Uses H2O server engine."
  print " DL_modern:	Deep Learning Neural Network basing on the modern PDB dataset. Uses H2O server engine."
  print " kNN_basic:	k-nearest neighbors classifier basing on the basic PDB dataset. Uses scikit engine."
  print " kNN_modern:	k-nearest neighbors classifier basing on the modern PDB dataset. Uses scikit engine."
  print ""
  print "UNDOCUMENTED MODELS:"
  print " NB_modern:	Naive Bayes basing on the modern PDB dataset. Uses H2O server engine."
  print " RF_modern:	Random Forests basing on the modern PDB dataset. Uses scikit engine."

  print
  print "More information aobut algorithms used:"
  print "------------------------------"
  print "H2O algorithms: http://docs.h2o.ai/h2o/latest-stable/index.html#algorithms"
  print "scikit-learn algorithms: http://scikit-learn.org/stable/supervised_learning.html#supervised-learning"


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WHITE = '\033[1m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class info:
  ok   = "[" + bcolors.OKGREEN + "  OK  " + bcolors.ENDC + "] "
  fail = "[" + bcolors.FAIL    + " Fail " + bcolors.ENDC + "] "
  info = "[" + bcolors.WHITE   + " Info " + bcolors.ENDC + "] "


# supress biopython warning about lack of element in columns 77-78 for simrna pdb file
# http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM
# 77 - 78        LString(2)    element      Element symbol, right-justified.
warnings.simplefilter('ignore', PDBConstructionWarning)


####### DEFINITIONS #######

#### RNA atom pairs (table 1) - COARSE GRAIN RNA (from SImRNA)
RNApairs = [["C4'", 'P'], ['P', "C4'"],  ["C4'", 'N1'], ["C4'", 'N9'], ['N1', 'C2'], ['N1', 'C4'], ['N9', 'C2'], ['N9', 'C6']]

RNApossibleBases = [ 'C', 'U', 'A', 'G' ]

## list of atoms set in SimRNA CG methods.
RNApossibleAtoms = {}

RNApossibleAtoms['backbone'] = ["P", "C4'"]

# level 1—three beads, positioned at the following atoms: N1-C2-C4 for pyrimidines
RNApossibleAtoms['C'] = ["N1", "C2", "C4"] + RNApossibleAtoms['backbone']
RNApossibleAtoms['U'] = RNApossibleAtoms['C']

# level 1—three beads, positioned at the following atoms: N9-C2-C6 for purines
RNApossibleAtoms['A'] = ["N9", "C2", "C6"] + RNApossibleAtoms['backbone']
RNApossibleAtoms['G'] = RNApossibleAtoms['A']



# triangles for calculating the plane of the base and normals
RNAtriangles = {}
RNAtriangles['A'] = [ ["P","C4'","N9"], ["C2","C6","N9"] ]
RNAtriangles['G'] = [ ["P","C4'","N9"], ["C2","C6","N9"] ]
RNAtriangles['C'] = [ ["P","N1","C4'"], ["N1","C2","C4"] ]
RNAtriangles['U'] = [ ["P","N1","C4'"], ["N1","C2","C4"] ]


### PHARMACOPHORES ###

'''
http://silicos-it.be.s3-website-eu-west-1.amazonaws.com/software/align-it/1.0.4/align-it.html#concept

AROM	Aromatic ring
HDON	Hydrogen bond donor
HACC	Hydrogen bond acceptor
LIPO	Lipophilic region
POSC	Positive charge center
NEGC	Negative charge center
HYBH	Hydrogen bond donor and acceptor
HYBL	Aromatic and lipophilic
EXCL	Exclusion sphere
'''

# arbitrary assigning numbers in array
mol2atomtypes = {'AROM': 1, 'HDON': 2, 'HACC': 3, 'LIPO': 4, 'POSC': 5, 'NEGC': 6, 'HYBH': 7, 'HYBL': 8, 'EXCL': 9}

def norm_angle(df):
  '''Normalize angle in degrees'''
  df -= 0  # equivalent to df = df - 0
  df /= 180  # equivalent to df = df / 180
  return df


def norm_dist(df, distanceCutOff):
  '''Normalize distance'''
  df -= 0  # equivalent to df = df - df.min()
  df /= int(distanceCutOff)  # equivalent to df = df / df.max()
  return df


#### Calculate derivatives of previously calculated values
### formerly known as ogarnij_statystyki function

def normalize_df(df, distanceCutOff=10):
    '''Normalize dataframe - angles and distances'''
    df.dist = norm_dist(df.dist, distanceCutOff)
    df.angle = norm_angle(df.angle)
    df.angleN = norm_angle(df.angleN)
    df.angleCross = norm_angle(df.angleCross)
    df.angleNCross = norm_angle(df.angleNCross)
    return df


def which(program):
    ''' analog of linux which program, based on:
    https://stackoverflow.com/questions/377017/test-if-executable-exists-in-python '''

    import os
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None



def generatePhar(ligandFile):
  '''Convert any docking file to .phar file'''

  alignitPath = which("align-it")

  outputFilename = os.path.splitext(ligandFile)[0] + ".phar"

  bashCommand = "%s --dbase %s --pharmacophore %s --noHybrid" % ( alignitPath, ligandFile, outputFilename)
  process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
  output = process.communicate()[0]

  return outputFilename

def parse_input_ligand_file_to_sdf(ligandFile):
    '''
    reads input ligand file
    fix titles, if necesarry
    and saves it as sdffile
    to be used later;
    returns the new sdf file name with fixed titles
    '''
    filename, extension = os.path.splitext(ligandFile)
    extension = extension.split(".")[-1]
    ligandSdfFileName = "%s.titles.sdf" % (filename)
    compoundId = 0

    # output file
    output = pybel.Outputfile("sdf", ligandSdfFileName, overwrite=True )

    for mol in pybel.readfile(extension, ligandFile):
        compoundId+=1
        #print mol.molwt, mol.title
        title = mol.title
        if title == '':
            # assign subsequent numbers to molecules OR fixed values
            #title = "molecule_%05i" % ( compoundId )
            title = "molecule_with_no_title"
            mol.title = title
        output.write(mol)

    output.close()
    return ligandSdfFileName


def calculate_ligands_energy(ligandFile, outputFilename):
    '''Convert ligand full atom file to set of descriptors (Total Energy values)'''

    obconversion = openbabel.OBConversion()
    obconversion.SetInFormat("sdf")
    obmol = openbabel.OBMol()

    energiesPandasAll = ''

    ## Forcefield setup
    ffield = 'GAFF'
    ff = openbabel.OBForceField.FindForceField(ffield)
    if ff == 0:
          print info.warn + "Could not find the forcefield", ffield
          exit(2)


    notatend = obconversion.ReadFile(obmol,ligandSdfFileName)

    compoundId = 1
    while notatend:
        # here do the magic: iterate over molecules and calculate energies

        #print "MW:", obmol.GetMolWt()
        # NumAtoms () NumHvyAtoms () GetMolWt (bool implicitH=true)	see: http://openbabel.org/dev-api/classOpenBabel_1_1OBMol.shtml

        # remove all hydrogens
        obmol.DeleteHydrogens()
        # and add polar only
        obmol.AddHydrogens(True)

        # Setup the molecule. This assigns atoms types, charges and parameters
        if ff.Setup(obmol) == 0:
            print "Could not setup forcefield:", ffield
            E_total = ''
        else:
            E_total = ff.Energy()

        title = obmol.GetTitle()
        if title == '':
            title = "molecule_%05i" % ( compoundId ) # :WARNING: musimy zapisac ten plik do sdf'a czy coś! To jest później czytane

        energiesPandas = pd.DataFrame.from_dict([{ "compoundId": compoundId, "compound": title, "E_ligand": E_total }])

        if len(energiesPandasAll) == 0:
            energiesPandasAll = energiesPandas
        else:
            energiesPandasAll = pd.concat([energiesPandasAll, energiesPandas])


        # here ends the magic (and iteration...)
        obmol = openbabel.OBMol()
        notatend = obconversion.Read(obmol)
        compoundId += 1

    #return energiesPandasAll
    energiesPandasAll.to_csv(outputFilename, sep=",", index=False, compression='bz2')


def mkdir_p(path):
    '''Simulate bash mkdir -p'''
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


def unit_vector(vector):
    '''Returns the unit vector of the vector.'''
    return vector / np.linalg.norm(vector)

def angle_between(x, y, z):
    '''Calculate angle between three points'''

    x = np.float64(x)
    y = np.float64(y)
    z = np.float64(z)

    v1 = x-y
    v2 = z-y


    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    #angle = np.arccos(np.dot(v1_u, v2_u))		# in radians
    angle = np.degrees(np.arccos(np.dot(v1_u, v2_u))) 	# in degrees
    if np.isnan(angle):
        if (v1_u == v2_u).all():
            return 0.0
        else:
            return np.degrees(np.pi)
    #print angle.dtype

    return angle

def file_append(filename, content):
  '''Append content (string) to a file'''
  f = open( filename, 'a' )
  f.write(content)
  f.close()

def printMemInfo():
  '''Print memory usage - useful for debugging or controlling the memory usage'''
  print info.info + "Maximum resident set size:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "kB;"#, "Shared memory size", resource.getrusage(resource.RUSAGE_SELF).ru_ixrss, "kB;", "Unshared memory size", resource.getrusage(resource.RUSAGE_SELF).ru_idrss, "kB"

# from: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 70):
    """Print progress information.
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '█' * filledLength + ' ' * (barLength - filledLength)
    sys.stdout.write('\r%s [%s] %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write(info.ok + "\n")
    sys.stdout.flush()


def yapdb_parser(rnaFile, rnaFileClean):
    '''
    Based on rna-pdb-tools
    https://github.com/mmagnus/rna-pdb-tools
    Magnus, Marcin. (2016). rna-pdb-tools. Zenodo. 10.5281/zenodo.60933
    '''
    s = StrucFile( rnaFile )

    s.decap_gtp()
    s.fix_resn()
    s.remove_hydrogen()
    s.remove_ion()
    s.remove_water()
    s.fix_op_atoms()
    s.renum_atoms()

    s.get_simrna_ready()

    f = open( rnaFileClean, 'w' )
    f.write( s.get_text() )
    f.close()



def pdb2simrna(rnaFileClean, rnaFileCleanSimrna):
  ''' convert all-atom PDB file to SimRNA representation and save to rnaFileCleanSimrna '''

  p = PDB.PDBParser()
  structure = p.get_structure("pdbfile", rnaFileClean)

  for atom in list(structure.get_atoms()):
    if atom.id not in RNApossibleAtoms[atom.parent.resname.strip()]:
        residue = atom.parent
        residue.detach_child(atom.id)

  from Bio.PDB import PDBIO
  w = PDBIO()
  w.set_structure( structure )
  w.save( rnaFileCleanSimrna )

def checkPdbFile(rnaFile):
    '''Check if all residues are correctly represented, i.e., has all necesary atoms to create SimRNA representation'''

    errors = 0	# number of errors

    p = PDB.PDBParser()
    s = p.get_structure("pdbfile", rnaFile)

    residues = [ res.resname.strip() for res in s.get_residues() ]

    # Check if all residues are OK

    if set(residues) > set( RNApossibleBases ):		# residues > possible bases
      print info.fail + "The structure has unrecognized residues/ligands:", ",".join( list( set(residues) - set( RNApossibleBases ) ) )
      errors += 1

    for res in s.get_residues():
        resname = res.resname.strip()

        if resname in RNApossibleBases:	# we only consider GCAU

            atoms = [atom.name.strip() for atom in res]
            chain = res.get_parent().get_id().strip()
            resNumber = res.get_full_id()[3][1]

            if not set(RNApossibleAtoms[resname]) < set(atoms): # if not all atoms are complete
                print info.fail + "In residue", resname, resNumber, "there are no atoms:", ",".join( list(set(RNApossibleAtoms[resname]) - set(atoms)) )
                errors += 1

    if errors > 0:
      print info.fail + "There are", errors, "errors in the given RNA structure. Please fix it and resubmit."
      exit(2)
    else:
      print info.ok + "The structure is OK!"


def weightDistance(distance_vector, function="L-J"):
  '''Weight distance vector by a given distance function'''

  rm = 0.3 #float(3) / max_distance	# normalized rm

  if function == "L-J":
    '''Uses Lennard-Jones potential 12-6 like function (also termed the L-J potential, 6-12 potential, or 12-6 potential)'''
    # rm is the distance at which the potential reaches its minimum
    # distance is normalized from 0-10 A to 0-1
    # -1*((rm/x)^12 - 2 * (rm/x)^6)

    # max_distance = 10 # we collect information in the radius <= 10 A


    return -1*((rm/distance_vector)**12 - 2 * (rm/distance_vector)**6)

  elif function == "linear":
    return -1 * distance_vector + 1

  elif function == "1/x":
    # (1/(x+0.3))
    return 1 / (distance_vector + rm)

  elif function == "exp":
    # -1*e^x+2
    return -1 * 2.718281 ** distance_vector + 2

  elif function == "x^2":
    # -1*x^2+1
    return -1 * distance_vector ** 2 + 1

  elif function == "log":
    # -1*log(x)
    return -1 * np.log(distance_vector)


def distanceCutoff(distance_vector, cutoff=False):
          distanceCutOff = float(10) # tak było to przeskalowane początkowo
          cutoff = cutoff / distanceCutOff # normalizujemy podany cutoff do 0-1
          return distance_vector <= cutoff


def transformProba(proba, function=False):
  '''Transforms probablility vector'''
  if function != False:
    if function == 'PMF':
      # Potential of Mean Force See: Bernauer, RNA, 2011, 17, 1066-1075; E=-kT sum (ln(Pobs/Pref))
      # Boltzmann constant k = 1.38064852 × 10-23 m2 kg s-2 K-1
      # T = 300 K
      # k*T = 4.1419456e-21

      #return -4.1419456e-21 * np.log(proba)	# np.log = ln
      return -1*np.log(proba)	# np.log = ln

    else:
      print info.fail + "Unknown transformation function", function
      exit(1)

def get_statistics(rnaFile, ligandFile, outputFilename, distanceCutOff=10):
    '''Gather a statistics of RNA-Ligand interactions
    and saves as csv.bz2 file'''

    csvHeaders = ['compoundId', 'compound', 'base', 'at1', 'at2', 'atom_type', 'angle', 'dist', 'angleN', 'angleCross', 'angleNCross']
    pd.DataFrame(columns = csvHeaders).to_csv(outputFilename + ".csv", sep="\t", index=False)
    csvDataToSave = pd.DataFrame({})

    p = PDB.PDBParser()
    s = p.get_structure("pdbfile", rnaFile)

    #ligandName = os.path.basename(ligandFile).split(".")[0]

    RNApairsCoords = {} # coordinates of atoms from RNApairs
    resids = []

    ### iterate through RNA to get coordinates of the atoms

    for at1, at2 in RNApairs:	# for each base pair
      for res in s.get_residues():	# for each RNA residue
          chain = res.get_parent().get_id().strip()
          #print "chain: ", chain
          #exit(1)
          for atom in res:
              resid = res.id[1]
              resids += [resid]	# residue numbers
              (resname, name, coord) = res.resname.strip(), atom.name.strip(), atom.coord
              # G C5' [-15.50800037  -7.05600023  13.91800022]
              if (name == at1 or name == at2):
                RNApairsCoords[chain, resid, name]  = [resname, coord]

    resids = set(resids)
    seqLen = len(resids) # sequence lengths

    # precalculate cross (normalne - wektory prostopadłe do płaszczyzny) for all residues
    RNAcross = {}
    for res in s.get_residues():
      resname = res.get_resname().strip()
      resid = res.get_id()[1]
      chain = res.get_parent().get_id().strip()
      for Tat1, Tat2, Tat3 in RNAtriangles[resname]:
        #print "resname, resid, [Tat1, Tat2, Tat3]", resname, resid, Tat1, Tat2, Tat3	# G 1 P C4' N9
        Tat1coord = res[Tat1].get_coord()	# eg: [ 66.93199921  54.71699905  32.50600052]
        Tat2coord = res[Tat2].get_coord()
        Tat3coord = res[Tat3].get_coord()

        v1 = Tat2coord - Tat1coord # vector from eg C2 to C6
        v2 = Tat3coord - Tat2coord # vevtor from eg C6 to N9

        cross = np.cross(v1, v2)

        RNAcross[chain, resid, Tat1] = cross + Tat1coord	# cross vector moved to the position of Tat atom
        RNAcross[chain, resid, Tat2] = cross + Tat2coord
        RNAcross[chain, resid, Tat3] = cross + Tat3coord	# N9 from the base triangle superseed the N9 normal from the backbone
        # example: (1, 'P'): array([ 56.68252563,  46.80329514,  33.73231888]

    print info.info + "Collecting statistics..."
    RNApairsLen = len(RNApairs)
    RNApairsCount = 0

    for at1, at2 in RNApairs:	# for each base pair (table 1)	   at2 - the "main" RNA atom
      RNApairsCount += 1
      #print info.info, "[%i/%i] %s-%s" %(RNApairsCount,RNApairsLen, at1, at2)
      resCount = 0

      for res in s.get_residues():
        resCount += 1
        resname = res.get_resname().strip()
        resid = res.get_id()[1]
        chain = res.get_parent().get_id().strip()		# for each unique residue numbers

        printProgress(resCount, seqLen, prefix="[%i/%i] %s:%s-%s\t" %(RNApairsCount,RNApairsLen, resname, at1, at2))


        # find all RNA pairs and coordinates for contacts
        if (chain, resid,at1) in RNApairsCoords and (chain, resid, at2) in RNApairsCoords:
          base = RNApairsCoords[(chain, resid, at1)][0]

          at1xyz = np.array(RNApairsCoords[(chain, resid, at1)][1])
          at2xyz = np.array(RNApairsCoords[(chain, resid, at2)][1])

          with open(ligandFile,'r') as file:
              compounds = file.read()
          #file.close()

          #for compound in compounds.strip().split('$$$$'):
          # process the phar file and save data to
          '''Process the data read from phar file (compounds) and append it to the already initialized outputFilename file'''

          compoundNr = 0	# subsequent number of compund/pose in the collective file
          for compound in compounds.split('$$$$\n'):

            # if compound is not empty, proceed
            if compound != '':
              for lineNr, line in enumerate(compound.rstrip().splitlines()): # .rstrip() = strip at the end of the string

              #print lineNr, line
              #print compoundNr

               # the first line of the compound. let's look for the compound's name
               if lineNr == 0:
                  compoundNr += 1
                  # empty title line, let's make up a title
                  if line.strip() == "":
                      compoundTitle = "molecule_%05i" % ( compoundNr )

                  # nonempty line, let's extract the title
                  else:
                      compoundTitle = line.replace("\t", " ")	# not to mess with TSV files

               elif len(line.split()) == 9:
                  # CODE Cx Cy Cz α norm Nx Ny Nz
                  atom_type, x, y, z, alpha, norm, Nx, Ny, Nz = line.split()
                  x, y, z = float(x), float(y), float(z)
                  Nx, Ny, Nz = float(Nx), float(Ny), float(Nz)

                  ligxyz = np.array([x, y, z])
                  dist = np.linalg.norm(ligxyz - at2xyz)
                  #print dist
                  if dist <= distanceCutOff:
                    angle = angle_between(at1xyz, at2xyz, ligxyz)

                    if int(norm) == 1:		# if there is vector to calculate
                      # angle between Norm vector of pharmacophore and RNA atom
                      ligNxyz = np.array([Nx, Ny, Nz])
                      angleN = angle_between(ligNxyz, ligxyz, at2xyz)

                      # angle between Norm vector of pharmacophore and Cross vector (wektor prostopadły do płaszczyzny zasady)
                      # move Norm vector to base atom x y z (at2xyz)
                      przesuniecieOWektor = ligxyz - at2xyz
                      ligNxyzAt2 = ligNxyz - przesuniecieOWektor
                      angleNCross = angle_between(ligNxyzAt2, at2xyz, RNAcross[chain, resid, at2])

                    else:
                      angleN = -1
                      angleNCross = -1

                    ### Kąty miedzy płaszczyznami a punktem

                    angleCross = angle_between(ligxyz, at2xyz, RNAcross[chain, resid, at2])

                    #file_append(outputFilename + ".csv", "\t".join([str(x) for x in (compoundNr, compoundTitle, base, at1, at2, atom_type, angle, dist, angleN, angleCross, angleNCross)]) + "\n" )

                    dataArr = pd.DataFrame([compoundNr, compoundTitle, base, at1, at2, atom_type, angle, dist, angleN, angleCross, angleNCross]).T

                    if len(csvDataToSave) == 0:
                      csvDataToSave = pd.DataFrame({})		# initializing empty data frame

                    csvDataToSave = pd.concat([csvDataToSave, dataArr])
          #tu jest koniec loopki związku $$$$
           # end of compound here, nonempty
           #print "koniec związku niepustego"
        if len(csvDataToSave) > 0:
              # normalize and then append to csv
              csvDataToSave.columns = csvHeaders

              normalize_df(csvDataToSave).to_csv(outputFilename + ".csv", sep="\t", index=False, mode='a', header=False)
              csvDataToSave = pd.DataFrame({})





      # end compound
      #print "\t", info.ok, "Done"
    print info.info + "bzip2 compression of statistics..."
    bashCommand = "bzip2 %s.csv" % ( outputFilename)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output = process.communicate()[0]

##------------------------------------------------------------------------------------------------------------------------ ###

## RMSD and clustering

def squared_distance(coordsA, coordsB):
    """Find the squared distance between two 3-tuples"""
    if coordsA != None and coordsB != None:
      sqrdist = sum( (a-b)**2 for a, b in zip(coordsA, coordsB) )
      return sqrdist
    else:
      #print coordsA, coordsB
      return 0 # cholera, a może coś innego byłoby lepsze?

def rmsd(allcoordsA, allcoordsB):
    """Find the RMSD between two lists of 3-tuples"""
    deviation = sum(squared_distance(atomA, atomB) for
                    (atomA, atomB) in zip(allcoordsA, allcoordsB))
    return math.sqrt(deviation / float(len(allcoordsA)))


def calc_rmsd(dockedpose, referenceMol):
    mappings = pybel.ob.vvpairUIntUInt()
    bitvec = pybel.ob.OBBitVec()
    lookup = []
    for i, atom in enumerate(referenceMol):
        if not atom.OBAtom.IsHydrogen():
            bitvec.SetBitOn(i+1)
            lookup.append(i)
    referenceMolcoords = [atom.coords for atom in referenceMol if not atom.OBAtom.IsHydrogen()]
    success = pybel.ob.FindAutomorphisms(referenceMol.OBMol, mappings, bitvec)

    posecoords = [atom.coords for atom in dockedpose if not atom.OBAtom.IsHydrogen()]
    minrmsd = 999999999999
    for mapping in mappings:
        automorph_coords = [None] * len(referenceMolcoords)
        for x, y in mapping:
            automorph_coords[lookup.index(x)] = referenceMolcoords[lookup.index(y)]
        mapping_rmsd = rmsd(posecoords, automorph_coords)
        if mapping_rmsd < minrmsd:
            minrmsd = mapping_rmsd
    return float("%.2f" % (minrmsd))



def calculateRmsdMatrix(scoresAndMolecules):	# resultsRMSD - global RMSD to the reference structure, subdir - positive/negative
  '''Calculate RMSD matrix from the input structural file'''

  print info.info + "Calculating RMSD matrix ....", len(scoresAndMolecules), "x", len(scoresAndMolecules)
  rmsdMatrix = np.array([])
  firstLine = True
  ilePoz = len(scoresAndMolecules)
  pozaNr = 0

  ## Liczenie wzajemne macierzy RMSD n x n

  for row in scoresAndMolecules.iterrows():
    mol = row[1]['molecule']

    lineRmsdTmp = np.array([])
    pozaNr += 1

    printProgress(pozaNr, ilePoz, prefix="[%i/%i] " %(pozaNr, ilePoz))

    poza2Nr = 0
    for row2 in scoresAndMolecules.iterrows():
        mol2 = row2[1]['molecule']

        poza2Nr += 1
        if poza2Nr >= pozaNr:
            if mol != mol2:
              rmsd = float(calc_rmsd(mol, mol2))
            else:
              rmsd = 0
        else:
           rmsd = False # czyli zero, szczerze mówiąc

        lineRmsdTmp = np.append(lineRmsdTmp, rmsd)

    if firstLine == True:
      rmsdMatrix = lineRmsdTmp
      firstLine = False
    else:
      rmsdMatrix = np.vstack( (rmsdMatrix, lineRmsdTmp ))


  ## symmetrization of the matrix
  rmsdMatrix = rmsdMatrix + rmsdMatrix.T - np.diag(rmsdMatrix.diagonal())

  return rmsdMatrix



def sdfread(sdffile):
  '''Put SDF data to a pandas DataFrame'''

  molecules = pd.DataFrame({})

  headers = ['compoundId', 'compound', 'molecule']
  molecules = pd.DataFrame(columns = headers)
  molecules.molecule.astype('int64')


  pose = 0
  title = ""

  for mol in pybel.readfile('sdf', sdffile):
    pose += 1

    # the first iteration
    if title == "":
      title = mol.title

    # nowy tytuł cząsteczki, znaczy się, zmieniamy numer pozy
    if mol.title != title:
      title = mol.title

    molecules.loc[len(molecules)]=[pose, mol.title, mol]

  return molecules



def cluster(outputScore, sdffile, ClusteringMethod, topFraction, rmsdCutoff, outputFilename, modelName):
  '''reading molecules from SDF file and put it to the pandas DataFrame
  cluster results
  put diversified results to the SDF file and returns diversified input table.'''

  print info.info + "Clustering with " + ClusteringMethod

  molecules = sdfread(sdffile) # tu muszą być nazwy!

  # tu grupowanie po molekule
  outputScoreGrouped = outputScore.groupby(by=['compound'])

  selectedMoleculesGlobal = ""	# global set of selectedMolecules

  for name, outputScoreGrupa in outputScoreGrouped:
      print info.info + "Processing", name, "..."

      # take topFraction of best scores
      topFractionScores = outputScoreGrupa.nsmallest(int(topFraction*len(outputScoreGrupa)), 'score' ) #.ix[:, ['compoundId', 'compound', 'score']]
      # merge
      scoresAndMolecules = pd.merge(topFractionScores, molecules, left_on=['compoundId', 'compound'], right_on=['compoundId', 'compound'],  how='left')

      # print scoresAndMolecules

      ##############################################
      ## AutoDock Style Clustering
      ##############################################
      if ClusteringMethod == 'AD':
        print info.info + "AD Clustering..."
        clusterNumber = 0
        outputFilenameSDF = "%s_clusters__%s_TOP%s_RMSD%s_AD_representatives.sdf" % (outputFilename, modelName, topFraction, rmsdCutoff)
        # initialize structure with the one, first, top scoring molecule:
        selectedMolecules = pd.DataFrame(scoresAndMolecules.ix[0]).T
        selectedMolecules.set_value(0, 'cluster', 0 )

        #selectedMolecules = pd.DataFrame(data=None, columns=scoresAndMolecules.columns,index=scoresAndMolecules.index)

        # iterate row by row and select diverse poses
        for index, molecule in scoresAndMolecules.iloc[1:].iterrows(): ### .iloc[1:].- skip the zero row
          mol = molecule['molecule']

          # compare with already selected molecules
          rmsds = [] # values of rmsds of the big pool

          for index2, selectedMolecule in selectedMolecules.iterrows():
              rmsd = calc_rmsd(selectedMolecule['molecule'], mol)
              rmsds.extend([rmsd])

          # dodajemy każdą molekułę
          selectedMolecules.loc[len(selectedMolecules)] = molecule
          if min(rmsds) >= rmsdCutoff:
              ## selecting - new cluster
              clusterNumber+=1
              clusterNumberInsert = clusterNumber
              #scoresAndMolecules.set_value(len(selectedMolecules), 'cluster', clusterNumber)

              print "     ", index+1, " minRMSD:", min(rmsds), ". Starting a new cluster #", clusterNumberInsert
          else:
              ## dodajemy do już istniejącego klastra.
              mostSimilarCompound = rmsds.index(min(rmsds))
              clusterNumberInsert = selectedMolecules.ix[mostSimilarCompound]['cluster']

              #print "rmsd:", rmsds, "min(rmsds)", min(rmsds), "rmsds.index(min(rmsds))", rmsds.index(min(rmsds))
              print "     ", index+1, " minRMSD:", min(rmsds), ". Adding to the existing cluster #", clusterNumberInsert
              #selectedMolecules.loc[len(selectedMolecules)] = molecule

          selectedMolecules.set_value(index, 'cluster', int(clusterNumberInsert) )

        selectedMoleculesGrouped = selectedMolecules.groupby('cluster')
        for name, klasterDane in selectedMoleculesGrouped:

          #print klasterDane #tutaj
          ### uśrednianie struktury
          if averageStructure == True:
              doKlastraDane = averageMolecule(klasterDane).to_frame().T

          ### selection of the best molecule (they go to: selectedMoleculesGlobal)
          else:
              doKlastraDane = klasterDane.iloc[0, :].to_frame().T	# najlepiej oceniona struktura

          # print "klasterDane"
          # print klasterDane
          # print ""
          # print "doKlastraDane"
          # print doKlastraDane
          # exit(1)
          # dotąd jest ok

          if len(selectedMoleculesGlobal) == 0:
            selectedMoleculesGlobal = doKlastraDane
          else:
            selectedMoleculesGlobal = pd.concat([ selectedMoleculesGlobal, doKlastraDane ])


        #print selectedMoleculesGlobal

      ##############################################
      ## SimRNA style clustering with Affinity AffinityPropagation	# Enter Sandman
      ##############################################
      elif ClusteringMethod == 'AP':
        print info.info + "AP Clustering..."
        from sklearn.cluster import AffinityPropagation	# for clustering

        outputFilenameSDF = "%s_clusters__%s_TOP%s_AP_representatives.sdf" % (outputFilename, modelName, topFraction)

        selectedMolecules = pd.DataFrame(data=None, columns=scoresAndMolecules.columns)
        rmsdMatrix = calculateRmsdMatrix(scoresAndMolecules)
        rmsdMatrix = -1 * rmsdMatrix # similarity

        if len(rmsdMatrix) >= 3:
            #print "len(rmsdMatrix)", len(rmsdMatrix)

            af = AffinityPropagation(affinity="precomputed").fit(rmsdMatrix)	# Data matrix or, if affinity is precomputed, matrix of similarities / affinities.
            cluster_centers_indices = af.cluster_centers_indices_
            labels = af.labels_
            n_clusters_ = len(cluster_centers_indices)
            #print 'Estimated number of clusters: %d' % (n_clusters_)
            #print "cluster_centers_indices", cluster_centers_indices

            clusterLens = {}
            clusterLens = [ len(np.nonzero( labels == k )[0]) for k in range(n_clusters_)  ]
            #print "clusterLens: ", clusterLens

            clusterLensSortedIndexes = sorted(range(len(clusterLens)), key=lambda k: clusterLens[k], reverse=True)
            #print "clusterLensSortedIndexes", clusterLensSortedIndexes

            #i=0
            for k in clusterLensSortedIndexes:

                class_members = np.nonzero( labels == k )[0]

                #print "class_members", class_members
                #print "cluster size:", len(class_members)
                #print "center of the cluster:", cluster_centers_indices[k]

                #print "członkowie klastra:"
                #print scoresAndMolecules.ix[ class_members, : ]
                #print "Środek klastra:"
                srodekKlastraDane = scoresAndMolecules.ix[ cluster_centers_indices[k], : ]

                ## Multiply te score by the cluster size. Give more power to the top populated clusters
                srodekKlastraDane['score'] = srodekKlastraDane['score'] * len(class_members)

                selectedMolecules.loc[len(selectedMolecules)] = srodekKlastraDane

                ###
                '''
                ## save clusters to separate files
                i+=1
                outputMol = pybel.Outputfile("sdf", "/home/fstefaniak/gitlab/CGIRLS-release/tests/1AJU-sdf/output-sdf/cgirls-stats_clusters__kNN_basic_TOP0.5_AffinityPropagation_klaster_%s.sdf" % (i), overwrite=True)
                for class_member in class_members:
                    mol = scoresAndMolecules.ix[ class_member, : ]['molecule']
                    #outputMol.write(resultsPose[cluster_center])
                    outputMol.write(mol)
                outputMol.close()

                '''
                ###

        else:
            print "Less than three poses. We are taking them all!"
            selectedMolecules = scoresAndMolecules

        if len(selectedMoleculesGlobal) == 0:
          selectedMoleculesGlobal = selectedMolecules
        else:
          selectedMoleculesGlobal = pd.concat([selectedMoleculesGlobal, selectedMolecules])


      ##############################################
      ## SimRNA style clustering with distance cutoff
      ##############################################
      elif ClusteringMethod == 'SR':
        print info.info + "SR Clustering..."

        outputFilenameSDF = "%s_clusters__%s_TOP%s_RMSD%s_SR_representatives.sdf" % (outputFilename, modelName, topFraction, rmsdCutoff)

        selectedMolecules = pd.DataFrame(data=None, columns=scoresAndMolecules.columns)
        rmsdMatrix = calculateRmsdMatrix(scoresAndMolecules)

        #np.set_printoptions(threshold=np.nan, linewidth=2000)
        #print rmsdMatrix
        #np.savetxt("rmsdMatrix.csv", rmsdMatrix, delimiter=",")
        #exit(2)

        if len(rmsdMatrix) >= 3:
            rmsdZeroJeden = rmsdMatrix <= rmsdCutoff
            #print rmsdZeroJeden

            sumaWKlastrze = rmsdZeroJeden.sum(axis=1)	# suma wystąpień struktur o rmsd < cutoff

            orginalIndexes = range(0, len(sumaWKlastrze) ) # orginalne indeksy molekuł
            #print "orginalIndexes", orginalIndexes

            while(len(sumaWKlastrze) > 0):


                #print "sumaWKlastrze", sumaWKlastrze
                #print "len sumaWKlastrze", len(sumaWKlastrze)

                pierwszyArgmax =  np.argmax(sumaWKlastrze) # indeks maksymalnej wartości (sumy)
                #print "pierwszyArgmax", pierwszyArgmax

                indeksyKlastra = np.nonzero( rmsdZeroJeden[ pierwszyArgmax ] )[0]
                #print "indeksyKlastra aktualnego", indeksyKlastra

                indeksyKlastraOrginalnego = [orginalIndexes[i] for i in indeksyKlastra]
                #print "indeksyKlastraOrginalnego", indeksyKlastraOrginalnego	# to są indeksy orginalne cząsteczek
                #print "członkowie klastra explicite:"

                czlonkowieKlastra = scoresAndMolecules.ix[ indeksyKlastraOrginalnego, : ]

                wielkoscKlastra = len(indeksyKlastra)
                #print "wielkoscKlastra", wielkoscKlastra


                if averageStructure == True:

                    srodekKlastraDane = averageMolecule(czlonkowieKlastra)

                else:

                    reprezentantKlastra = indeksyKlastraOrginalnego[0]
                    #print "reprezentant klastra", reprezentantKlastra	# to są indeksy oryginalne cząsteczki

                    srodekKlastraDane = scoresAndMolecules.ix[ reprezentantKlastra, : ]

                    ## Multiply te score by the cluster size. Give more power to the top populated clusters
                    srodekKlastraDane['score'] = srodekKlastraDane['score'] * wielkoscKlastra

                ## else end -----

                ## postprocessing arrays
                # usunięcie z macierzy elementów zdefiniowanego właśnie klastra
                rmsdZeroJeden = np.delete(rmsdZeroJeden, indeksyKlastra, axis=0)
                rmsdZeroJeden = np.delete(rmsdZeroJeden, indeksyKlastra, axis=1)
                #print rmsdZeroJeden.shape

                orginalIndexes = [x for i,x in enumerate(orginalIndexes) if i not in indeksyKlastra] # remove indexes indeksyKlastra from orginal indexes
                #print "orginalIndexes po wycięciu tegoż klastra", orginalIndexes


                selectedMolecules.loc[len(selectedMolecules)] = srodekKlastraDane

                sumaWKlastrze = rmsdZeroJeden.sum(axis=1)	# suma wystąpień struktur o rmsd < cutoff

                #print scoresAndMolecules.ix[4]
                #print "------"

            # while ends here

        else:
            print "Less than three poses. We are taking them all!"
            selectedMolecules = scoresAndMolecules

        if len(selectedMoleculesGlobal) == 0:
          selectedMoleculesGlobal = selectedMolecules
        else:
          selectedMoleculesGlobal = pd.concat([selectedMoleculesGlobal, selectedMolecules])

      #print selectedMoleculesGlobal
      #koniec SR


  # save diversified molecules
  outputMol = pybel.Outputfile("sdf", outputFilenameSDF, overwrite=True)
  for index, selectedMoleculeGlobal in selectedMoleculesGlobal.iterrows():
      #outputMol.write(resultsPose[cluster_center])
      mol = selectedMoleculeGlobal['molecule']
      mol.data['Pose_Number'] = selectedMoleculeGlobal['compoundId']
      #mol.data['SANTIAGO:%s' % (modelName) ] = selectedMoleculeGlobal['score']
      mol.data['AnnapuRNA Score'] = selectedMoleculeGlobal['score']
      outputMol.write(mol)

  outputMol.close()

  ### copy output file to the file with a simple name
  shutil.copy(outputFilenameSDF, "%s_output.sdf" % (outputFilename) )


  if averageStructure == True:
      optimize3D(outputFilenameSDF)

  # return diversifiedScores = outputScore and diversified poses
  diversifiedScores = selectedMoleculesGlobal.drop('molecule', 1)   #.loc[:, ['compoundId',       'compound',    'score']]
  return diversifiedScores


def saveSdfWithScores(outputScore, sdffile, outputFilename, modelName):
    ''' Saves sdf file with input structures with scores field attached '''
    outputFilenameSDF = "%s__%s+scores.sdf" % (outputFilename, modelName)


    molecules = sdfread(sdffile) # tu muszą być nazwy!

    scoresAndMolecules = pd.merge(outputScore, molecules, left_on=['compoundId', 'compound'], right_on=['compoundId', 'compound'],  how='left')

    outputMol = pybel.Outputfile("sdf", outputFilenameSDF, overwrite=True)
    for index, scoreAndMolecule in scoresAndMolecules.iterrows():
        #outputMol.write(resultsPose[cluster_center])
        mol = scoreAndMolecule['molecule']
        mol.data['Pose_Number'] = scoreAndMolecule['compoundId']
        mol.data['AnnapuRNA Score'] = scoreAndMolecule['score']
        outputMol.write(mol)

    outputMol.close()

    ### copy output file to the file with a simple name
    shutil.copy(outputFilenameSDF, "%s_output.sdf" % (outputFilename) )

def averageMolecule(zawartoscKlastraDane):
    '''
    Average coordinates of the molecule;
    nazwaPliku = core of the filename, without extension
    '''

    headers = ['id', 'x', 'y', 'z']
    atoms = pd.DataFrame(columns = headers)

    for index, molecule in zawartoscKlastraDane.iterrows():
        mol = molecule.molecule
        #print mol

        for a in mol.atoms:
            #print a.idx, a.coords
            atoms.loc[len(atoms)]=[a.idx, a.coords[0], a.coords[1], a.coords[2]]

    averagedMolecule = atoms.groupby(by=['id']).mean()

    for index,atomId in averagedMolecule.iterrows():
        index = int(index)
        a = mol.atoms[index-1].OBAtom
        a.SetVector(atomId[0],atomId[1],atomId[2])

    #mol.write("sdf", nazwaPliku + ".sdf", overwrite=True)

    zawartoscKlastraDaneOut = zawartoscKlastraDane.iloc[0].copy()
    zawartoscKlastraDaneOut['score'] = zawartoscKlastraDane['score'].mean() # the score is average of the scores of the cluster members
    zawartoscKlastraDaneOut['molecule'] = mol

    return zawartoscKlastraDaneOut


def optimize3D(sdffile):
    ### 3D optimization of the averaged molecule (or any other, to be honest...)
    #print os.path.splitext(sdffile)[0]
    #print "%s_localopt.sdf" % (os.path.splitext(sdffile)[0])
    print info.info + "Local 3D optimization..."

    output = pybel.Outputfile("sdf", "%s_localopt.sdf" % (os.path.splitext(sdffile)[0]), overwrite=True )

    for mol in pybel.readfile("sdf", sdffile):

        mol.addh()
        mol.localopt()
        output.write(mol)

    output.close()


##------------------------------------------------------------------------------------------------------------------------ ###


### Group and score

def scoreLoop(infile, outputFilename, models, sdffile, ClusteringMethod, ClusteringFraction, ClusteringCutoff, groupByName = False, E_weight = 0.1):
      '''For each supplied model in models[] loop over data form infile, predict probabilities and save it in outputFilename '''
      printMemInfo()
      print info.info + "Reading statistics file..."
      statComplex = pd.read_csv(infile, delimiter="\t")
      printMemInfo()
      print info.info + "Grouping..."
      statComplexGrouped = statComplex.groupby(['base', 'at2', 'atom_type'])
      del statComplex
      printMemInfo()

      print info.info + "Reading Energy data..."
      E_ligand = pd.read_csv(outputFilename + ".ligand_energy.csv.bz2", delimiter=",")


      for modelName in models:

          print info.info + "Scoring with model: *" + modelName + "* - good."
          outfile = outputFilename + "." + modelName + ".csv"

          ## Do the scoring here
          #output = score(statComplexGrouped, modelName, tempFilename = outputFilename + ".scores.tmp")


          score(statComplexGrouped, modelName, tempFilename = outputFilename + ".scores.tmp")

          # WARNING - tu może zrobić chunks?
          output = pd.read_csv(outputFilename + ".scores.tmp", delimiter=",")

          os.remove(outputFilename + ".scores.tmp") # delete the temp file

          print info.info + "Grouping results..."

          #output.to_csv(outfile + ".tmp", index=False, sep="\t")	# intermediate file save for debugging

          outputScore = output.groupby(['compoundId','compound']).aggregate(np.sum)

          del output
          outputScore['score_RNA-Ligand'] = -1*outputScore['0']	# to have negative score - the lower the better
          outputScore.drop(['0'], inplace = True, axis=1)

          outputScore.reset_index(level=['compoundId', 'compound'], inplace=True)

          outputScore[['interactions_missing', 'interactions_used', 'compoundId']] = outputScore[['interactions_missing', 'interactions_used', 'compoundId']].astype(int) # no of interactions count as integer

          ## here merge with energy terms and calculate energy
          outputScore = pd.merge(outputScore, E_ligand, left_on=['compoundId', 'compound'], right_on=['compoundId', 'compound'],  how='left');

          outputScore['score_ligand'] = E_weight * (outputScore['E_ligand'] - 473.583213)
          # 473.583213 - 3/4 of experminetally determined ligands poses (PDB) has lower energy

          # Why E has positive values: While GAFF, MMFF94 and other force fields attempt to replicate heats of formation, the heat for formation of an organic molecule need not always be negative. (Consider, for example cyclobutane, which is perfectly stable but has a positive DeltaG.) https://www2.chemistry.msu.edu/faculty/reusch/virttxtjml/energy1.htm

          outputScore['score'] = outputScore['score_RNA-Ligand'] + outputScore['score_ligand']
          #print outputScore
          #exit(1)



          if ClusteringMethod != False and ClusteringFraction > 0:
            #outputScore_noClustering = outputScore # keep old - not clusrered data
            outputScore = cluster(outputScore, sdffile, ClusteringMethod, ClusteringFraction, ClusteringCutoff, outputFilename, modelName)

            #outputScore_noClustering[['interactions_missing', 'interactions_used', 'compoundId']] = outputScore_noClustering[['interactions_missing', 'interactions_used', 'compoundId']].astype(int) # no of interactions count as integer
            #print info.info + "Saving results before clustering..."
            #outputScore_noClustering.to_csv(outfile_noClustering, index=False, sep="\t", float_format='%.3f')

          else:
              # save sdf with scores anyway
              saveSdfWithScores(outputScore, sdffile, outputFilename, modelName)
              #print "outputScore\n\n", outputScore

          # WARNING
          # Save n top scoring structures basing on the clustering
          # for clustering the sorting is already done
          # if no clustering there, we have to read/process sdf file from sources, sort and save top n structures.
          #saveTopScoringPoses(outputScore, sdffile, nPoses=3)

          outputScore[['interactions_missing', 'interactions_used', 'compoundId']] = outputScore[['interactions_missing', 'interactions_used', 'compoundId']].astype(int) # no of interactions count as integer

          print info.info + "Saving ..."
          outputScore.to_csv(outfile, index=False, sep="\t", float_format='%.3f')

          print info.ok + "Results successfully saved in", outfile


          if groupByName == True:

              print info.info + "Grouping by compound's name ..."
              df_grouped = outputScore.groupby(['compound']).agg({'score':'min'})
              df_grouped = df_grouped.reset_index()
              df_grouped = df_grouped.rename(columns={'score':'score_min'})
              df = pd.merge(outputScore, df_grouped, how='left', on=['compound'])
              df = df[df['score'] == df['score_min']]
              df.drop(['score_min'], inplace = True, axis=1)

              print info.info + "Saving ..."
              outfile = outputFilename + "." + modelName + ".grouped.csv"
              df.to_csv(outfile, index=False, sep="\t", float_format='%.3f')
              print info.ok + "Results successfully saved in", outfile
              del df_grouped, df

          del outputScore
          printMemInfo()






def score(statComplexGrouped, modelName, tempFilename):
      '''Score dataframe with modelName model'''
      engine = models_engines[modelName]
      modele = modelDir + engine + "/" + modelName + "/"
      model = 'DefaultNorm_NoContext'

      print info.info + "Setup of the scoring engine..."

      if engine == 'scipy':
        print info.info + "Using scikit engine"
        from sklearn.externals import joblib

      elif engine == 'h2o':
        print info.info + "Using H2O engine"
        import h2o
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # start the cluster with: start_h2o.sh
        #  or: java -Xmx4g -jar external_software/h2o-3.9.1.3501/h2o.jar -port 30000 -name cgirls

        try:
            h2o.init(ip=h2o_ip, max_mem_size = 4, port=args.h2oPort, start_h2o=False)

        except ValueError:
            print info.fail + "Can't connect to a H2O server. Please start it (e.g., with start_h2o.sh) and re-run this program."
            exit(2)

        h2o.remove_all()                          #clean slate, in case cluster was already running
        h2o.__PROGRESS_BAR__ = False
        h2o.no_progress()

      #output = ''
      noModelCount = 0 # brak modeli
      groupsCount = 0
      maxGroupCount = len(statComplexGrouped)

      ## Loop over the atoms groups
      print info.info + "Calculating scores ..."
      for group, data in statComplexGrouped:
            groupsCount += 1
            target = "_".join(group)

            data = data.replace('', np.nan)
            data = data.dropna(axis='columns', how='all')

            outputTmp = data.iloc[:,:2].reset_index(drop=True) # compoundId i compound

            data = data.drop(['base','at1','at2','atom_type',  'compoundId', 'compound'], axis=1)	# wywala kolumny które już nie potrzebne

            #WARNING
            '''
            if len(output) == 0:
              output = pd.DataFrame(outputTmp.iloc[0:0,:])	# headers
              print output
              exit(1)
            '''

            if args.beVerbose == True:
              print "[%3i/%i] %s\t" % (groupsCount, maxGroupCount, target),
            else:
              printProgress(groupsCount, maxGroupCount, prefix="[%3i/%i] %s\t" %(groupsCount, maxGroupCount, target))


            if(os.path.isfile(modele + model + "/" + target + ".pkl") ):

                    ## if we should use distance weight to weight score vectors
                    if args.useWeightDistance != False:
                      weightDistanceVector = weightDistance(data.values[:,[1]], function=args.useWeightDistance)

                    ## Let's score!
                    if engine == 'scipy':

                        classifier = joblib.load(modele + model + "/" + target + ".pkl")
                        dist = pd.DataFrame(classifier.predict_proba(data.values)[:,1])	# probabilities

                    elif engine == 'h2o':

                        classifier = h2o.h2o.load_model(modele + model + "/" + target + ".pkl")

                        data.insert(0, "fake", "")	# fake column to delete later as we need columns C2-C6
                        dataH2O = h2o.H2OFrame(data)
                        dataH2O = dataH2O.drop(0)	# remove class column (C1)

                        pred = classifier.predict( dataH2O )
                        dist =  h2o.as_list(pred["p1"])
                        dist.rename(columns={"p1": 0}, inplace=True)

                        h2o.remove_all()

                    ## wszystkie endżiny


                    ##  if we should use distance weight to weight score vectors
                    if args.useWeightDistance != False:
                      dist = dist * weightDistanceVector

                    ## We are limiting distance to this cutoff:
                    if args.useDistanceCutoff != False:
                      dist = dist * distanceCutoff(data.values[:,[1]], cutoff=args.useDistanceCutoff)

                    ## If we are transforming the output probablility somehow

                    if args.doTransformProba != False:
                      print dist
                      dist = transformProba(dist, function = args.doTransformProba)
                      #print dist
                      #exit(1)

                    outputTmp['interactions_missing'] = 0	# add the number of missing models
                    outputTmp['interactions_used'] = 1	# add the number of used models
                    outputTmp = outputTmp.join(dist, how='outer')	# mamy połączone po indeksie

                    if args.beVerbose == True:
                      stdVal = outputTmp[0].std()
                      minVal = outputTmp[0].min()
                      maxVal = outputTmp[0].max()
                      print info.ok + "std: %.2f\tmin: %.2f\t max: %.2f\t records: %i\t mem: %9i kB" % (stdVal, minVal, maxVal, len(data.values), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

            else:
                  if args.beVerbose == True:
                    print info.fail + "No model :-("

                  outputTmp['interactions_missing'] = 1	# add the number of missing models
                  outputTmp['interactions_used'] = 0	# add the number of used models

                  noModelCount += 1


            outputTmp = outputTmp.groupby(['compoundId','compound']).aggregate(np.sum)

            if groupsCount == 1:
              outputTmp.to_csv(tempFilename, sep=",", header=True)
            else:
              outputTmp.to_csv(tempFilename, sep=",", mode='a', header=False)



      # end of for group, data in statComplexGrouped
      print info.info + "Total number of models used:", groupsCount
      noModelCountPrecent = float(noModelCount) / float(groupsCount)
      print info.info + "Missing number of models pairs count:", noModelCount, " =", noModelCountPrecent * 100, "%"




def mergeOutputFiles(outputFilename, models, groupByName = False):
      '''Merge all output scores into a single multicolumn file'''

      print info.info + "Merging files ..."

      merged = ''		# initializing
      mergedGrouped = ''	# initializing

      for modelName in models:
          outfile = outputFilename + "." + modelName + ".csv"
          print info.info + "Processing file:", outfile
          data = pd.read_csv(outfile, delimiter="\t")
          data = data.ix[:, ['compoundId', 'compound', 'score'] ]			# last number - the column with the actual score
          data = data.rename(columns={'score': 'AnnapuRNA Score:' + modelName})

          if len(merged) == 0:		# first pass
            merged = data
          else:
            merged = pd.merge(merged, data, left_on=['compoundId', 'compound'], right_on=['compoundId', 'compound'],  how='outer')

          ## --- if we process also grouped files -----

          if groupByName == True:
                outfileGrouped = outputFilename + "." + modelName + ".grouped.csv"
                print info.info + "Processing file:", outfileGrouped
                dataGrouped = pd.read_csv(outfileGrouped, delimiter="\t")
                dataGrouped = dataGrouped.ix[:, ['compoundId', 'compound', 'score'] ]
                dataGrouped = dataGrouped.reindex_axis(['compound', 'compoundId', 'score'], axis=1)

                dataGrouped = dataGrouped.rename(columns={'score': 'AnnapuRNA:' + modelName, 'compoundId': 'BestPose:' + modelName})

                if len(mergedGrouped) == 0:		# first pass
                  mergedGrouped = dataGrouped
                else:
                  mergedGrouped = pd.merge(mergedGrouped, dataGrouped, left_on=['compound'], right_on=['compound'],  how='outer')


      print info.info + "Saving merged data to file ..."
      merged.to_csv(outputFilename + ".merged.csv", sep="\t", index=False)
      print info.ok + "Merged data saved to", outputFilename + ".merged.csv"

      if groupByName == True:
          #mergedGrouped = mergedGrouped.reindex_axis(sorted(mergedGrouped.columns), axis=1)
          mergedGrouped.to_csv(outputFilename + ".grouped.merged.csv", sep="\t", index=False)
          print info.ok + "Merged grouped data saved to", outputFilename + ".grouped.merged.csv"





#-----------------------------------------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    args = parse_options()

    ## general options check
    if args.useDistanceCutoff != False:
        if args.useDistanceCutoff <= 0 or args.useDistanceCutoff > 10:
          print info.fail + "Wrong distance cutoff set!", args.useDistanceCutoff
          exit(1)

    ## other option check

    ### WARNING BUG SPRAWDZIĆ PO CO TO .predictions.csv WARNING
    if os.path.isfile(args.outputFilename + ".predictions.csv") and args.overwriteResults == False:
        print info.fail + "Output file exists, please rename/remove it, specify other destination file or use --overwrite swich."
        exit(2)
    elif os.path.isfile(args.outputFilename + ".csv.bz2") and os.path.isfile(args.outputFilename + ".ligand_energy.csv.bz2") and args.skipStatistics == True:
        ## If ligand dixed file exists:

        filename, extension = os.path.splitext(args.ligandFile)
        extension = extension.split(".")[-1]
        ligandSdfFileName = "%s.titles.sdf" % (filename)

        if not os.path.isfile(ligandSdfFileName):
            print info.info + "Normalizing the ligand file, adding titles ..."
            ligandSdfFileName = parse_input_ligand_file_to_sdf(args.ligandFile)

        print info.info + "Skipping statistics collecting."



    elif ( os.path.isfile(args.outputFilename + ".csv.bz2") or os.path.isfile(args.outputFilename + ".ligand_energy.csv.bz2") ) and args.skipStatistics != True:
        print info.fail + "Intermediate output file exists, please remove it, specify other destination file or reuse it (-s switch)."
        exit(2)
    else:
        # There is no statistics file, we proceed

        if "/" in args.outputFilename:
            mkdir_p( os.path.dirname( args.outputFilename) )


        if args.skipCleaning == False:
            print info.info + "Cleaning PDB file ..."
            yapdb_parser( args.rnaFile, args.outputFilename + ".RNA.clean.pdb"  )

        if args.skipSimrnaing == False:
            print info.info + "Converting PDB file to the SimRNA representation ..."
            pdb2simrna ( args.outputFilename + ".RNA.clean.pdb", args.outputFilename + ".RNA.clean.simrna.pdb" )


        print info.info + "Checking RNA structure ..."
        checkPdbFile( args.outputFilename + ".RNA.clean.pdb" )

        ## ligands stuff
        print info.info + "Normalizing the ligand file, adding titles ..."
        ligandSdfFileName = parse_input_ligand_file_to_sdf(args.ligandFile)

        print info.info + "Calculating ligands energy values ..."
        calculate_ligands_energy(ligandSdfFileName, args.outputFilename + ".ligand_energy.csv.bz2")

        print info.info + "Trying to convert ligand file to .phar file."
        args.ligandFilePhar = generatePhar(ligandSdfFileName)

        print info.info + "Collecting statistics ..."
        get_statistics( args.outputFilename + ".RNA.clean.simrna.pdb", args.ligandFilePhar, args.outputFilename )


    if 'ALL' in args.modelName:	# if user provided ALL keyword for models:
        print info.info + "Using all available models."
        args.modelName = models_engines.keys()


    scoreLoop(args.outputFilename + ".csv.bz2", args.outputFilename, args.modelName, sdffile=ligandSdfFileName, ClusteringMethod=args.ClusteringMethod, ClusteringFraction=args.ClusteringFraction, ClusteringCutoff=args.ClusteringCutoff,\
      groupByName = args.groupByName, E_weight = args.EnergyWeight)

    if args.mergeOutputs == True:
        mergeOutputFiles( args.outputFilename, args.modelName, groupByName = args.groupByName)
