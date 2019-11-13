import argparse
import ase.io
import glob
import json
import logging
import numpy as np
import os
import sys
import shutil

from ase.optimize import BFGS, FIRE
from ase.constraints import UnitCellFilter
from cStringIO import StringIO
from pprint import pprint

from relax import relax_gb
from slabmaker.slabmaker import build_tilt_sym_gb, build_twist_sym_gb
from slabmaker.gengb_from_quat import QuaternionGB

try:
    from quippy import Atoms, Potential
    from quippy import set_fortran_indexing, fzeros, frange
    from quippy.io import AtomsWriter, AtomsReader, write
except ImportError:
    pass

set_fortran_indexing(False)

class Capturing(list):
    """:class:`Capturing` wraps a function to capture output for redirection.
    """
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = self._stringio = StringIO()
        sys.stderr = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout
        sys.stderr = self._stderr

class ImeallIO(object):
    """:class:`ImeallIO` contains methods for searching the Imeall Directory tree,
    and creating new :class:`imeall.gb_models.GrainBoundary` and :class:`imeall.gb_models.SubGrainBoundary` directories.
    Each :class:`SubGrainBoundary` directory contains supercells of the parent canonical
    :class:`GrainBoundary`.
    """
    def __init__(self):
#IO variables for VASP Calculations.
#Might be better to have a separate VASP OBJECT templated POSCAR, INCAR Files
#then an Espresso Object Template
        self.vasp_template_dir =  '/projects/SiO2_Fracture/iron/vasp_template/'
        self.vasp_dict         = {'kpar'  :32, 'npar':16, 'magmom':3.0, 'n_at':'NOTSET','ediffg': -0.05}
        self.kpt_grid          = {'kx':12, 'ky':12, 'kz':1}
        self.runsh             = {'nodes':512, 'time':360}

    def make_dir(self, target_dir, dir_name):
        """Create :class:`GrainBoundary` directory if it does not exist and
        return the concatenated string name.

        Args:
          target_dir(str): target directory.
          dir_name(str): name of new directory in target directory.

        Returns:
          str: target directory name.
        """
        target_subdir = os.path.join(target_dir, dir_name)
        if not os.path.isdir(target_subdir):
            os.makedirs(target_subdir)
            print 'Created {0}'.format(target_subdir)
        else:
            print '\t directory already exists'
        return target_subdir

    def load_json(self, json_file):
        """Helper function to return gb_data as dict.

        Args:
          json_file(str): name of gb.json or subgb.json file.

        Returns:
          dict: grain boundary data dictionary.
        """

        with open(json_file, 'r') as datfile:
            gb_data = json.load(datfile)
        return gb_data

    def find_subdir(self, target_dir, calc_suffix):
        """
        Find if named directory is in target directory.
        Args:
          target_dir(str): directory to search.
          calc_suffix(str): name of directory to look for.

        Returns:
          directory location and name.
        """
        for _dir in os.listdir(target_dir):
            if calc_suffix in _dir:
                from_dir = os.path.join(target_dir, _dir)
                from_dir_name = _dir
                return from_dir, from_dir_name
        print 'no directory with calc_suffix,', calc_suffix
        return None

class GBRelax(object):
    def __init__(self, grain_dir='./', gbid='0000000000', calc_type='EAM',
                 potential = 'IP EAM_ErcolAd', param_file = 'iron_mish.xml',
                 traj_file='traj.xyz'):
        """:class:`GBRelax` is responsible for generating the
        initial configuration of the :class:`SubGrainBoundary` before relaxation occurs.

        Args:
          grain_dir (str, optional): root directory to build grain boundary tree.
          gbid (str,optional): gbid of grain boundary root.
          calc_type(str,optional): Type of calculation: EAM, DFT, TB, GAP.
          potential(str,optional): String specifying the :class:`quippy.potential` type.
          param_file(str,optional): Name of interatomic potential file.
          traj_file(str, optional): Name of target structure file.
        """

        self.gbid        =  gbid
        self.grain_dir    = grain_dir
        self.calc_dir   = os.path.join(grain_dir, calc_type)
        self.subgrain_dir =''
        self.name        = 'subgrain'
        self.calc_type   = calc_type
        self.struct_file = ''
        self.param_file  = param_file
        self.potential   = potential
        self.fmax        = 0.5E-2
        self.traj_file   = traj_file
        print 'Canonical Grain Directory:'
        print '\t', self.grain_dir
        print 'Generate Job in:'
        print '\t', self.calc_dir

    def gen_super_rbt(self, bp=[],v=[], angle=None, rbt=[0.0, 0.0], sup_v=6, sup_bxv=2, rcut=2.0, gb_type="tilt"):
        """
        Create a :class:`SubGrainBoundary` supercell with rigid body translations (rbt) 
        and a particular atom deletion criterion 2.0.

        Args:
          bp (list): Boundary plane normal.
          v (list): Perpendicular vector to the boundary plane normal.
          rbt (list): rigid body translation as fractional translations of the supercell.
          sup_v(int): Size of supercell along v.
          sup_bxv(int): Size of supercell along boundary_plane_normal crossed with v.
          rcut(float): Atom deletion criterion angstrom.
          gb_type(str): Determine whether to generate a tilt or twist boundary.

        Returns:
          None
        """
        io = ImeallIO()
        if gb_type=="tilt":
            grain = build_tilt_sym_gb(bp=bp, v=v, rbt=rbt)
            logging.debug('bp: {} v: {} rbt: {}'.format(bp, v, rbt))
        elif gb_type=="twist":
            #requires latt_vec from angle to orient crystal.
            quatgb = QuaternionGB()
            angle_list = quatgb.gen_sym_twist()
            v = filter(lambda x: x[0] == round(180.*angle/np.pi, 2), angle_list)
            print 'V', v[0]
            grain = build_twist_sym_gb(bp=bp, v=v[0][1], rbt=rbt)
        #For RBT we build a top level dir with just the translated supercell and no deletion criterion
        m, n, grain  = self.gen_super(grain=grain, rbt=rbt, sup_v=sup_v, sup_bxv=sup_bxv,  rcut=0.0)
        self.name    = '{0}_v{1}bxv{2}_tv{3}bxv{4}'.format(self.gbid,
        str(m), str(n), str(round(rbt[0],2)), str(round(rbt[1],2)))
        self.subgrain_dir = io.make_dir(self.calc_dir, self.name)
        grain.write('{0}.xyz'.format(os.path.join(self.subgrain_dir, self.name)))
        self.name    = '{0}_v{1}bxv{2}_tv{3}bxv{4}_d{5}z'.format(self.gbid,
        str(sup_v), str(sup_bxv), str(round(rbt[0],2)), str(round(rbt[1],2)), str(rcut))
        self.subgrain_dir = io.make_dir(self.subgrain_dir, self.name)
        print "delete atoms"
        grain = self.delete_atoms(grain=grain, rcut=rcut)
        grain.write('{0}.xyz'.format(os.path.join(self.subgrain_dir, self.name)))
        #write json file with subgb information.
        try:
            f = open('{0}/subgb.json'.format(self.subgrain_dir), 'r')
            j_dict = json.load(f)
            f.close()
        except IOError:
            f = open('{0}/subgb.json'.format(self.subgrain_dir), 'w')
            j_dict = {}
        #Terms to append to subgrain dictionary:
        cell           = grain.get_cell()
        cell_area      = cell[0,0]*cell[1,1]
        cell_height    = cell[2,2]
        j_dict['param_file'] = self.param_file
        j_dict['potential']  = self.param_file
        j_dict['name'] = self.name
        j_dict['rbt']  = rbt
        j_dict['rcut'] = rcut
        j_dict['H']    = cell_height
        j_dict['A']    = cell_area
        j_dict['converged'] = False
        j_dict['area']      = cell_area
        j_dict['n_at']      = len(grain)
        f = open('{0}/subgb.json'.format(self.subgrain_dir), 'w')
        json.dump(j_dict, f, indent=2)
        f.close()

    def gen_super(self, grain=None, rbt=None, sup_v=6, sup_bxv=2, rcut=2.0):
        """ :method:`gen_super` Creates a :class:SubGrainBoundary super cell according to
        conventions described in Rittner and Seidman (PRB 54 6999).

        Args:
          grain(:class:`ase.Atoms`): atoms object passed from gen_super_rbt.
          rbt (list): rigid body translation as fractional translations of the supercell.
          sup_v(int): Size of supercell along v.
          sup_bxv(int): Size of supercell along boundary_plane_normal crossed with v.
          rcut(float): Atom deletion criterion in angstrom.
        """
        io = ImeallIO()
        if rbt == None:
            x = Atoms('{0}.xyz'.format(os.path.join(self.grain_dir, self.gbid)))
        else:
            x = Atoms(grain)

        struct_dir = os.path.join(self.grain_dir, 'structs')
        self.name = '{0}_v{1}bxv{2}_tv{3}bxv{4}_d{5}z'.format(self.gbid,
        str(sup_v), str(sup_bxv), '0.0', '0.0', str(rcut))
#TODO fix this so it is more transparent.
#if rcut = 0 then only a super cell is generated with no deletion of atoms.
        if rcut > 0.0:
            x.set_cutoff(2.4)
            x.calc_connect()
            x.calc_dists()
            rem = []
            u = np.zeros(3)
            for i in range(x.n):
                for n in range(x.n_neighbours(i)):
                    j = x.neighbour(i, n, distance=3.0, diff=u)
                    if x.distance_min_image(i,j) < rcut and j != i:
                        rem.append(sorted([j,i]))
            rem = list(set([a[0] for a in rem]))
            if len(rem) > 0:
                x.remove_atoms(rem)
            else:
                print 'No duplicate atoms in list.'
        else:
            x = x*(sup_v, sup_bxv, 1)
            x.set_scaled_positions(x.get_scaled_positions())

        if rbt == None:
            self.struct_file = self.name
            self.subgrain_dir = io.make_dir(self.calc_dir, self.name)
            try:
                with open('{0}/subgb.json'.format(self.subgrain_dir), 'r') as f:
                    j_dict = json.load(f)
            except IOError:
                j_dict             = {}

            j_dict['name']       = self.name
            j_dict['param_file'] = self.param_file
            j_dict['rbt']        = [0.0, 0.0]
            j_dict['rcut']       = rcut

            with open('{0}/subgb.json'.format(self.subgrain_dir), 'w') as f:
                json.dump(j_dict, f, indent=2)
        else:
            return sup_v, sup_bxv, x

    def delete_atoms(self, grain=None, rcut=2.0):
        """
        Delete atoms below a certain distance threshold.

        Args:
          grain(:class:`quippy.Atoms`): Atoms object of the grain.
          rcut(float): Atom deletion criterion.

        Returns:
          :class:`quippy.Atoms` object with atoms nearer than deletion criterion removed.
        """
        io = ImeallIO()
        if grain == None:
            x = Atoms('{0}.xyz'.format(os.path.join(self.grain_dir, self.gbid)))
        else:
            x = Atoms(grain)
        x.set_cutoff(2.4)
        x.calc_connect()
        x.calc_dists()
        rem=[]
        u=fzeros(3)
        for i in frange(x.n):
            for n in frange(x.n_neighbours(i)):
                j = x.neighbour(i, n, distance=3.0, diff=u)
                if x.distance_min_image(i, j) < rcut and j!=i:
                    rem.append(sorted([j,i]))
        rem = list(set([a[0] for a in rem]))
        if len(rem) > 0:
            x.remove_atoms(rem)
        else:
            print 'No duplicate atoms in list.'
        if grain == None:
            self.name    = '{0}_d{1}'.format(self.gbid, str(rcut))
            self.subgrain_dir = io.make_dir(self.calc_dir, self.name)
            self.struct_file  = gbid + '_' + 'n' + str(len(rem)) + 'd' + str(rcut)
            x.write('{0}.xyz'.format(os.path.join(self.subgrain_dir, self.struct_file)))
            return len(rem)
        else:
            return x

    def gen_pbs(self, time='02:30:00', queue='serial.q', template_str='/users/k1511981/pymodules/templates/calc_ada.pbs'):
        """
        :method:`gen_pbs` generates job pbs file.

        Args:
          time(str): length of job time in string format.
          queue(str): name of queue to generate submission script for
          template_str(str): Name of location for pbs template file (example templates in imeall directory).
        """
        pbs_str = open(template_str, 'r').read()
        pbs_str = pbs_str.format(jname='fe'+self.name, xyz_file='{0}.xyz'.format(self.name),
                                 time=time, queue=queue)
        print os.path.join(self.subgrain_dir, 'fe{0}.pbs'.format(self.name))
        with open(os.path.join(self.subgrain_dir, 'fe{0}.pbs'.format(self.name)) ,'w') as pbs_file:
            print >> pbs_file, pbs_str

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prefix", help="Subsequent commands will act on all \
                                                  subdirectories with first characters matching prefix.", default='001')
    parser.add_argument("-ct", "--calc_type", help="Name of calculation type TB, EAM, DFT, etc.", default='PotBH')
    parser.add_argument("-q",  "--queue",     help="Jobs will be submitted to this queue.", default='smp.q')
    parser.add_argument("-t",  "--time",      help="Time limit on jobs.", default='1:00:00')
    parser.add_argument("-hyd", "--hydrogen", type = int, help="If greater than 0 add n hydrogens to the boundary.", default=0)
    parser.add_argument("-rc",  "--rcut",     type = float, help="Deletion criterion for nearest neighbour atoms.", default=2.0)
    parser.add_argument("-i_v",   "--i_v",    type = float, help="Rigid body translation along i_v.", default=0.0)
    parser.add_argument("-i_bxv", "--i_bxv",  type = float, help="Rigid body translation along i_bxv.", default=0.0)
    parser.add_argument("-gbt",   "--gb_type", help="Specify type of boundary twist or tilt.", default="tilt")
    parser.add_argument("-fs","--from_script", help="Whether this pyscript is run from a script.", action="store_false", default=True)

    args = parser.parse_args()
    prefix = args.prefix
    queue = args.queue
    time = args.time
    rcut = float(args.rcut)

# Each calculation type is associated with a potential:
    if args.calc_type == 'EAM_Mish':
        param_file = 'iron_mish.xml'
    elif args.calc_type == 'PotBH':
        param_file = 'PotBH.xml'
    elif args.calc_type == 'EAM_Men':
        param_file = 'Fe_Mendelev.xml'
    elif args.calc_type == 'EAM_Ack':
        param_file = 'Fe_Ackland.xml'
    elif args.calc_type == 'EAM_Dud':
        param_file = 'Fe_Dudarev.xml'
    elif args.calc_type == 'GAP':
        param_file = 'gp33b.xml'
    else:
        print 'No available potential corresponds to this calculation type.'
        sys.exit()

    if not args.from_script:
        jobdirs = []
        for target_dir in os.listdir('./'):
            if os.path.isdir(target_dir) and thing[:len(prefix)]==prefix:
                jobdirs.append(target_dir)

        for job_dir in jobdirs[:]:
            gbid    = job_dir.strip('/')
            print '\n'
            print '\t', gbid
            print '\n'
            gbrelax = GBRelax(grain_dir=job_dir, gbid=gbid, calc_type=args.calc_type,
                              potential='IP EAM_ErcolAd', param_file=param_file)

            if args.gb_type=="twist":
                sup_v   = 3
                sup_bxv = 3
            elif args.gb_type=="tilt":
                sup_v   = 6
                sup_bxv = 2

            with open(os.path.join(job_dir,'gb.json')) as f:
                grain_dict = json.load(f)

            bp = grain_dict['boundary_plane']
            v = grain_dict['orientation_axis']
            angle = grain_dict['angle']
            for i in np.linspace(0.125, 1.0):
                for j in np.linspace(0.125, 1.0):
                    gbrelax.gen_super_rbt(rcut=rcut, bp=bp, v=v, angle=angle, sup_v=sup_v, sup_bxv=sup_bxv, rbt=[i,j])
                    gbrelax.gen_pbs(time=time, queue=queue)
    elif args.from_script:
        job_dir = os.getcwd()
        print job_dir
        with open(os.path.join(job_dir,'gb.json')) as f:
            grain_dict = json.load(f)
        gbid = grain_dict['gbid']
        bp = grain_dict['boundary_plane']
        v = grain_dict['orientation_axis']
        angle = grain_dict['angle']
        gbrelax = GBRelax(grain_dir=job_dir, gbid=gbid, calc_type=args.calc_type,
                          potential = 'IP EAM_ErcolAd', param_file=param_file)
        if args.gb_type=="twist":
            sup_v = 3
            sup_bxv = 3
        elif args.gb_type=="tilt":
            sup_v = 6
            sup_bxv = 2
        print "Boundary Plane", bp, "Orientation Axis", v
        i_v = float(args.i_v)
        i_bxv = float(args.i_bxv)
# Generate the appropriate grain boundary in this directory.
        gbrelax.gen_super_rbt(rcut=rcut, bp=bp, v=v, sup_v=sup_v, angle=angle, sup_bxv=sup_bxv, rbt=[i_v, i_bxv], gb_type=args.gb_type)
# Switch to the appropriate subgrain directory.
        os.chdir(gbrelax.subgrain_dir)
# Call the relax function from this directory, reads in the initial struct_file,
        relax_gb(gb_file = gbrelax.name)
