import os
import sys
import argparse
import glob
import json
import logging
import numpy as np
import re
import shutil
import slabmaker.slabmaker as slabmaker

from  scipy.spatial import Voronoi, voronoi_plot_2d

try:
    from  imeall import app
except ImportError:
    print 'No app'

try:
    from flask  import Flask, request, session, g, redirect
    from flask  import url_for, abort, render_template, flash
except ImportError:
    print 'No Flask Module Available'

# Currently Our models are stored by hand
# and then we handle the interactions with
# directory structure manually. Which should allow us to serve
# files.

class PotentialParameters(object):
    """Contains parameter dictionaries for basic properties associated with
    interatomic potentials in the database.

    The ground state energy per atom for bulk iron
    for the different inter-atomic potentials used in the Imeall Database.
    It also contains the values required to rescale the relaxed lattice parameters
    for bulk iron predicted by the inter-atomic potentials
    to the target DFT value.
    """

    def __init__(self):
        self.name = 'Potential Parameters'

    def gs_ener_per_atom(self):
        """
        Returns:
          eperat(dict): {'parameter file': energy per atom (eV)}
        """
        eperat = {'Fe_Mendelev.xml' : -4.12243503431,
                  'PotBH.xml'       : -4.01298214176,
                  'iron_mish.xml'   : -4.28000356875,
                  'Fe_Ackland.xml'  : -4.01298226805,
                  'Fe_Dudarev.xml'  : -4.31608690638,
                  'dft_vasp_pbe'    : -8.238035,
                  'gp33b.xml'       : -3460.93341688
                 }
        return eperat

    def eam_rscale(self):
        """Dictionary of rescaling parameters of EAM to approximate pbe DFT value of 2.83.

        Returns:
          rscale:{parameter file: rscale parameter}
        """

        rscale = {'Fe_Mendelev.xml' : 1.00894848312,
                  'PotBH.xml'       : 1.00894848312,
                  'iron_mish.xml'   : 1.0129007626,
                  'Fe_Ackland.xml'  : 1.00894185389,
                  'Fe_Dudarev.xml'  : 1.01279093417,
                  'dft_vasp_pbe'    : 1.00000000000,
                  'gp33b.xml'       : 1.0015226318
                  }
        return rscale

    def paramfile_dict(self):
        """Dictionary Mapping directory names to quip xml files.

        Returns:
          dict: {potential_directory:potential_file}
        """

        paramfile      = {'DFT':'dft_vasp_pbe',
                          'PotBH':'PotBH.xml',
                          'EAM_Ack':'Fe_Ackland.xml',
                          'EAM_Men':'Fe_Mendelev.xml',
                          'EAM_Mish':'iron_mish.xml',
                          'EAM_Dud':'Fe_Dudarev.xml',
                          'GAP':'gp33b.xml'}
        return paramfile

    def potdir_dict(self):
        """Invert keys from :py:func:`paramfile_dict` to give potential file to
        directory mapping.

        Returns:
          dict: {potential_file:potential_directory}
        """

        paramfile_dict = self.paramfile_dict()
        potdir = {}
        for k,v in paramfile_dict.items():
            potdir[v] = k
        return potdir

    def calc_e_gb(self, at, E_bulk):
        """Calculate grain boundary energy relative to bulk value.

        Returns:
          float: E_gb grain boundary energy J/m^{2}
        """

        cell = at.get_cell()
        A    = cell[0,0]*cell[1,1]
        E_gb = (at.get_potential_energy()-(at.n*(E_bulk)))/(2.*A)
        at.get_potential_energies()
        E_gb = 16.02*(at.get_potential_energy()-(at.n*(E_bulk)))/(2.*A)
        return E_gb

class GBQuery(object):
    """
    Routines for grabbing desired .xyz files and paths.
    """

    def __init__(self):
        self.__repr__=="FetchStructure"

    def copy_gb_dir(self, gd, target_dir='./'):
        """
        Given a :py:class:`SubGrainBoundary` model as a dictionary copy
        the directory
        """
        gbid = gd['gbid']
        new_dir_name = gbid.split('_')[0]
        new_dir_name = os.path.join(target_dir, new_dir_name)
        try:
            os.mkdir(new_dir_name)
        except OSError:
            print 'Directory already exists.'
        struct_name = gbid+'_traj.xyz'
        dir_path = gd['path']
    #grab the gb.json file path.
        gb_path = gd['path']
        gb_path = '/'.join(gb_path.split('/')[0:3])+'/gb.json'
        print gb_path
        gb_path = os.path.join(app.config['GRAIN_DATABASE'], gb_path)
    #grab the struct file and the subgb.json path.
        dir_path = os.path.join(app.config['GRAIN_DATABASE'], dir_path)
        struct_path = os.path.join(dir_path, struct_name)
        subgb_path = os.path.join(dir_path, 'subgb.json')
        shutil.copy(struct_path, new_dir_name)
        shutil.copy(subgb_path, new_dir_name)
        shutil.copy(gb_path, new_dir_name)

    def copy_gb_dirtree(self, material="alphaFe", or_axis="0,0,1", pots=['PotBH.xml'],
                        target_dir='./', gb_type='tilt'):
        """Pull all minimum energy structures, using :py:func:`pull_minen_structs`,
        from database for an orientation axis and copy them to target_dir. Useful for checking out
        structures into a unique directory for specialized analysis.

        Args:
          material(str): material name
          or_axis(str): csv serialized vector of orientation axis
          pots: list of potential parameter files.
          target_dir: Location to copy files to  (Default: './').
          gb_type(str): Options 'tilt' or 'twist'.

        Returns:
          list[:py:class:`SubGrainBoundary`]: List of :py:class:`SubGrainBoundary` :py:class:`Model`
          represented as dictionaries.

        Todo:
          * Add kwargs to pull_minen_structs to provide additional selection criteria.
        """

        grain_dicts = self.pull_minen_structs(material=material, or_axis=or_axis, pots=pots, gb_type=gb_type)
        for gd in grain_dicts:
            self.copy_gb_dir(gd, target_dir=target_dir)
        return grain_dicts

    def pull_minen_structs(self, material="alphaFe", or_axis="1,1,1", pots=['PotBH.xml'], gb_type='tilt'):
        """Grab the minimum energy structure json dictionaries
        for a given material, orientation_axis, and potential(s) parameter filenames.

        Args:
          material(str,optional): Material to investigate.
          or_axis(str,optional): Orientation axis "1,1,1".
          pots(list): list of potentials parameter files.
          gb_type(str): Options 'tilt' or 'twist'.

        Returns:
          list[:py:class:`SubGrainBoundary`]: List of :py:class:`SubGrainBoundary` :py:class:`Model`
          represented as dictionaries.
        """

        from gb_models import database, GrainBoundary, SubGrainBoundary
        from collections import OrderedDict

        database.connect()
        pot_param     = PotentialParameters()
        ener_per_atom = pot_param.gs_ener_per_atom()

        if gb_type=='tilt':
            gbs = (GrainBoundary.select()
                              .where(GrainBoundary.orientation_axis == or_axis)
                              .where(GrainBoundary.boundary_plane != or_axis))
        elif gb_type=='twist':
            gbs = (GrainBoundary.select()
                              .where(GrainBoundary.orientation_axis == or_axis)
                              .where(GrainBoundary.boundary_plane == or_axis))
        else:
            sys.exit("Unsupported gb_type. Options:'tilt' or 'twist'")

        dict_list = []
        for gb in gbs.order_by(GrainBoundary.angle):
            pot_dict = OrderedDict({})
            for potential in pots:
                subgbs = (gb.subgrains.select(GrainBoundary, SubGrainBoundary)
                              .where(SubGrainBoundary.potential==potential)
                              .join(GrainBoundary)
                              .order_by(SubGrainBoundary.E_gb)
                              .dicts())
                subgbs = [(16.02*(subgb['E_gb']-float(subgb['n_at']*ener_per_atom[potential]))/
                          (2.0*subgb['area']), subgb) for subgb in subgbs]
                subgbs.sort(key = lambda x: x[0])
                try:
                    if subgbs[0][1]['converged'] == True:
                        pot_dict[potential] = subgbs[0][0]
                        dict_list.append(subgbs[0][1])
                except IndexError:
                    print 'No subgbs for: ', gb.gbid, potential
                #print '{:.3f}'.format(180.0/np.pi*gb.angle), ' '.join(['{:.3f}'.format(x) for x in pot_dict.values()])
        database.close()
        return dict_list

class Job(object):
    """Collection of routines for generating and submitting pbs submission scripts.
    """

    def __init__(self):
        self.pbs_file = ''
        self.job_dir  = ''
        self.job_id   = ''

    def sub_pbs(self, job_dir, exclude='DFT', suffix='v6bxv2z', regex=None):
        """Given an explicit suffix, or a regex this routine recurses through
        the directory structure and change directory to location of pbs file and
        submits pbs files that match the suffix or regex pattern.
        Exclude avoids certain directories.

        Useful submission patterns include:
        REGEX:
        Submit all pbs files with a rigid body translation
        and any atom deletion criterion hydrogen concentration etc.: tv[.0-9]+bxv[.0-9]+_.*?

        Submit all sub-pbs files with a rigid body translation
        and different atom _deletion criterion: tv[.0-9]+bxv[.0-9]+_d[.0-9]+

        All translations with a specific deletion criterion in this case 2.3 A: tv[.0-9]+bxv[.0-9]+_d2.3 etc.
        SUFFIX:
        submit all super cells: v6bxv2z

        Args:
          job_dir(str): root directory to walk downwards from.
          exclude(str): directory names to exclude.
          suffix(str): pattern that job file must contain to be submitted.
          regex(str): regex pattern.
        """

        lst = os.listdir(job_dir)
        for target_dir in lst:
            target_dir = os.path.join(job_dir, target_dir)
            if regex == None:
                if os.path.isdir(target_dir) and target_dir != 'DFT':
                    self.sub_pbs(dir, suffix=suffix, regex=regex)
                elif target_dir.split('_')[-1] == suffix:
                    pbs_dir = os.path.join(sub_dir, target_dir)
                    os.system("cd {0}; qsub fe{1}.pbs".format(pbs_dir, job_dir+'_'+suffix))
                else:
                    pass
            else:
                if os.path.isdir(target_dir) and target_dir != 'DFT':
                    self.sub_pbs(target_dir, suffix=suffix, regex=regex)
                elif regex.match(target_dir):
                    try:
                        target_dir  = '/'.join(target_dir.split('/')[:-1])
                        name = target_dir.split('/')[-1]
                        os.system("cd {0}; qsub fe{1}.pbs".format(target_dir, name))
                    except:
                        print 'Job Submit Failed'
                else:
                    pass

class GBMaintenance(object):
    """Collection of maintenance routines for the GB database.
    Methods include routines to regenerate all the csl lattices in the database,
    take a new grain boundary profile picture for multiple directories,
    update the gb.json information if a new grain boundary property is desired.
    """

    def __init__(self):
        self.materials = ['alphaFe']

    def retake_pic(self, fname, translate=False, toggle=False, confirm=True):
        """DEPRECATED If using AtomEye take grain boundary profile pic in directory
        requires gb directory with gbid.xyz file in it. Set confirm=False
        to not prompt for overwrite. In future versions OVITO will be default image generator.
        """

        if confirm:
            var = 'n'
            var = raw_input('Retake photo (y/n)?')
        else:
            var = 'y'
        if var =='y':
            fname = os.path.join(fname,fname)
            slabmaker.take_pic(fname, translate=translate, toggle=toggle)
            print 'retaking photo'
        elif var =='n':
            pass

    def remove_eo_files(self, path, num_deleted_files, dryrun=False):
        """
        Remove files with pattern matching jobname.[eo][0-9]+.

        Args:
          path(str): root path to start walking downwards from.
          num_deleted_files(int): Number of job files deleted.
          dryrun(bool): (Default: True).

        Returns:
          Number of deleted files.
        """

        eo_regex = re.compile(r'[eo][0-9]+')
        lst = os.listdir(path)
        for filename in lst:
            filename = os.path.join(path, filename)
            if os.path.isdir(filename):
                if dryrun:
                    rec_deleted_files = self.remove_eo_files(filename, 0)
                num_deleted_files += rec_deleted_files
            elif eo_regex.match(filename.split('.')[-1]):
                print filename
                os.remove(filename)
                num_deleted_files += 1
            else:
                pass
        return num_deleted_files

    def add_key_to_dict(self, dirname, new_key, new_value, dict_type='subgb.json', modify=False):
        """Add single key to .json dictionary.

        Args:
          dirname(str): directory name with json dictionary.
          new_key(str): key to add to dictionary.
          new_value(generic): value to add to dictionary.
          dict_type(str, optional): Type of json dict 'subgb.json' or 'gb.json'.
          modify(bool): If True json file will be overwritten.

        Returns:
          dict: Updated json dictionary.
        """

        json_path = os.path.join(dirname, dict_type)
        new_json = {}
        with open(json_path,'r') as json_old:
            old_json = json.load(json_old)
        for key in old_json.keys():
            new_json[key] = old_json[key]
        #modify key or add it
        new_json[new_key] = new_value
        if modify:
            with open(json_path,'w') as json_new_file:
                json.dump(new_json, json_new_file, indent=2)
        return new_json

    def update_json(self, json_path, new_keys=[], new_values=[], dryrun=True):
        """Update or add multiple keys to json dictionaries.

        Args:
          json_path(str): directory name where gb.json file needs to be updated.
          new_keys(list): list of new keys.
          new_values(list): list of new values
          dryrun(bool):If False json file is over written with new keys.

        Returns:
          dict: Updated json dictionary.

        """

        assert len(new_keys)==len(new_values)
        assert type(new_keys)==list
        assert type(new_values)==list
        with open(json_path,'r') as json_old:
            old_json = json.load(json_old)
        new_json = old_json
        for key, value in zip(new_keys, new_values):
            new_json[key] = value
        if dryrun:
            print new_json
        else:
            with open(json_path,'w') as json_new_file:
                json.dump(new_json, json_new_file, indent=2)
        return new_json

    def fix_json(self, path):
        """If json file is corrupted and contains two {}{} dictionaries,
        this function selects the second of the two dictionaries and overwrites json file.

        Args:
          path(str): path to corrupted dictionary.
        """

        lst = os.listdir(path)
        for filename in lst:
            new_path = os.path.join(path, filename)
            if os.path.isdir(new_path):
                self.fix_json(new_path)
            elif new_path[-10:] == 'subgb.json':
                try:
                    with open(new_path,'r') as f:
                        j_file = json.load(f)
                except ValueError:
                    print 'Value Error', new_path
                    with open(new_path,'r') as f:
                        j_file = f.read()
                    with open(new_path,'w') as f:
                        print >> f, j_file.split('}')[1]+'}'
                try:
                    with open(new_path,'r') as f:
                        j_file = json.load(f)
                    print 'j_file fixed'
                except:
                    print new_path, 'Still Broken'
            else:
                pass

class GBAnalysis(object):
    """Contains methods for pulling structural.
    energetic, and meta data from the grain boundary data tree.
    Also routines for identifying unconverged structural relaxations.
    """

    def __init__(self):
        pass

    def find_gb_json(self, path, j_list, filetype):
        """ Populates the list j_list with lists of the form [/directory_path/, /subgb_file_path].
        Args:
          path     : root directory to begin recursive search
          j_list   : empty list  to populate
          filetype : 'subgb.json', 'gb.json'

        Returns:
          None
        """

        lst = os.listdir(path)
        for filename in lst:
            filename = os.path.join(path, filename)
            if os.path.isdir(filename):
                self.find_gb_json(filename, j_list, filetype)
            elif filename.split('/')[-1] == filetype:
                j_list.append([path,filename])
            else:
                pass

    def extract_energies(self, material='alphaFe', or_axis='001', gb_type='tilt'):
        """Pull GB formation energies using recursion from json files.
        Go into a grain boundary directory, recurse through directory
        to find subgb.json files and gb_energies and pull them out.

        Args:
          material(str): material directory to pull from.
          or_axis(str): orientation axis.
          gb_type(str): tilt or twist boundary.

        Returns:
          list: A list of dictionaries. Each dictionary contains keys
          gbid, orientation_axis, angle, boundary_plane, param_file,
          and energies.
        """

        pot_param     = PotentialParameters()
        ener_per_atom = pot_param.gs_ener_per_atom()
        gb_files = []
#dir str lets us point to whatever material and orientation axisi it is we want in the database.
        dir_str  = os.path.join(material, or_axis)
        self.find_gb_json('{0}'.format(os.path.join(app.config['GRAIN_DATABASE'], dir_str)), gb_files, 'gb.json')
        grain_energies = []
        for grain in  gb_files:
            path = grain[0]
            with open(grain[1],'r') as f:
                j_dict    = json.load(f)
            subgb_files = []
            self.find_gb_json(path, subgb_files, 'subgb.json')
            calc_types  = []
#Find all calculation types associated with this grain
            for subgrain in subgb_files:
                with open(subgrain[1],'r') as f:
                    try:
                        sub_dict = json.load(f)
                    except:
                        pass
                try:
                    if sub_dict['param_file'] not in calc_types:
                        calc_types.append(sub_dict['param_file'])
                except KeyError:
                    pass
#Initialize a dictionary of dictionaries for each calc type:
            gb_dict = {}
            for calc in calc_types:
                tmp_dict = {}
                tmp_dict['orientation_axis'] = j_dict['orientation_axis']
                tmp_dict['angle']            = j_dict['angle']*(180./np.pi)
                tmp_dict['boundary_plane']   = j_dict['boundary_plane']
                tmp_dict['energies']         = []
                tmp_dict['param_file']       = calc
                tmp_dict['gbid']             = j_dict['gbid']
                gb_dict[calc] = tmp_dict

            for subgrain in subgb_files:
                with open(subgrain[1],'r') as f:
                    try:
                        sub_dict = json.load(f)
                        try:
                            if (sub_dict['param_file'] in ener_per_atom.keys()):
                                append_energy = True
                                gb_ener = 16.02*((sub_dict['E_gb']-(ener_per_atom[sub_dict['param_file']]*float(sub_dict['n_at'])))/(2*sub_dict['A']))
                            else:
                                append_energy = False
                                print 'Ground state energy not know for this potential!'
                            if append_energy == True:
                                gb_dict[sub_dict['param_file']]['energies'].append(gb_ener)
                        except KeyError:
                            pass
                    except:
                        print 'Missing json file.'
                        pass
            for gdict in gb_dict.values():
                grain_energies.append(gdict)
        return grain_energies

    def calc_energy(self, gb_dict, param_file='PotBH.xml'):
        """Given a subgb.json dictionary, and a potential
        calculate grainboundary energy.

        Args:
          gb_dict(dict): sub grain dictionary.
          param_file(str): Inter-atomic potential filename.

        Returns:
          float: Grain boundary energy in J/m^{-2}.
        """

        pot_param     = PotentialParameters()
        ener_per_atom = pot_param.gs_ener_per_atom()
        try:
            gb_ener = 16.02*((gb_dict['E_gb']-(ener_per_atom[param_file]*float(gb_dict['n_at'])))/(2*gb_dict['A']))
        except KeyError:
            return None
        else:
            return gb_ener

    def pull_gamsurf(self, path="./",  potential="PotBH"):
        """Loop over subgrain directories of a potential
        to find the minimum and maximum subgrain energies for the canonical grain, return
        a dictionary, with information about the lowest and highest energy structures, and the
        directory they are contained in.

        Args:
          path(str): File path to begin search for subgb.json files.
          potential(str): Name of potential directory.

        Returns:
          dictionary: {'max_en':float, 'min_en':float,'min_coords':list,'max_coords':list, 'path':str}
        """

        potparams = PotentialParameters()
        paramfile_dict = potparams.paramfile_dict()
        subgb_files = []
        if os.path.isdir(os.path.join(path, potential)):
            self.find_gb_json(os.path.join(path, potential), subgb_files, 'subgb.json')
            gam_surfs   = []
            unconv      = []
#Only pulling for PotBH:
            for gb in subgb_files:
                with open(gb[1],'r') as f:
                    gb_json = json.load(f)
                ener = self.calc_energy(gb_json, param_file=paramfile_dict[potential])
                if ener != None:
                    try:
                        gam_surfs.append((gb_json['rcut'], gb_json['rbt'][0], gb_json['rbt'][1], ener, gb[0], gb_json['gbid']))
                    except KeyError:
                        gam_surfs.append((gb_json['rcut'], gb_json['rbt'][0], gb_json['rbt'][1], ener, gb[0], gb_json['name']))
                else:
                    unconv.append(gb[1])
            en_list     = [x[3] for x in gam_surfs]
            try:
                min_en      = min(en_list)
            except ValueError:
                return {'max_en':0.0, 'min_en':0.0, 'min_coords':[], 'max_coords':[], 'min_path':'', 'max_path':''}
            else:
#Create lists of minimum energy structures (vx bxv rcut).
                max_en = max(en_list)
                min_coords = [(gam[1], gam[2], gam[0]) for gam in filter(lambda x: round(x[3], 5) == round(min_en, 5), gam_surfs)]
                max_coords = [(gam[1], gam[2], gam[0]) for gam in filter(lambda x: round(x[3], 5) == round(max_en, 5), gam_surfs)]
                min_path = [(gam[4], gam[5]) for gam in filter(lambda x: round(x[3], 5) == round(min_en, 5), gam_surfs)]
                max_path = [(gam[4], gam[5]) for gam in filter(lambda x: round(x[3], 5) == round(max_en, 5), gam_surfs)]
                min_path = '/'.join(min_path[0])+'_traj.xyz'
                max_path = '/'.join(max_path[0])+'_traj.xyz'
                min_path = os.path.relpath(min_path, app.config['GRAIN_DATABASE'])
                max_path = os.path.relpath(max_path, app.config['GRAIN_DATABASE'])
                gam_dict = {'max_en':max_en, 'min_en':min_en, 'min_coords':min_coords, 'max_coords':max_coords,
                            'min_path':min_path, 'max_path':max_path}
        else:
            print "No potential directory:", potential, "found."
            gam_dict = {'max_en':0.0, 'min_en':0.0,'min_coords':[],'max_coords':[], 'max_path':'', 'min_path':''}
        return gam_dict

    def plot_gamsurf(self, pot_dir='PotBH', rcut=None, print_gamsurf=False):
        """
        Returns:
          min and max rigid body translations for a particular rcut value. If rcut is None
          return min and max rigid body translation for minimum energy structure for all
          atom deletion criterion.
        """

        subgb_files = []
        self.find_gb_json(pot_dir, subgb_files, 'subgb.json')
        gam_surfs   = []
        for gb in subgb_files:
            with open(gb[1],'r') as f:
                gb_json = json.load(f)
            locen = self.calc_energy(gb_json)
            if locen is not None and locen >= 0.0:
                gam_surfs.append((gb_json['rcut'], gb_json['rbt'][0], gb_json['rbt'][1], locen, gb[0]))
            else:
                pass
        logging.info('gam_surfs', gam_surfs)
        if rcut is not None:
            gam_surf_rcut = filter(lambda x: x[0]==rcut, gam_surfs)
        else:
            gam_surf_rcut = gam_surfs
        if print_gamsurf:
            for count, gs in enumerate(gam_surf_rcut):
                print gs[1], gs[2], gs[3]
                if ((count+1)%6==0):
                    print '\n'
        en_list = [x[3] for x in gam_surfs]
        en_list = [x for x in en_list if x is not None]
        min_en  = min(en_list)
        print 'Min Energy: ', min_en, 'J/m^{2}'
        min_coords = filter(lambda x: round(x[3], 5) == round(min_en, 5), gam_surfs)
        print 'Coordinates of Min Energy Grain Boundaries:'
        for m in min_coords:
            print m
        max_en  = max(en_list)
        print 'Max Energy: ', max_en, 'J/m^{2}'
        print 'Coordinates of Max Energy Grain Boundaries:'
        max_coords = filter(lambda x: round(x[3], 5)==round(max_en, 5), gam_surfs)
        for m in max_coords:
            print m
        return min_coords, max_coords

    def list_all_unconverged(self, pattern='b111'):
        """Searches through subdirectories that match against pattern for
        subgb.json files.

        Parameters:
          pattern: regex to pattern match against.

        Returns:
          Three lists converged_dirs, unconverged_dirs, dirs_missing_convergence_keys
        """

        jobdirs = glob.glob('{0}*'.format(pattern))
        jobdirs = filter(os.path.isdir, jobdirs)
        scratch = os.getcwd()
        converged_list   = []
        unconverged_list = []
        missing_key_list = []
        for job in jobdirs:
            os.chdir(os.path.join(scratch, job))
            subgb_files = []
            self.find_gb_json('./', subgb_files, 'subgb.json')
            for gb in subgb_files:
                with open(gb[1],'r') as f:
                    gb_json = json.load(f)
                if 'converged' in gb_json:
                    if gb_json['converged']:
                        #print 'Converged', gb
                        converged_list.append([job]+gb)
                    elif not gb_json['converged']:
                        #print 'Not Converged', gb
                        unconverged_list.append([job]+gb)
                elif 'converged' not in gb:
                        #print 'subgb missing converged key', gb
                    missing_key_list.append(gb)
        os.chdir(scratch)
        return  converged_list, unconverged_list, missing_key_list

    def list_unconverged(self, pattern='001', potential='PotBH'):
        """ Find all unconverged subgrains. Convergence flag is in
        subgb.json file.

        Args:
          pattern(str): Orientation axis
          potential(str): Interatomic Potential to search.

        Returns:
          Three lists converged_dirs, unconverged_dirs, dirs_missing_convergence_keys
        """

        jobdirs = glob.glob('{0}*'.format(pattern))
        jobdirs = filter(os.path.isdir, jobdirs)
        scratch = os.getcwd()
        converged_list   = []
        unconverged_list = []
        missing_key_list = []
        for job in jobdirs:
            os.chdir(os.path.join(scratch, job))
            subgb_files = []
            if os.path.isdir(potential):
                self.find_gb_json(potential, subgb_files, 'subgb.json')
                for gb in subgb_files:
                    with open(gb[1],'r') as f:
                        gb_json = json.load(f)
                    if 'converged' in gb_json:
                        if gb_json['converged']:
                         # print 'Converged', gb
                            converged_list.append([job]+gb)
                        elif not gb_json['converged']:
                         # print 'Not Converged', gb
                            unconverged_list.append([job]+gb)
                    elif 'converged' not in gb:
                         # print 'subgb missing converged key', gb
                        missing_key_list.append(gb)
        os.chdir(scratch)
        return  converged_list, unconverged_list, missing_key_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--extracten", action="store_true", help="Pull all energies for an orientation axis \
                                                      print lowest energies to terminal. List is ordered by angle.")
    parser.add_argument("-g", "--gam_min", action="store_true", help="Pull gamma surface for specified potential directory.")
    parser.add_argument("-d", "--directory", default="PotBH", help="Directory to search for min_en structure. (Default PotBH).")
    parser.add_argument("-m", "--material", help="The material we wish to query. Default (alphaFe).", default="alphaFe")
    parser.add_argument("-o", "--or_axis", help="Orientation axis.", default="001")
    parser.add_argument("-pt", "--potential", help="Potential file.", default ="PotBH.xml")
    args = parser.parse_args()
    analyze =  GBAnalysis()

    if args.extracten:
        or_axis = args.or_axis
        gb_list = analyze.extract_energies(or_axis=or_axis)
        for gb in sorted(gb_list, key = lambda x: x['angle']):
            if gb['param_file'] == args.potential:
                try:
                    print gb['param_file'], gb['angle'], min(gb['energies']), max(gb['energies'])
                except ValueError:
                    print 'No Valid Energy: ', gb['param_file'], gb['angle']

    if args.gam_min:
#   Search potential directory for the gamma surface it contains
#   for all the cutoff radii.
        analyze.gam_min(directory=args.directory)
