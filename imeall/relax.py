import os
import sys
import json
import shutil
import ase.io
import logging
import argparse
import numpy as np
import logging

from pprint import pprint
from cStringIO import StringIO
from ase.optimize.sciopt import SciPyFminCG
from ase.optimize import BFGS, FIRE, LBFGS, MDMin, QuasiNewton
from ase.constraints import UnitCellFilter, StrainFilter

try:
    from quippy.io import  write, AtomsWriter
    from quippy import Atoms, Potential, frange, set_fortran_indexing
except ImportError:
    pass

from imeall import app

set_fortran_indexing(False)

def relax_gb(gb_file='file_name', traj_steps=120, total_steps=1200, force_tol = 0.05):
    """Method to relax a grain_boundary bicrystal structure. Requires a .json
    file with at a minimum the 'param_file' variable specified.

    Args:
      gb_file(str): gbid.
      traj_steps(int): number of steps between print trajectories.
      total_steps(int): total number of force relaxation steps.
      force_tol(float): Force relaxation criterion in ev/A.

    Returns:
      :class:`ase.Atoms` Object
    """
    def converged(grain, smax, fmax):
        maxstress = max(grain.get_stress().ravel())
        rmsforces = np.sum(grain.get_forces()**2, axis=1)**0.5
        maxforce  = max(rmsforces)
        if maxforce < fmax and maxstress < smax:
            return True
        return False

    with open('subgb.json', 'r') as outfile:
        j_dict = json.load(outfile)
    try:
        POT_DIR = os.path.join(app.root_path, 'potentials')
    except KeyError:
        sys.exit("Please set POTDIR in os environment. `export POTDIR='path/to/potfiles/`")
    try:
        param_file = j_dict['param_file']
        if param_file == 'iron_mish.xml':
            eam_pot = os.path.join(POT_DIR, 'iron_mish.xml')
            r_scale = 1.0129007626
        elif param_file == 'Fe_Mendelev.xml':
            eam_pot = os.path.join(POT_DIR, 'Fe_Mendelev.xml')
            r_scale = 1.00894848312
        elif param_file == 'PotBH.xml':
            eam_pot = os.path.join(POT_DIR, 'PotBH.xml')
            r_scale = 1.00894848312
        elif param_file == 'Fe_Ackland.xml':
            eam_pot = os.path.join(POT_DIR,'Fe_Ackland.xml')
            r_scale = 1.00894185389
        elif param_file == 'Fe_Dudarev.xml':
            eam_pot = os.path.join(POT_DIR,'Fe_Dudarev.xml')
            r_scale = 1.01279093417
        elif param_file == 'gp33b.xml':
            eam_pot = os.path.join(POT_DIR,'gp33b.xml')
            sparse_file = 'gp33b.xml.sparseX.GAP_2016_10_3_60_19_29_10_8911'
            eam_pot_sparse = os.path.join(POT_DIR, sparse_file)
            shutil.copy(eam_pot, './')
            shutil.copy(eam_pot_sparse, './')
        else:
            print 'No paramfile found!'
            sys.exit()
    except KeyError:
        print 'No EAM potential file with that name. Relax failed.'
        sys.exit()

    print 'Using: ', eam_pot
    pot_file    = eam_pot.split('/')[-1]
    print '{0}.xyz'.format(gb_file)
    print os.getcwd()
    grain = io.read('{0}.xyz'.format(gb_file), index='-1')
    if param_file != 'gp33b.xml':
        pot = Potential('IP EAM_ErcolAd do_rescale_r=T r_scale={0}'.format(r_scale), param_filename=eam_pot)
    else:
        pot = Potential('IP GAP', param_filename=eam_pot)

    grain.set_calculator(pot)
    grain.info['adsorbate_info'] = None
    E_gb_init   = grain.get_potential_energy()
    traj_file   = gb_file
    if 'traj' in traj_file:
        out = AtomsWriter('{0}'.format('{0}.xyz'.format(traj_file)))
    else:
        out = AtomsWriter('{0}'.format('{0}_traj.xyz'.format(traj_file)))
    strain_mask = [0,0,1,0,0,0]
    ucf = UnitCellFilter(grain, strain_mask)
    opt = FIRE(ucf)
    cell = grain.get_cell()
    A = cell[0][0]*cell[1][1]
    H = cell[2][2]
    #Calculation dumps total energyenergy and grainboundary area data to json file.
    with open('subgb.json','r') as f:
        gb_dict = json.load(f)

    #Write an initial dict so we know if the system has been initialized but the calculation is not finished.
    with open('subgb.json', 'w') as outfile:
        for key, value in gb_dict.items():
            j_dict[key] = value
        json.dump(j_dict, outfile, indent=2)

    CONVERGED = False
    FORCE_TOL = force_tol

#default to 5 if traj_steps = 120, otherwise increases
    num_iters = int(float(total_steps)/float(traj_steps))
    logging.debug('num_iters: {}'.format(num_iters))
    for i in range(num_iters):
        opt.run(fmax=FORCE_TOL, steps=traj_steps)
        out.write(grain)
        force_array = grain.get_forces()
        max_force_II = max([max(f) for f in force_array])
        max_forces = [np.sqrt(fx**2+fy**2+fz**2) for fx, fy, fz in zip(grain.properties['force'][0],
                      grain.properties['force'][1], grain.properties['force'][2])]
        if max(max_forces) <= FORCE_TOL:
            CONVERGED = True
            break
    out.close()

    gb_dict['converged'] = CONVERGED
    E_gb    = grain.get_potential_energy()
    gb_dict['E_gb']      = E_gb
    gb_dict['E_gb_init'] = E_gb_init
    gb_dict['area'] = A
    with open('subgb.json', 'w') as outfile:
        for key, value in gb_dict.items():
            j_dict[key] = value
        json.dump(j_dict, outfile, indent=2)

    if param_file == 'gp33b.xml':
        os.remove(param_file)
        os.remove(sparse_file)
    else:
        pass
    return grain

if __name__ == '__main__':
#Command line tool for relaxing grainboundary structure
    parser = argparse.ArgumentParser()
    parser.add_argument('-inp', '--input_file', help='name of input structure file')
    parser.add_argument('-ts',  '--traj_steps', help='Number of steps to write trajectory to file', type=int, default=1200)
    parser.add_argument('-f',  '--force_tol', help='Force tolerance for minimization', type=float, default=0.05)
    args = parser.parse_args()
    input_file = args.input_file
    relax_gb(gb_file=input_file, traj_steps=args.traj_steps, force_tol=args.force_tol)
