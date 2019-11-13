import os
import sys
import argparse
import numpy as np

from ase.optimize import BFGS, FIRE
from ase.constraints import UnitCellFilter
from ase.lattice.cubic import BodyCenteredCubic
from ase import Atoms as aseAtoms

from imeall import app

try:
    from quippy import Atoms, Potential, AtomsReader
except ImportError:
    pass

def calc_bulk_dissolution(args):
    """Calculate the bulk dissolution energy for hydrogen
    in a tetrahedral position in bcc iron.
    Args:
      args(list): determine applied strain to unit cell.
    """
    POT_DIR = os.path.join(app.root_path, 'potentials')
    eam_pot = os.path.join(POT_DIR, 'PotBH.xml')
    r_scale = 1.00894848312
    pot = Potential('IP EAM_ErcolAd do_rescale_r=T r_scale={0}'.format(r_scale), param_filename=eam_pot)
    alat = 2.83

    gb = BodyCenteredCubic(directions = [[1,0,0], [0,1,0], [0,0,1]],
                           size = (6,6,6), symbol='Fe', pbc=(1,1,1),
                           latticeconstant = alat)
    cell = gb.get_cell()
    print 'Fe Cell', cell

    e1 = np.array([1,0,0])
    e2 = np.array([0,1,0])
    e3 = np.array([0,0,1])

    if args.hydrostatic != 0.0:
        strain_tensor = np.eye(3) + args.hydrostatic*np.eye(3)
        cell = cell*strain_tensor
        gb.set_cell(cell, scale_atoms=True)
        print 'Hydrostatic strain', args.hydrostatic
        print 'strain tensor', strain_tensor
        print gb.get_cell()
    elif args.stretch != 0.0:
        strain_tensor = np.tensordot(e2, e2, axes=0)
        strain_tensor = np.eye(3) + args.stretch*strain_tensor
        cell = strain_tensor*cell
        print 'Stretch strain'
        print 'Cell:', cell
        gb.set_cell(cell, scale_atoms=True)
    elif args.shear != 0.0:
        strain_tensor = np.tensordot(e1, e2, axes=0)
        strain_tensor = np.eye(3) + args.shear*strain_tensor
        cell = strain_tensor.dot(cell)
        print 'Shear Strain', strain_tensor
        print 'Cell:', cell
        gb.set_cell(cell, scale_atoms=True)
        gb.write('sheared.xyz')
    else:
        print 'No strain applied.'

    tetra_pos = alat*np.array([0.25, 0.0, 0.5])
    h2 = aseAtoms('H2', positions=[[0, 0, 0],[0, 0, 0.7]])
    h2 = Atoms(h2)

    gb = Atoms(gb)
    gb_h = gb.copy()
    gb_h.add_atoms(tetra_pos, 1)

    #caclulators
    gb.set_calculator(pot)
    h2.set_calculator(pot)
    gb_h.set_calculator(pot)
    gb_h.write('hydrogen_bcc.xyz')

    #Calc Hydrogen molecule energy
    opt = BFGS(h2)
    opt.run(fmax=0.0001)
    E_h2  = h2.get_potential_energy()
    h2.write('h2mol.xyz')

    #strain_mask  = [1,1,1,0,0,0]
    strain_mask  = [0,0,0,0,0,0]
    ucf = UnitCellFilter(gb_h, strain_mask)

    #opt = BFGS(gb_h)
    opt = FIRE(ucf)
    opt.run(fmax=0.0001)
    E_gb  = gb.get_potential_energy()
    E_gbh = gb_h.get_potential_energy()
    E_dis = E_gbh - E_gb - 0.5*E_h2

    print 'E_gb', E_gb
    print 'E_gbh', E_gbh
    print 'H2 Formation Energy', E_h2
    print 'Dissolution Energy', E_dis


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a','--hydrostatic', help='float, hydrostatic strain percentage on unit cell.', default=0.0, type=float)
    parser.add_argument('-u','--stretch', help='float, stretch percentage on unit cell.', default=0.0, type=float)
    parser.add_argument('-s','--shear', help='float, shear strain percentage on unit cell.', default=0.0, type=float)
    args = parser.parse_args()

    calc_bulk_dissolution(args)

#Hydrogen interatomic distance 0.73
#Hydrogen formation energy -4.
