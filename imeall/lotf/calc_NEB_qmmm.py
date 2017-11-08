from __future__ import print_function
import matplotlib
matplotlib.use('Agg')

import os
import sys

import argparse

import json 
import numpy as np

from ase.neb import fit0
from ase.io import write
from ase.optimize import FIRE
from ase.optimize.precon import PreconFIRE, Exp


from distutils import spawn

from imeall import app
from ForceMixerCarver import ForceMixingCarvingCalculator
from matscipy.socketcalc import VaspClient, SocketCalculator

from quippy import AtomsReader, AtomsWriter
from quippy import Potential

import matplotlib.pyplot as plt

from imeall.calc_elast_dipole import find_h_atom

#from ase.neb import NEB
from nebForceIntegrator import NEB

class NEBAnalysis(object):
  def __init__(self):
    pass

  def save_barriers(self, images, neb, prefix=""):
    """ Saves the NEB results and plots energy barrier.
    Arguments:
      images :: 
      prefix ::
      neb :: :class:ase:`NEB`
    """
    if not os.path.exists('NEB_images'):
        os.mkdir('NEB_images')

    for i, image in enumerate(images):
        write("NEB_images/" + prefix + "ini_NEB_image%03i.xyz" % i, image)

    plot_dir = "NEB_plots"
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    R = [atoms.positions for atoms in images]
    E = neb.get_potential_energies()
    F = [atoms.get_forces() for atoms in images]
    A = images[0].cell
    pbc = images[0].pbc
    s, E, Sfit, Efit, lines = fit0(E, F, R, A, pbc)

    s = np.array(s)
    norm = s.max()
    s /= norm
    Sfit /= norm

    for x, y in lines:
        x /= norm
        ax.plot(x, y, '-C0')

    ax.plot(Sfit, Efit, 'C0-')
    Ef = max(Efit) - E[0]
    ax.plot(s, E, "oC0", label="%s: $E_f = %.4f$"%("QM/MM EAM result", Ef))
    ax.legend(loc="best")
    ax.grid(True, linestyle="dashed")

    np.savetxt(plot_dir + '/' + prefix + "_s_E.txt", np.array([s, E]))
    np.savetxt(plot_dir + '/' + prefix + "_fit.txt", np.array([Sfit, Efit]))
    np.savetxt(plot_dir + '/' + prefix + "_lines.txt", np.vstack(lines))

    fig.savefig(plot_dir + '/' + prefix + "_barrier.eps")
    return None


class NEBPaths(object):
  def __init__(self):
    pass

  def build_h_nebpath(self, struct_path="fe_bcc.xyz", neb_path=np.array([0.25, 0.0, -0.25]), alat=2.8297, knots=5, sup_cell=5, fmax=1.e-4):
    """
    Takes a vector neb_path, and generates n intermediate images along the minimum energy path.
    the struct path should point to the relaxed structure.
    """
    POT_DIR = os.path.join(app.root_path, 'potentials')
    eam_pot = os.path.join(POT_DIR, 'PotBH.xml')
    r_scale = 1.00894848312
    mm_pot = Potential('IP EAM_ErcolAd do_rescale_r=T r_scale={0}'.format(r_scale), param_filename=eam_pot)

    ats_ini = AtomsReader(struct_path)[-1]
    #Pick the top atom in the cell.
    tetra_pos = alat*np.array([0.5, 0.0, 0.75])
    mid_point = 0.5*(np.diag(ats_ini.get_cell()))
    mid_point = [((sup_cell-1)/2.)*alat for sp in range(3)]
    h_pos = tetra_pos + mid_point
    images = []
    for knot in range(knots+1):
      ats_tmp = ats_ini.copy()
      h_tmp = h_pos + ((float(knot)/float(knots))*alat)*neb_path
      ats_tmp.add_atoms(np.array(h_tmp), 1)
      images.append(ats_tmp)

#Relax images at the start and end of trajectory.
    opt = FIRE(images[0])
    images[0].set_calculator(mm_pot)
    opt.run(fmax=fmax)

    opt = FIRE(images[-1])
    images[-1].set_calculator(mm_pot)
    opt.run(fmax=fmax)
    return images

mpirun = spawn.find_executable('mpirun')
vasp = '/home/mmm0007/vasp/vasp.5.4.1/bin/vasp_std'

parser = argparse.ArgumentParser()
parser.add_argument("--buff", "-b", type=float, default=6.0)
parser.add_argument("--qm_radius", "-q", type=float, default=2.0)
parser.add_argument("--use_socket", "-u", action="store_false")
parser.add_argument("--neb_path", "-n", nargs="+", type=float, default=[0.25, 0, -0.25])
parser.add_argument("--fmax", "-f", type=float, help="maximum force for relaxation.", default=0.01)
args = parser.parse_args()

buff = args.buff
qm_radius = args.qm_radius

POT_DIR = os.path.join(app.root_path, 'potentials')
eam_pot = os.path.join(POT_DIR, 'PotBH.xml')
r_scale = 1.00894848312
mm_pot = Potential('IP EAM_ErcolAd do_rescale_r=T r_scale={0}'.format(r_scale), param_filename=eam_pot)

#disloc_fin, disloc_ini, bulk = sd.make_barrier_configurations(calculator=lammps, cylinder_r=cylinder_r)
gb_cell = AtomsReader('fe_bcc_h.xyz')[-1]

defect = find_h_atom(gb_cell)
h_pos = defect.position
x, y, z = gb_cell.positions.T
radius1 = np.sqrt((x - h_pos[0])**2 + (y-h_pos[1])**2 + (z-h_pos[2])**2)

qm_region_mask = (radius1 < qm_radius)
qm_buffer_mask = (radius1 < qm_radius + buff)

print ("\nNumber of atoms in qm region of %.1f" % qm_radius +
                                "A : %i" % np.count_nonzero(qm_region_mask))

print ("together with the buffer of %.1f" % (qm_radius + buff ) +
                                "A %i" % np.count_nonzero(qm_buffer_mask))

magmoms=[2.6 for _ in range(np.count_nonzero(qm_buffer_mask))]
vasp_args = dict(xc='PBE', amix=0.01, amin=0.001, bmix=0.001, amix_mag=0.01, bmix_mag=0.001,
                 kpts=[1, 1, 1], kpar=1, lreal='auto', nelmdl=-15, ispin=2, prec='Accurate', ediff=1.e-3,
                 nelm=100, algo='VeryFast', lplane=False, lwave=False, lcharg=False, istart=0, encut=320,
                 magmom=magmoms, maxmix=30, #https://www.vasp.at/vasp-workshop/slides/handsonIV.pdf #for badly behaved clusters.
                 voskown=0, ismear=1, sigma=0.1, isym=0) # possibly try iwavpr=12, should be faster if it works

# parallel config.
procs = 48
kpts = [1, 1, 1]
# need to have procs % n_par == 0
n_par = 1
if procs <= 8:
  n_par = procs
else:
  for _npar in range(2, int(np.sqrt(1.*procs))):
    if procs % int(_npar) == 0:
      n_par = procs // int(_npar)

if args.use_socket:
  vasp_client = VaspClient(client_id=0, npj=procs, ppn=1,
                           exe=vasp, mpirun=mpirun, parmode='mpi',
                           ibrion=13, nsw=1000000,
                           npar=n_par, **vasp_args)
  qm_pot = SocketCalculator(vasp_client)
else:
#for entirely mm potential
  qm_pot = Potential('IP EAM_ErcolAd do_rescale_r=T r_scale={0}'.format(1.00894848312), param_filename=eam_pot)


nebpath = NEBPaths()
nebanalysis = NEBAnalysis()
images = nebpath.build_h_nebpath(neb_path=np.array(args.neb_path), fmax = args.fmax)

for image in images:
  qmmm_pot = ForceMixingCarvingCalculator(image, qm_region_mask,
                                          mm_pot, qm_pot,
                                          buffer_width=buff,
                                          pbc_type=[False, False, False])
  image.set_calculator(qmmm_pot)

neb = NEB(images, force_only=True)
neb.interpolate(mic=False)

nebanalysis.save_barriers(images, neb, prefix="ini")

opt = FIRE(neb)
opt.run(fmax=args.fmax)

nebanalysis.save_barriers(images, neb, prefix="fin")

if args.use_socket:
  sock_calc.shutdown()

