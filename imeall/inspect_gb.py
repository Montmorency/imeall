import os
import sys
import json
import glob
import argparse
from   peewee    import *
from   datetime  import datetime, timedelta
from   models   import GBAnalysis, PotentialParameters
from   gb_models import GrainBoundary, SubGrainBoundary


GRAIN_DATABASE = app.config["GRAIN_DATABASE"]
DATABASE       = app.config["GRAIN_DATABASE_SQL"]
database       = SqliteDatabase(DATABASE)

class Inspector(object):
    """
    :object:`Inspector` This class inspects the SQLdatabase to list GB energies according to potential,
    orientation axis etc.
    """
    def __init__(self):
        pass

    def list_gb(self, potential="PotBH.xml", or_axis="001", print_unconverged=True):
        """
        :method:`list_gb` list energies, and convergence for grain boundaries for a particular
        potential and orientation axis.
        :attributes:
          potential: Potential used to determine energies and forces.
          or_axis: orientation_axis
          print_unconverged: If true prints the files for unconverged grainboundaries which can
          be resubmitted with `sub_unconv.py`.
        """
        pot_param     = PotentialParameters()
        ener_per_atom = pot_param.gs_ener_per_atom()
        #serialize orientation vector:
        oraxis        = ','.join([c for c in or_axis])
        gbs           = GrainBoundary.select().where(GrainBoundary.orientation_axis==oraxis).order_by(GrainBoundary.angle)
        unconverged   = []
        for gb in gbs:
            print "OR axis: {or_axis}, Angle: {angle}".format(or_axis=gb.orientation_axis, angle=round(gb.angle*(180.0/3.14159),2))
            subgbs = (gb.subgrains.select(GrainBoundary, SubGrainBoundary)
                        .where(SubGrainBoundary.potential==potential)
                        .join(GrainBoundary)
                        .order_by(SubGrainBoundary.E_gb)
                        .dicts())
            if len(subgbs) > 0:
                subgbs = [(16.02*(subgb['E_gb']-float(subgb['n_at']*ener_per_atom['PotBH.xml']))/(2.0*subgb['area']), subgb) for subgb in subgbs]
                subgbs.sort(key = lambda x: x[0])
                for subgb in subgbs:
                    print "\t", subgb[1]['gbid'], subgb[0], subgb[1]['converged']
                    if not subgb[1]['converged']:
                        unconverged.append((subgb[1]['gbid'], subgb[1]['path']))
            else:
                print "No Subgrains."
        print "{} unconverged subgrains".format(len(unconverged))
        if print_unconverged:
            with open('unconv_list_{or_axis}_{pot}.txt'.format(or_axis=or_axis, pot=potential.split('.')[0]), 'w') as f:
                for unconv in unconverged:
                    print >>f,  unconv[1]

    def list_pot_dir(self, potential="EAM_Dud", or_axis="001", material="alphaFe"):
        target_dir = os.path.join(GRAIN_DATABASE, material)
        target_dir = os.path.join(target_dir, or_axis)
        jobdirs    =  os.listdir(target_dir)
        no_pot_list = []
        no_subgb_list = []
        for job in jobdirs:
            pot_dir = os.path.join(target_dir, job)
            if os.path.isdir(pot_dir):
                pass
            else:
                continue
            pot_dir = os.path.join(pot_dir, potential)
            try:
                assert os.path.isdir(pot_dir)
            except AssertionError:
                print pot_dir, 'Does not exist'
                no_pot_list.append(pot_dir.split('/')[-2])
            else:
                if len(os.listdir(pot_dir)) < 5:
                    no_subgb_list.append(pot_dir.split('/')[-2])
                    print pot_dir, 'Short'
                for subgb in os.listdir(pot_dir):
                    print subgb
        print 'No potential files:'
        for gb in no_pot_list:
            print gb
        print 'Too few subgrains:'
        for subgb in no_subgb_list:
            print subgb

if __name__=="__main__":
    parser    = argparse.ArgumentParser()
    parser.add_argument("-p", "--potential", help="potential to pull from database.", default="PotBH.xml")
    parser.add_argument("-d", "--directory", help="name of potential directory.", default="PotBH")
    parser.add_argument("-o", "--or_axis",   help="orientation axis to pull from database.", default="001")
    parser.add_argument("-c", "--converged", help="print list of unconverged grains.", action="store_true")
    parser.add_argument("-l", "--list_pot_dir", help="print subgrains in the PotDir.", action="store_true")
    args      = parser.parse_args()

    inspector = Inspector()
    if args.list_pot_dir:
        inspector.list_pot_dir(potential=args.directory)

    if args.converged:
        inspector.list_gb(potential=args.potential, or_axis=args.or_axis, print_unconverged=args.converged)
