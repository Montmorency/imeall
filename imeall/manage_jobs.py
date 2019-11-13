import re
import os
import sys
import glob
import argparse
import subprocess
import shutil

class VaspWriter(object):
    """
    Pass object a str representation of incar file.
    method mod_car is overloaded to accept a dictionary
    of variables that you may wish to modify {npar = 8}.
    """
    def __init__(self, incar_str):
        self.incar_str =incar_str

    def mod_incar(self, **kwargs):
        for k, v in kwargs.items():
            if k=='npar':
                self.incar_str = re.sub('NPAR\s+=\s+[0-9]+', 'NPAR = {npar}'.format(npar=v), self.incar_str)
            if k=='kpar':
                self.incar_str =  re.sub('KPAR\s+=\s+[0-9]+', 'KPAR = {kpar}'.format(kpar=v), self.incar_str)
            if k=='amix':
                self.incar_str =  re.sub('AMIX\s+=\s+[0-9.]+', 'AMIX = {amix}'.format(amix=v), self.incar_str)
            if k=='amix_mag':
                self.incar_str =  re.sub('AMIX_MAG\s+=\s+[0-9.]+', 'AMIX_MAG = {amix_mag}'.format(amix_mag=v), self.incar_str)
            if k=='bmix':
                self.incar_str =  re.sub('BMIX\s+=\s+[0-9.]+', 'BMIX = {bmix}'.format(bmix=v), self.incar_str)
            if k=='bmix_mag':
                self.incar_str =  re.sub('BMIX_MAG\s+=\s+[0-9.]+', 'BMIX_MAG = {bmix_mag}'.format(bmix_mag=v), self.incar_str)
            if k=='amin':
                self.incar_str =  re.sub('AMIN\s+=\s+[0-9.]+', 'AMIN = {amin}'.format(amin=v), self.incar_str)
            if k=='voskown':
                self.incar_str =  re.sub('VOSKOWN\s+=\s+[0-9]+', 'VOSKOWN = {voskown}'.format(voskown=v), self.incar_str)

    def write_incar(self):
        """
        :method:`write_incar` write the vasp INCAR file.
        """
        with open('INCAR','w') as f:
            print >> f, self.incar_str

    def gen_vasp(self, struct_file)
        from ase import Atoms
        from ase.io.vasp import write_vasp
        from ase.calculators.vasp import Vasp
        ats = Atoms(struct_file)
        vasp_args=dict(xc='PBE', amix=0.01, amin=0.001, bmix=0.001, amix_mag=0.01, bmix_mag=0.001, ediff=1.0e-8,
                   kpts=[3, 3, 3], kpar=9, lreal='auto', ibrion=-1, nsw=0, nelmdl=-15, ispin=2, prec='Accurate',
                   nelm=100, algo='VeryFast', npar=24, lplane=False, lwave=False, lcharg=False, istart=0,
                   voskown=0, ismear=1, sigma=0.1, isym=2) # possibly try iwavpr=12, should be faster if it works
        vasp = Vasp(**vasp_args)
        vasp.initialize(ats)
        write_vasp('POSCAR', vasp.atoms_sorted, symbol_count=vasp.symbol_count, vasp5=True)
        vasp.write_incar(ats)
        vasp.write_potcar()
        vasp.write_kpoints()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--submit', help='If present jobs will be submitted to scheduler.', action='store_true')
    parser.add_argument('-m', '--mod_key', help='INCAR file will be modified.', action='store_true')
    parser.add_argument('-u', '--update', help='If present CONTCAR will be copied to POSCAR before job is resubmitted.', action='store_true')
    parser.add_argument('-p', '--pattern', help='If directory pattern for submitting jobs. first characters for job pattern.')
    args = parser.parse_args()

    jobs = glob.glob('{}*'.format(args.pattern))
    jobs = filter(lambda x: os.path.isdir(x), jobs)

    scratch = os.getcwd()
    for job in jobs[:]:
        os.chdir(job)
        print 'Current Working Directory', os.getcwd()
        job = subprocess.Popen('cp /home/mmm0007/pymodules/templates/vasp.sh ./'.split())
        job.wait()
        with open('INCAR','r') as f:
            incar_str = f.read()

        vaspwriter = VaspWriter(incar_str)
        vaspwriter.mod_incar(amix=0.05, amin=0.4, bmix=0.01, amix_mag=0.1, voskown=0, bmix_mag=0.01)
        incar_str = vaspwriter.incar_str
        #incar_str = incar_str.strip()+'\n EDIFF = -0.001\n'
        incar_str = incar_str.strip()
        print incar_str
        if args.mod_key:
            with open('INCAR','w') as f:
                print >> f, incar_str
        with open('CONTCAR','r') as f:
            contcar = f.read()
            if len(contcar) > 0:
                if args.update == True:
                    shutil.copy('CONTCAR', 'POSCAR')
                    print 'COPYING CONTCAR: ', os.getcwd(), len(contcar)
            else:
                print 'CONTCAR EMPTY', os.getcwd()
        with open('POSCAR','r') as f:
            poscar = f.read()
        with open('POSCAR','w') as f:
            print >>f, poscar.replace('F   F   F', 'T   T   T')
        if args.submit:
            job = subprocess.Popen('qsub vasp.sh'.split())
            job.wait()
        os.chdir(scratch)
