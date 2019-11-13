import json
import spglib
import numpy as np

from ase.geometry import geometry
from ase import Atoms

from pylada_defect_port.pylada_defect import get_unique_ints, Voronoi


def remove_atoms(x, rcut=3.0, cutoff=0.2):
    """ase delete Atoms seems to cause total chaos with quippy atoms.
    """
    x.set_cutoff(rcut)
    x.calc_connect()
    x.calc_dists()
    rem = []
    u = np.zeros(3)
    for i in range(x.n):
        for n in range(x.n_neighbours(i)):
            j = x.neighbour(i, n, distance=3.0, diff=u)
            if x.distance_min_image(i,j) < cutoff and j != i:
                rem.append(sorted([j,i]))
    rem = list(set([a[0] for a in rem]))
    #if len(rem) > 0:
    #  x.remove_atoms(rem)
    #else:
    #  print 'No atoms removed'
    return rem

def decorate_interface():
    ats = Atoms('interface.xyz')
    dataset = spglib.get_symmetry_dataset(ats, symprec=1e-5)

    with open('unique_lattice_sites.json','w') as f:
        json.dump([list(ats[site_num].position) for site_num in np.unique(dataset['equivalent_atoms'])], f)

    unique_atoms = []
    for at in ats:
        unique_atoms.append(at.position)
    voronoi = Voronoi(limits=tuple(np.diag(ats.cell)), periodic=(True, True, False))
    cntr = voronoi.compute_voronoi(unique_atoms)
    ints_list = []
    for site_num in np.unique(dataset['equivalent_atoms']):
        for vert in voronoi.get_vertices(site_num, cntr):
            ints_list.append(vert.tolist())

    for unique in ints_list:
        ats.add_atoms(unique, 1)
    for i in range(len(ats)):
        ats.id[i] = i
    #remove voronoi duplicates
    print 'Fe_H atoms', len(ats)
    ats.wrap()
    del_ats = Atoms()
    for at in ats:
        del_ats.append(at)
    geometry.get_duplicate_atoms(del_ats, cutoff=0.2, delete=True)
    ats = del_ats.copy()
    print 'Fe_H atoms remove duplicates', len(ats)
    #select unique hydrogens
    #for i in range(len(ats)):
    #  ats.id[i] = i
    ints_list = [at.position for at in ats if at.number==1]
    with open('unique_h_sites.json', 'w') as f:
        json.dump([list(u)for u in ints_list], f)
    ats.write('hydrogenated_grain.xyz')

if __name__=='__main__':
    decorate_interface()
