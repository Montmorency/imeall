import os
import glob
import pickle
import shutil
import argparse
import numpy as np
import scipy.spatial as spatial

from ase.geometry import get_duplicate_atoms
from quippy import Atoms, set_fortran_indexing
from imeall.slabmaker.slabmaker import rotate_vec
from imeall.slabmaker import transformations as quat

set_fortran_indexing(False)

#@deprecated('to be replaced by external pylada_defects library.')
class Hydrify(object):
    """"
    class: Hydrify contains methods for adding hydrogens at specific positions in a
    simulation cell handles different situations: crack geometry, grainboundry,
    and conventional crystal.
    """
    def __init__(self):
        pass

    def __repr__(self):
        print 'Hydrify Object.'

    def tetravol(self, a,b,c,d):
        """
        Calculates the volume of a tetrahedron, given vertices a,b,c and d (triplets)
        http://cs.smith.edu/~orourke/books/compgeom.html
        http://stackoverflow.com/questions/19634993/volume-of-voronoi-cell-python
        """
        tetravol=abs(np.dot((a-d),np.cross((b-d),(c-d))))/6
        return tetravol

    def write_cluster(self, cl,name='cluster.xyz'):
        cl.center(vacuum=2.0)
        cl.write(name)

    def append_if_thresh(self, h_pos, rcut = 1.6):
        """
        append_if_thresh add position vector to list if
        it is greater than a specified distance from 
        existing vectors.

        Arguments:
            h_pos: list of hydrogen position vectors.
            rcut: minimum spacing of hydrogens.

        Return:
            pared_h: list pared down so minimum spacing requirement is met.
        """
        pared_h = h_pos[0]
        for h in h_pos[1:]:
            if all([np.linalg.norm(h-h_unique) > rcut for h_unique in pared_h]):
                pared_h =  np.vstack((pared_h, h))
        return pared_h

    def hydrogenate_gb(self, gb, mode='GB', z_plane=None, rr=10.0, bp=np.array([1.,9.,0.]),
                       d_plane=2.83, d_H=1.6, tetrahedral=True, alat=2.83, crackpos_fix=None):
        """
        If mode is GB Given a grain boundary, find a bulk plane to dump hydrogen in,
        and a platelet of hydrogen parallel to the grain boundary. Routine
        returns positions of a suitable "plane" +/- 2A.
        Else if mode is CrackTip we populate a spherical cluster with H.

        Arguments:
          mode    := 'GB' or 'CrackTip' either decorate a planar cluser of material with hydrogen appropriate for a grainboundary,
                      or a spherical cluster appropriate for a crack tip.
          d_H     := Nearest neighbour threshold for hydrogen atoms. Effectively determines the platelet concentration,
          d_plane := width of planar slice of bulk to cut out for Delaunay Triangulation
          bp      := boundary plane

        Returns:
            h_list := list of np.array vectors of hydrogen positions in cluster coordinates

        """
        #return empty list if H spacing is not greater than 0
        if not (d_H > 0.0):
            return []

        if mode=='GB':
            if z_plane == None:
                z_plane     = gb.lattice[2,2]/2.0
            fixed_mask = (np.sqrt(np.square(gb.positions[:,2]-z_plane)) <= d_plane)
            cl         = gb.select(fixed_mask, orig_index=True)
            cl.write('plane.xyz')
        elif mode=='CrackTip':
            #if crackpos_fix == None:
            #    fixed_mask = (np.sqrt(map(sum, map(np.square, gb.positions[:,0:3]-gb.params['CrackPos'][0:3]))) <= rr)
            #else:
            #    fixed_mask = (np.sqrt(map(sum, map(np.square, gb.positions[:,0:3]-crackpos_fix[:]))) <= rr)
            #cl         = gb.select(fixed_mask, orig_index=True)
            #cl.write('cluster.xyz')
            cl = gb.copy()
        elif mode=='Defect':
            cl = gb.copy()
        else:
            sys.exit('No Mode chosen.')
# Select the bulk plane:
        tri   = spatial.Delaunay(cl.positions, furthest_site=False)
#http://stackoverflow.com/questions/10650645/python-calculate-voronoi-tesselation-from-scipys-delaunay-triangulation-in-3d?rq=1
        p = tri.points[tri.vertices]
        A = p[:,0,:].T
        B = p[:,1,:].T
        C = p[:,2,:].T
        a = A - C
        b = B - C
        def dot2(u, v):
            return u[0]*v[0] + u[1]*v[1]

        def cross2(u, v, w):
            """u x (v x w)"""
            return dot2(u, w)*v - dot2(u, v)*w

        def ncross2(u, v):
            """|| u x v ||^2"""
            return sq2(u)*sq2(v) - dot2(u, v)**2

        def sq2(u):
            return dot2(u, u)
#answer then goes on to grab the Voronoi edges but we just
#need the circumcenters
#We want to append hydrogens at unique sites in the circumcenters.
#This list has the circumcenters for lots of different simplices
#so we removed them
        cc = cross2(sq2(a) * b - sq2(b) * a, a, b) / (2*ncross2(a, b)) + C
        plane_cc = cc.T

        if mode=='GB':
            oct_sites=filter(lambda x: np.sqrt(np.square(x[2]-z_plane)) <= 1.0, plane_cc)
            oct_sites = self.append_if_thresh(oct_sites)
        elif mode=='CrackTip':
            #oct_sites=filter(lambda x: np.sqrt(sum(map(np.square, x[:]-gb.params['CrackPos'][:]))) <= rr, plane_cc)
            oct_sites = filter(lambda x: True, plane_cc)
            oct_sites = self.append_if_thresh(oct_sites)
        elif mode=='Defect':
            #keep sites within radius 1.5 A of center of cell.
            cell = cl.get_cell()
            mid_point = np.array([cell[0][0]/2.0, cell[1][1]/2.0, cell[2][2]/2.0])
            oct_sites=filter(lambda x: np.sqrt(sum(map(np.square, x[:]-mid_point[:]))) <= 1.5, plane_cc)
            oct_sites=self.append_if_thresh(oct_sites)
        else:
            sys.exit('No valid mode chosen.')

        if tetrahedral:
            tetra_lattice = alat*np.array([[0.0,0.0,0.25]])
#Depending on orientation of the unit cell we rotate the lattice
#so that the addition of a vector from tetra_lattice is in the new x,y,z coordinate system.
#i.e. we move from ([0,0,1], [0,1,0] ,[0,0,1]) to (bp,or,bpxor). For [001] oriented grain boundaries this is just a rotation
#in the y-z plane. Other wise it is slightly more complicated. We rotate the x-coordinate [1,0,0]
#to the new orientation axis (1,1,0) or (1,1,1) and then rotate y,z to the bpxv, and z=bp (i.e.
#boundary plane and  boundary plane crossed witht the orientation axis.
            tetra = []
#the orientation axis is actually or = [0,0,1] so the z coordinate wrt to the boundary plane
#is actually the "x coordinate". Hence we dot the boundary plane with x. However the angle is same and when
#in the gb cell the orientation is along x,y,z as expected. So we take the angle from dotting [1,0,0]
#and then rotate lattice vectors in to x, y',z'.
            if mode =='GB':
                z = np.array([0.,1., 0.])
                #z = np.array([0.,0., 1.])
                theta = np.arccos(bp.dot(z)/(np.linalg.norm(bp)*(np.linalg.norm(z))))
                print 'Rotating z by: ', (180/np.pi)*theta
                rot_quat = quat.quaternion_about_axis(-theta, np.array([1,0,0]))
                for t in tetra_lattice:
                    tetra.append(rotate_vec(rot_quat,t))
#loop over octahedral sites decorating cluster
                h_list = []
                for h in oct_sites:
                    for lat in tetra:
                        h_pos = h + lat
                        h_list.append(h_pos)
                h_list = self.append_if_thresh(h_list, rcut = d_H)
                #for h_pos in h_list:
                #  cl.add_atoms(h_pos,1)
            elif mode =='CrackTip':
                #with open('gb.json','r') as f:
                #    crack_dict = pickle.load(f)
                CrackFront = gb.params['CrackFront']/np.linalg.norm(gb.params['CrackFront'])
                CleavagePlane = gb.params['CleavagePlane']/np.linalg.norm(gb.params['CleavagePlane'])
                CrackDirection = gb.params['CrackDirection']/np.linalg.norm(gb.params['CrackDirection'])

                mat = np.array([CrackDirection, CleavagePlane, CrackFront]).T
                print mat
                mat = np.matrix(mat)
                mat = mat.I
            #rotate lattice vector to new coordinate system
                print tetra_lattice[0]
                print mat.dot(tetra_lattice[0])
                for t in tetra_lattice:
                    tetra.append((mat).dot(t))
                h_list = []
                for h in oct_sites:
                    for lat in tetra:
                        h_pos = h + lat
                        h_list.append(np.array(h_pos))
                h_list = self.append_if_thresh(h_list, rcut = d_H)
            elif mode=='Defect':
                h_list = []
                print 'Octahedral Sites', oct_sites
                #assumes orthorhombic cell along [100], [010], [001].
                tetra = tetra_lattice
                print 'Tetra lattice vector', tetra
                for h in [oct_sites]:
                    print 'loop'
                    h_pos = h + tetra
                    h_list.append(np.array(h_pos))
                h_list = self.append_if_thresh(h_list, rcut = d_H)
                for h_pos in h_list:
                    gb.add_atoms(h_pos,1)
        else:
# In this case we just add to the octahedral or in a non-bulk
# environment largest volume sites.
            h_list = []
            for h in oct_sites:
                h_list.append(h)
            h_list = self.append_if_thresh(h_list, rcut = d_H)
            for h_pos in h_list:
                ats.add_atoms(h_pos,1)
        return h_list

if __name__=='__main__':
    parser  = argparse.ArgumentParser()
    parser.add_argument('-p','--pattern', default="1.traj.xyz")
    args    = parser.parse_args()
    pattern = args.pattern
    jobs = glob.glob(pattern)
    hydrify =  Hydrify()
    scratch = os.getcwd()
    for job in jobs:
        os.chdir(job)
        ats = Atoms('crack.xyz')
        hydrify.hydrogenate_gb(mode='CrackTip')
