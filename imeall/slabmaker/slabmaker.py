import os
import json
import time
import numpy as np
import transformations as quat

from ase.io import read
from ase.io import write
from ase.lattice.spacegroup import crystal
from ase.lattice.surface import surface, bcc111, bcc110
from ase.lattice.cubic   import BodyCenteredCubic
from ase.utils.geometry  import get_duplicate_atoms


def rotate_plane_z(grain, miller):
    """Rotates atoms in grain so that planes parallel to the plane defined by the vector
    miller are parallel to the xy plotting plane.

    Args:
      grain(:py:class:`Atoms`): grain structure to be rotated.
      miller(numpy array): 3x1 vector defining plane.

    Returns:
      rotation_quaternion to achieve this rotation.
    """
    z = np.array([0.,0., 1.])
    p = np.cross(miller, z)
    p *= 1./np.linalg.norm(p)
    theta = np.arccos(miller.dot(z)/(np.linalg.norm(miller)*(np.linalg.norm(z))))
    if theta != theta:
        theta = 0.
        print 'no miller plane specified', theta
        rotation_quaternion = np.array([1., 0., 0., 0.])
    else:
        rotation_quaternion = quat.quaternion_about_axis(theta, p)
    print '\t Aligning Orientation [', int(miller[0]), int(miller[1]), int(miller[2]), '] axis with +z-axis'
    print '\t rotation: ', theta, '(', theta*(180./np.pi), ')'
    print '\t around : ', p, '\n'
    print '\t Rotated Vector (should be along z): ', rotate_vec(rotation_quaternion, miller).round(3)
    grain = rotate_grain(grain, q=rotation_quaternion)
    return rotation_quaternion

def build_tilt_sym_gb(gbid='', bp=[3,3,2], v=[1,-1,0], min_spacing=12.0,
                      c_space=None, target_dir=None, rbt=None, chem_symbol='Fe',
                      alat=2.83):
    """Generate symmetric tilt grain boundary with appropriate configurations boundary
    plane (bp) oriented along z axis and orthogonal directions in the
    the other two planes given the orientation axis (v) and an orthogonal vector
    bpxv so we have a proper cube. If rbt is not None then rigid body
    translations are present, this is passed as a list of two numbers
    abs(rbt[0]) < 1. The cell is then displaced as a function of these numbers.
    This routine should work for any BCC crystal with the appropriate modification
    to alat and chem_symbol.

    Args:
      gbid(str):id label for the grain boundary.
      bp(list, int): boundary plane normal.
      v(list,int): orientation axis.
      min_spacing(float): minimum spacing between grain boundaries.
      c_space(float,optional): inter-planar spacing if not passed this is calculated automatically.
      target_dir(str): string specifying grain boundary target directory.
      rbt(list,float): rigid body translations.
      alat(float): Lattice constant of the unit cell.

    Returns:
      if target_dir==None returns [z_planes, len(dups), n_grain_unit, grain_c]
      i.e the location of the interfacial planes, the number of duplicate atoms
      when the grain is built, the
      number of atoms in the canonical unit cell, and the bicrystal :class:`ase.Atoms` object.
      else returns :class:`ase.Atoms`.
    """

    #bpxv [boundary plane]X[orientation axis] cross product. Gives y axis of bicrystal.
    bpxv = [(bp[1]*v[2]-v[1]*bp[2]),(bp[2]*v[0]-bp[0]*v[2]),(bp[0]*v[1]- v[0]*bp[1])]
    #orientation axis == x
    grain_a = BodyCenteredCubic(directions=[v, bpxv, bp],
                                size=(1,1,1), symbol=chem_symbol, pbc=(1,1,1),
                                latticeconstant=alat)

    n_grain_unit = len(grain_a)
    #A safe minimum value is 12 A, in most cases should be at least 2*r_cut of potential
    n = 2
    while(grain_a.get_cell()[2,2] < min_spacing and n < 10):
        grain_a = BodyCenteredCubic(directions=[v, bpxv, bp],
                                    size=(1,1,n), symbol=chem_symbol, pbc=(1,1,1),
                                    latticeconstant=alat)
        n += 1
    print '\t {0} repeats in z direction'.format(n)
    grain_b = grain_a.copy()
    grain_c = grain_a.copy()
    print '\t', '{0} {1} {2}'.format(v[0],v[1],v[2])
    print '\t', '{0} {1} {2}'.format(bpxv[0], bpxv[1], bpxv[2])
    print '\t', '{0} {1} {2}'.format(bp[0], bp[1], bp[2])
    if c_space==None:
        s1 = surface('Fe', (map(int, bp)), 1)
        c_space = alat/np.sqrt(float(bp[0]**2+bp[1]**2 + bp[2]**2)) #-s1.positions[:,2].max()
        print s1.get_cell()
        s2 = surface('Fe', (map(int, v)), 1)
        x_space = s2.get_cell()[0,0] #-s1.positions[:,2].max()
        s3 = surface('Fe', (map(int, bpxv)), 1)
        y_space = s3.get_cell()[1,1] #-s1.positions[:,2].max()
    print '\t Interplanar spacing: ', n, x_space.round(2), y_space.round(2), c_space.round(2), 'A'

# Reflect grain b in z-axis (across mirror plane):
    print grain_a.get_cell()[2,2]-grain_a.positions[:,2].max()
    grain_b.positions[:,2]  = -1.0*grain_b.positions[:,2]
    grain_c.extend(grain_b)
    grain_c.set_cell([grain_c.get_cell()[0,0], grain_c.get_cell()[1,1], 2.*grain_c.get_cell()[2,2]])
    grain_c.positions[:,2] += abs(grain_c.positions[:,2].min())
    pos = [grain.position for grain in grain_c]
    pos = sorted(pos, key= lambda x: x[2])
    dups = get_duplicate_atoms(grain_c)

# Now build the second grain boundary by reflecting in 'y' plane
    grain_b = grain_c.copy()
    grain_b.positions[:,0] = -grain_b.positions[:,0]
    grain_b.positions[:,1] = -grain_b.positions[:,1]
    grain_b.wrap()
# In BCC lattice spacing is different depending on whether the
# z-plane has an even or odd number of odd miller indices:
    if sum([int(n)%2 for n in bp])%2 == 0 :
        grain_b.positions[:,2] -= (grain_b.positions[:,2].max()+2*c_space)
    else:
        grain_b.positions[:,2] -= (grain_b.positions[:,2].max()+c_space)
    grain_c.extend(grain_b)
    grain_c.set_cell([grain_c.get_cell()[0,0], grain_c.get_cell()[1,1], 2.*grain_c.get_cell()[2,2]])
    grain_c.positions[:,2] += abs(grain_c.positions[:,2].min())
    dups = get_duplicate_atoms(grain_c)
# Displace replicated atoms along the unit cell
    for dup in dups:
        grain_c[dup[1]].position[2] += grain_a.get_cell()[2,2]
    shared_atoms = [grain_c[dup[0]] for dup in dups]
    for at in shared_atoms:
        print '\t', at.index, at.position
    print '\t There are {0} shared atoms'.format(len(shared_atoms))
    z_planes = [round(atom.position[2],4) for atom in shared_atoms]
    z_planes = list(sorted(set(z_planes)))
    cell = grain_c.get_cell()
    if rbt != None:
        for grain in grain_c:
            if (z_planes[0] < grain.position[2].round(4) < z_planes[1]):
                grain.position[0] += rbt[0]*cell[0][0]
                grain.position[1] += rbt[1]*cell[1][1]
#so move one of the coincident atoms if there are any.
        for atom in shared_atoms:
            grain_c.positions[atom.index, 0] += rbt[0]*cell[0][0]
            grain_c.positions[atom.index, 1] += rbt[1]*cell[1][1]
        grain_c.wrap()
#Grain was explicitly set to 2*grain A before so we need
#to space out an additional layer.
    if sum([int(n)%2 for n in bp])%2 == 0 :
        grain_c.center(vacuum=c_space/2., axis=2)
    else:
        grain_c.center(vacuum=c_space/4., axis=2)
# if we specify a target directory take a picture and
# write the file there directly, otherwise just return the grain
# for further processing.
    if target_dir != None:
        print '\t Writing {0}.xyz to file'.format(gbid)
        write('{0}.xyz'.format(os.path.join(target_dir, gbid)), grain_c)
        return [z_planes, len(dups), n_grain_unit, grain_c]
    else:
        return grain_c

def calc_twist_sigma_csl(bp, v, chem_symbol="Fe", alat=2.83, n=3):
    """Calculate sigma number for a twist symmetric structure.

    Args:
      bp(:numpy:`array`): Boundary plane.
      v(:numpy:`array`): x-axis of unit cell.
      chem_symbol(str): Chemical element abbreviation.
      n(int): size of unit cell.

    Returns:
      list: number of atoms, duplicate pairs, Sigma
    """
    bpxv = [(bp[1]*v[2]-v[1]*bp[2]),(bp[2]*v[0]-bp[0]*v[2]),(bp[0]*v[1]-v[0]*bp[1])]
    grain_a = BodyCenteredCubic(directions=[list(v), list(bpxv), list(bp)],
                                 size=(n,n,n), symbol=chem_symbol, pbc=(1,1,1),
                                 latticeconstant=alat)
    n_at = len(grain_a)
    v2 = v.copy()
    if np.allclose(bp, [0,0,1]):
        v2[0] = -v2[0]
    elif np.allclose(bp, [1,1,0]):
        v2[2] = -v2[2]
    elif np.allclose(bp, [1,1,1]):
        v2[1] = -v2[1]
        v2[2] = -v2[2]

    bpxv = [(bp[1]*v2[2]-v2[1]*bp[2]),(bp[2]*v2[0]-bp[0]*v2[2]),(bp[0]*v2[1]- v2[0]*bp[1])]
    grain_b = BodyCenteredCubic(directions = [list(v2), list(bpxv), list(bp)],
                                size = (n,n,n), symbol=chem_symbol, pbc=(1,1,1),
                                latticeconstant = alat)
    grain_a.extend(grain_b)
    dups = get_duplicate_atoms(grain_a)
    print 'n_at', n_at, 'dups', len(dups), 'Sigma:', float(n_at)/float(len(dups))
    return n_at, len(dups), float(n_at)/float(len(dups))


def build_twist_sym_gb(gbid='', bp=[0,0,1], v=[3,5,0], min_spacing=14.0,
                       c_space=None, target_dir=None, rbt=None, chem_symbol='Fe',
                       alat=2.83):
    """Builder for twist boundary structures. For the twist boundaries the orientation
    axis (or_axis) also defines the boundary plane normal.

    Args:
      gbid(str): :class:`imeall.gb_models.SubGrainBoundary` id.
      bp(list[int]): Boundary plane.
      v(list[int]): Defines the 'x' axis of the orthorhombic bicrystal.
      min_spacing(float): Minimum separation between interfaces.
      c_space(float): Interplanar spacing of boundary planes.
      target_dir(str): Path str of where to deposit grain boundary.
      rbt(list[float]): Rigid body translations.
      chem_symbol(str): chemical symbol of bcc element.
      alat(float): lattice constanct BCC crystal.

    Returns:
      if target_dir==None returns list [z_planes, sigma_csl, n_grain_unit, grain_c]
      else returns :class:`ase.Atoms`.
    """
    #bpxv [boundary plane]X[orientation axis] cross product. Gives y axis of bicrystal.
    bpxv = [(bp[1]*v[2]-v[1]*bp[2]),(bp[2]*v[0]-bp[0]*v[2]),(bp[0]*v[1]-v[0]*bp[1])]
    grain_a =  BodyCenteredCubic(directions=[list(v), list(bpxv), list(bp)],
                                 size=(1,1,1), symbol=chem_symbol, pbc=(1,1,1),
                                 latticeconstant=alat)
    n_grain_unit = len(grain_a)
    n = 2
    while(grain_a.get_cell()[2,2]< min_spacing and n < 10):
        grain_a = BodyCenteredCubic(directions=[list(v), list(bpxv), list(bp)],
                                  size=(1,1,n), symbol=chem_symbol, pbc=(1,1,1),
                                  latticeconstant = alat)
        n += 1
    print '\n'
    print 'Cell 1 Lattice Orientation'
    print '\t v (xdim)', '{0} {1} {2}'.format(v[0],v[1],v[2])
    print '\t bpxv (ydim)', '{0} {1} {2}'.format(bpxv[0], bpxv[1], bpxv[2])
    print '\t bp (zdim)', '{0} {1} {2}'.format(bp[0], bp[1], bp[2])
    print '\t {0} repeats in z direction'.format(n)
    v2      = v.copy()
    if np.allclose(bp, [0,0,1]):
        v2[0]   = -v2[0]
    elif np.allclose(bp, [1,1,0]):
        v2[2]   = -v2[2]
    elif np.allclose(bp, [1,1,1]):
        v2[1]   = -v2[1]
        v2[2]   = -v2[2]
#Build second cell.
    bpxv    = [(bp[1]*v2[2]-v2[1]*bp[2]),(bp[2]*v2[0]-bp[0]*v2[2]),(bp[0]*v2[1]- v2[0]*bp[1])]
    grain_b = BodyCenteredCubic(directions = [list(v2), list(bpxv), list(bp)],
                                size = (1,1,1), symbol=chem_symbol, pbc=(1,1,1),
                                latticeconstant = alat)
    print '\n'
    print 'Cell 2 Lattice Orientation'
    print '\t v (xdim)', '{0} {1} {2}'.format(v2[0],v2[1],v2[2])
    print '\t bpxv (ydim)', '{0} {1} {2}'.format(bpxv[0], bpxv[1], bpxv[2])
    print '\t bp (zdim)', '{0} {1} {2}'.format(bp[0], bp[1], bp[2])
    print '\t {0} repeats in z direction'.format(n)

    print 'Calculated Misorientation Angle: ', np.arccos((v.dot(v2))/(np.linalg.norm(v)*np.linalg.norm(v2)))*(180./np.pi)

    n = 2
    while(grain_b.get_cell()[2,2] < 2*min_spacing and n < 10):
        grain_b = BodyCenteredCubic(directions = [list(v2), list(bpxv), list(bp)],
                                    size = (1,1,n), symbol=chem_symbol, pbc=(1,1,1),
                                    latticeconstant = alat)
        n += 1
    grain_c = grain_a.copy()
    if c_space==None:
        c_space = alat/np.sqrt(float(bp[0]**2+bp[1]**2 + bp[2]**2))

    if sum([int(n)%2 for n in bp])%2 == 0 :
        grain_b.positions[:,2]  -= (grain_b.positions[:,2].max() + c_space)
    else:
        grain_b.positions[:,2]  -= (grain_b.positions[:,2].max() + c_space/2.0)

    grain_c.extend(grain_b)
    grain_c.set_cell([grain_c.get_cell()[0,0], grain_c.get_cell()[1,1], 2.*grain_c.get_cell()[2,2]])
    grain_c.positions[:,2] += abs(grain_c.positions[:,2].min())
    #In BCC lattice spacing is different depending on whether the
    #z-plane has an even or odd number of odd miller indices:
    if sum([int(n)%2 for n in bp])%2 == 0 :
        grain_a.positions[:,2] -= (grain_a.positions[:,2].max() + c_space)
    else:
        grain_a.positions[:,2] -= (grain_a.positions[:,2].max() + c_space/2.0)
    grain_c.extend(grain_a)
    grain_c.positions[:,2] += abs(grain_c.positions[:,2].min())
    dups = get_duplicate_atoms(grain_c)
    shared_atoms = [grain_c[dup[0]] for dup in dups]
    for at in shared_atoms:
        print '\t', at.index, at.position
    print '\t There are {} shared atoms'.format(len(shared_atoms))
    #Get sigma number
    grain_d = grain_a.copy()
    grain_d.extend(grain_b)
    dups = get_duplicate_atoms(grain_d)
    shared_atoms_csl = [grain_d[dup[0]] for dup in dups]
    sigma_csl = float(len(grain_a))/float(len(shared_atoms_csl)+len(shared_atoms))
    dups = get_duplicate_atoms(grain_c, delete=True)
    # Space bicrystal.
    if np.allclose(bp, [0,0,1]):
        c_space = np.array(2.83)
    elif np.allclose(bp, [1,1,0]):
        c_space = np.array(2.001)
    elif np.allclose(bp, [1,1,1]):
        c_space = np.array(1.633)
    if sum([int(n)%2 for n in bp])%2 == 0 :
        grain_c.center(vacuum=c_space/2.0, axis=2)
    else:
        grain_c.center(vacuum=c_space/4.0, axis=2)
    z_planes = [round(atom.position[2],4) for atom in shared_atoms]
    z_planes = list(sorted(set(z_planes)))
    if target_dir != None:
        print '\t Writing {0}.xyz to file'.format(gbid)
        write('{0}.xyz'.format(os.path.join(target_dir, gbid)), grain_c)
        return [z_planes, sigma_csl, n_grain_unit, grain_c]
    else:
        return grain_c

def rotate_plane_y(grain, miller):
    """Rotates atoms in grain so that the miller plane
    in grain is parallel to the y axis. Returns the quaternion
    characterizing the rotation which achieves this.

    Args:
      grain(:py:class:`Atoms`): grain atoms object.
      miller(list,int): Vector specifying grain boundary interfacial plane.

    Returns:
      quaternion which acheives this rotation.
    """
    y = np.array([0., 1., 0.])
    p = np.cross(miller, y)
    p *= 1./np.linalg.norm(p)
    #angle between line in plane and desired line:
    theta = np.arccos(miller.dot(y)/(np.linalg.norm(miller)*(np.linalg.norm(y))))
    if theta != theta:
        theta = 0.
        rotation_quaternion = np.array([1., 0., 0., 0.])
        print 'No miller plane specified', theta
    else:
        rotation_quaternion = quat.quaternion_about_axis(theta, p)
        print '\t quaternion: ', rotation_quaternion.round(2)
    print '\t Rotated miller index: ', miller.round(3)
    print '\t Aligning miller plane with y-axis by angle: ', theta*(180./np.pi)
    print '\t around : ', p.round(3)
    print '\t Rotated vector (should be along y): ', rotate_vec(rotation_quaternion, miller).round(3), '\n'
    grain = rotate_grain(grain, q = rotation_quaternion)
    return rotation_quaternion

def gnu_plot_gb(boundary_plane, m, invm, gbid, mb=0.0, invmb=0.0,target_dir='./'):
    """Generate coincident site lattice plots on the fly in svg format for web based visualization.

    Args:
      boundary_plane(list): vector defining interfacial plane.
      m(float): slope of line separating red and blue crystal.
      invm(float): slope of line perpendicular to m.
      gbid(str):grain boundary id.
      mb(float,optional):deprecated.
      invmb(float,optional):deprecated.
      target_dir(str, optional):location to write plotting script.
    """

    f = open(os.path.join(target_dir, 'plot.gnu'), 'w')
    script = " \
    set terminal svg          \n \
    set output 'csl_{1}.svg'  \n \
    h(x) = {3}*x   \n \
    set xr[-10:10] \n \
    set yr[-15:15] \n \
    pl   'grainaT.dat' u 1:2 w p pt 7 ps 1.3 lt 1 t 'Grain A', \\\n\
         'grainaB.dat' u 1:2 w p pt 6 lt 1.0      t '', \\\n\
         'grainbT.dat' u 1:2 w p pt 7 ps 1 lt 3 t 'Grain B',\\\n\
         'grainbB.dat' u 1:2 w p pt 6 ps 1 lt 3 t '',\\\n".format(boundary_plane, gbid, m, invm, mb)
    print >> f, script
    f.close()

def gnu_plot_twist_gb(gbid="000000000", target_dir='./'):
    f = open(os.path.join(target_dir, 'plot.gnu'), 'w')
    script = " \
    set xr[0:15] \n \
    set yr[0:15] \n \
    pl   'grainaT.dat' u 1:2 w p pt 7 ps 1 lt 1 t 'Grain A' \n \
    repl 'grainaB.dat' u 1:2 w p pt 6 lt 1      t ''        \n \
    repl 'grainbT.dat' u 1:2 w p pt 7 ps 1 lt 3 t 'Grain B'   \n \
    repl 'grainbB.dat' u 1:2 w p pt 6 ps 1 lt 3 t ''          \n \
    set terminal svg          \n \
    set output 'csl_{gbid}.svg'  \n \
    repl                      ".format(gbid=gbid)
    print >> f, script
    f.close()

def bcc_csl_nn0(m, n, grain):
    """Rotate grain atom positions according to a rotation matrix generated from a
    quaternion.

    Example for integer m, and orientation axis defined by vector [n n 0],
    the angle of rotation is given by: arccos((m^{2} - 2n^{2})(m^{2} + 2n^{2})).

    Args:
      m(int): Integer of scalar part of quaternion.
      n(int): Integer for vector part.
      grain(:class:`ase.Atoms`): grain boundary structure.
    """

# Angle from zeiner
# 100, \psi = (m**2 - n**2)/(m**2-n**2)
# 110, \psi = (m**2 - 2*n**2)/(m**2-n**2)
# 111, \psi = (m**2 - 3*n**2)/(m**2-n**2)
    R = zeiner_matrix(np.array([m,n,n,0]))
    for i in range(len(grain)):
        grain[i].position = np.dot(R, grain[i].position)

def zeiner_matrix(q):
    """Generate a rotation mater from a quaternion.

    Args:
      q(quaternion): 4 vector representation of a quaternion.

    Returns:
      Rotation matrix generated from a quaternion.
    """

    r2 = q.dot(q)
    k = q[0]
    l = q[1]
    m = q[2]
    n = q[3]
    R = np.array([
          [ k*k+l*l-m*m-n*n,   -2*k*n + 2*l*m,    2*k*m + 2*l*n],
          [   2*k*n + 2*l*m,  k*k-l*l+m*m-n*n, -2*k*l + 2*m*n],
          [  -2*k*m + 2*l*n, 2*k*l + 2*m*n,  k*k-l*l-m*m + n*n]
          ])
    R = (1./r2)*R.T
    return R

def csl_lattice_vecs(m,n):
    """Returns lattice vectors defining a coincident site lattice
    from quaternion following Zeiner(05).
    """
    r = np.array([m,n,n,0])
    r0 = np.array([r[1], r[2],  r[3]])
    r1 = np.array([r[0], r[3], -r[2]])
    r2 = np.array([-r[3], r[0], r[1]])
    r3 = np.array([r[2], -r[1], r[0]])
    if np.mod(r.dot(r),2)==1:
        print '\t |r|^{2} is odd'
        a = r0
        b = r1
        c = r2
        d = 0.5*(r0 + r1+ r2 +r3)
    elif (np.mod(r.dot(r), 2) == 0 and np.mod(r.dot(r),4) != 0):
        print '\t |r|^{2}  is divisible by 2 but not 4'
        a = r0
        b = 0.5*(r0 +r1)
        c = 0.5*(r0+r2)
        d = 0.5*(r0 + r3)
    else:
        print '\t |r|^{2}  is divisible 4'
        a = 0.5*r0
        b = 0.5*(r1)
        c = 0.5*(r2)
        d = 0.5*(r3)
    return np.array([a,b,c,d])

def rotate_vec(q, vec):
    """Rotate 3 vector with quaternion: qvq^{-1}.

    Args:
      q(quaternion): rotation quaternion.
      vec(3x1 list): vector to rotate.

    Returns:
      Rotated 3 vector.
    """

    vec = np.array([0., vec[0], vec[1], vec[2]])
    qm = quat.quaternion_conjugate(q)
    pos_prime = quat.quaternion_multiply(q, vec)
    pos_prime = quat.quaternion_multiply(pos_prime, qm)
    vec = quat.quaternion_imag(pos_prime)
    return vec

def rotate_grain(grain, theta=0., x=[0.,0.,0.], q=[]):
    """Rotate the grain according to quaternion=[Theta, u,v,w]=[w,x,y,z].
    Standard routine is passed angle and vector this generates the quaternion to do the rotation
    however if a quaternion, q!=None, is passed the plane is rotated using q as: qvq^{-1}.

    Args:
      grain(:class:`ase.Atoms`): Grain boundary to rotate.
      theta(float): Angle to rotate vector.
      x(list[int]): Three vector defining orientation vector and vector component of quaternion.

    Returns:
      Rotated grain.
    """

    if q==[]:
        q = quat.quaternion_about_axis(theta, x)
        qm = quat.quaternion_conjugate(q)
    else:
        qm = quat.quaternion_conjugate(q)
    for i in range(len(grain)):
        pos       = np.array([0, grain[i].position[0], grain[i].position[1], grain[i].position[2]])
        pos_prime = quat.quaternion_multiply(q, pos)
        pos_prime = quat.quaternion_multiply(pos_prime, qm)
        grain[i].position = quat.quaternion_imag(pos_prime)
    return grain

def print_points(atoms, f, top_grain=True, gb_type='tilt'):
    """Print positions of atoms to file.

    Args:
      atoms(ase.Atoms): list of :class:`ase.Atom` object to print positions.
      f(file): file handle to print position list to.
    """

    if gb_type == 'tilt':
        for atom in atoms:
            if top_grain:
                if atom.position[0]>=0.0:
                    print>>f, '{0:12.7f} {1:12.7f} {2:12.7f}'.format(atom.position[0], atom.position[1], atom.position[2])
            else:
                if atom.position[0]<=0.0:
                    print>>f, '{0:12.7f} {1:12.7f} {2:12.7f}'.format(atom.position[0], atom.position[1], atom.position[2])
    else:
        for atom in atoms:
            print>>f, '{0:12.7f} {1:12.7f} {2:12.7f}'.format(atom.position[0], atom.position[1], atom.position[2])


def simplify_csl(m, b=0.0, target_dir='./'):
    """Simplify the dat files of the CSL so that atoms in grain_a are below the line
    defined by the boundary plane in the current projection and atoms from grain_b
    are above the line.

    Args:
      m(float): slope of the line.
      b(float): y-intercept.
      target_dir(str): target direcotry string.
    """

    graina_list = ['grainaT.dat', 'grainaB.dat']
    grainb_list = ['grainbT.dat', 'grainbB.dat']
    for name in graina_list:
        f = open('{0}'.format(os.path.join(target_dir, name)), 'r+')
        grain_a = f.read().split('\n')
        f.close()
        f = open('{0}'.format(os.path.join(target_dir, name)), 'w')
        del(grain_a[-1])
        for at in grain_a:
            position = np.array(map(float, at.split()))
            print >> f, '{0:12.7f} {1:12.7f} {2:12.7f}'.format(position[0], position[1], position[2])
        f.close()
    for name in grainb_list:
        f = open('{0}'.format(os.path.join(target_dir, name)),'r+')
        grain_b = f.read().split('\n')
        f.close()
        f = open('{0}'.format(os.path.join(target_dir, name)),'w')
        del(grain_b[-1])
        for at in grain_b:
            position = np.array(map(float, at.split()))
            print >> f, '{0:12.7f} {1:12.7f} {2:12.7f}'.format(position[0], position[1], position[2])
        f.close()

def find_densest_plane(grain_dict):
    """Return the indices for a set of adjacent planes with the largest number of points.
    Used in :func:`gen_csl`.

    Args:
      grain_dict(dict): Grain dictionary.

    Returns:
      float, float: key1 and key2 floating point values representing the z-coordinates of the
      two adjacent planes along z-axis with the most Atoms on them.
    """
    maxx     = max([len(a) for a in grain_dict.values()])
    len_keys = dict([(x,len(y)) for x,y in grain_dict.items()])
    keys_2   = [x for x,y in grain_dict.items()]
    keys_2   = sorted(keys_2)
    keys     = [x for x,y in grain_dict.items() if len(y) == maxx]
    print '\t Largest number of points: ', maxx, 'Number of z-planes: ', len(keys)
    key1     = keys[0]
    z_below  = keys_2.index(key1)-1
    key2     = keys_2[z_below]
    return key1, key2

def gen_csl(theta, orientation_axis, boundary_plane, target_dir='./', gbid='0000000000',
            alat=2.83, chem_symbol='Fe', gb_type="tilt"):
    """Generate the coincident site lattice representations.

    Args:
      theta(float): misorientation angle in radians.
      orientation_axis(list): orientation axis.
      boundary_plane(list[int]): boundary plane normal vector.
      target_dir(str): location to store coincident site lattice files.
      gbid(str): Grain boundary id.
      alat(float): Magnitude lattice vector.
      chem_symbol(str): Chemical symbol.
      gb_type(str): boundary type [tilt, twist] (Default: 'tilt').
    """
    grain = crystal(chem_symbol, [(0,0,0)], spacegroup=229,
                 cellpar=[alat, alat, alat, 90, 90, 90], size=[10,10,10])
    grain_a = grain.copy()
    grain_b = grain.copy()
    for i in range(3):
        grain_a.positions[:,i] -= 10*alat/2.
        grain_b.positions[:,i] -= 10*alat/2.

    rotm = quat.rotation_matrix(theta, orientation_axis)
    rotquat = quat.quaternion_from_matrix(rotm).round(8)
    angle_str = str(round((theta*180./np.pi),2)).replace('.', '')

#Truncate the angle string if it is greater than 4 characters
    if len(angle_str) > 4:
        angle_str = angle_str[:-1]
    elif len(angle_str) < 4:
        angle_str = angle_str + '0'

    print '\t Grain Boundary ID: ', gbid
    print '\t Rotation Axis: ', orientation_axis, 'Rotation Angle: ', (round(theta,6))*(180./np.pi), round(theta,4)
    print '\t Rotation quaternion: ', rotquat
    print '\t Integral Rotation quaternion: ', (1./(np.array([a for a in rotquat if a !=0]).min())*rotquat).round(5)
    print '\t Boundary plane grain A coordinate system: ', boundary_plane.round(3)

    rotm           = quat.rotation_matrix(theta, orientation_axis)
    rotmquat       = (1./(np.array([a for a in rotquat if a !=0.]).min())*rotquat).round(5)

    if np.allclose(orientation_axis, [1,1,0]):
        planequat_1    = np.array([0,0,0,1])
        planequat_2    = np.array([0,1,-1,0])
    elif np.allclose(orientation_axis, [0,0,1]):
        planequat_1    = np.array([0,0,1,0])
        planequat_2    = np.array([0,1,0,0])
    elif np.allclose(orientation_axis, [1,1,1]):
        planequat_1    = np.array([0,1,-1,0])
        planequat_2    = np.array([0,1,1,-2])

    print '\t These vectors satisfy the conditions for a Symmetric Boundary Plane: '
    n1 = quat.quaternion_multiply(planequat_1, rotmquat)
    n2 = quat.quaternion_multiply(planequat_2, rotmquat)
    comm_denom = []
    for a in n1:
        if (a%1).round(1) != 0:
            comm_denom.append(1./(np.abs(a)%1))
    comm_denom = np.array(comm_denom)
    if len(comm_denom)!=0:
        print '\t {}'.format((comm_denom.max()*n1).round(2))
        print '\t {}'.format((comm_denom.max()*n2).round(2))
        print '\n'
        print '\n'
    else:
        print '\t', n1
        print '\t', n2
        print '\n'
        print '\n'


    if theta != 0:
        m2 = 2*(1.+np.cos(theta))/(1.-np.cos(theta))
    else:
        m2 = 0

    if theta !=0:
        m = np.sqrt(m2).round(0)
        n = 1
    else:
        m = 0
        n = 0

    if gb_type=="tilt":
        csl_tilt_factory(orientation_axis, boundary_plane, m, n, gbid, grain_a, grain_b,
                    theta=theta, target_dir=target_dir, gb_type=gb_type)
    elif gb_type=="twist":
        csl_twist_factory(orientation_axis, boundary_plane, gbid, target_dir, gb_type=gb_type)

def csl_twist_factory(bp, v, gbid, target_dir, crystal_type='BCC', gb_type='twist'):
    """
    Generate coincident site lattice points for arbitraty twist boundary.

    Args:
      bp(list): Boundary plane.
      v(list): Orientation axis.
      gbid(str): Grain boundary id.
      target_dir(str): target_directory string.
      crystal_type(str): BCC

    """

    bpxv    = [(bp[1]*v[2]-v[1]*bp[2]), (bp[2]*v[0]-bp[0]*v[2]), (bp[0]*v[1]- v[0]*bp[1])]

    if crystal_type=='BCC':
        grain_a =  BodyCenteredCubic(directions = [list(v), list(bpxv), list(bp)],
                                     size = (1,1,1), symbol='Fe', pbc=(1,1,1),
                                     latticeconstant = 2.83)
    else:
        sys.exit('Crystal type not yet supported.')

    n_grain_unit = len(grain_a)
    n = 2

    v2 = v.copy()
    if np.allclose(bp, [0,0,1]):
        v2[0] = -v2[0]
    elif np.allclose(bp, [1,1,0]):
        v2[2]   = -v2[2]
    elif np.allclose(bp, [1,1,1]):
        v2[1]   = -v2[1]
        v2[2]   = -v2[2]

    bpxv = [(bp[1]*v2[2]-v2[1]*bp[2]),(bp[2]*v2[0]-bp[0]*v2[2]),(bp[0]*v2[1]- v2[0]*bp[1])]
    grain_b = BodyCenteredCubic(directions = [list(v2), list(bpxv), list(bp)],
                                size = (1,1,1), symbol='Fe', pbc=(1,1,1),
                                latticeconstant = 2.83)

    dat_files = ['grainaT.dat', 'grainaB.dat']
    grain_a = sorted(grain_a, key=lambda x: (x.position[2], x.position[0], x.position[1]))
    z_vals = []
    for grain in grain_a:
        z_vals.append(round(grain.position[2],3))
    z_vals = list(set(z_vals))
    grain_dict = {}
    for z in z_vals:
        grain_dict[z] = []
    for grain in grain_a:
        grain_dict[round(grain.position[2], 3)].append(grain)

    for key, dat_file in zip(grain_dict.keys()[:2], dat_files):
        with open(os.path.join(target_dir, dat_file), 'w') as f:
            print_points(grain_dict[key], f, gb_type=gb_type)

    dat_files = ['grainbT.dat', 'grainbB.dat']
    grain_b = sorted(grain_b, key=lambda x: (x.position[2], x.position[0], x.position[1]))
    for grain in grain_b:
        z_vals.append(round(grain.position[2],3))
    z_vals = list(set(z_vals))
    grain_dict = {}
    for z in z_vals:
        grain_dict[z] = []
    for grain in grain_b:
        grain_dict[round(grain.position[2], 3)].append(grain)

    for key, dat_file in zip(grain_dict.keys()[:2], dat_files):
        with open(os.path.join(target_dir, dat_file), 'w') as f:
            print_points(grain_dict[key], f, gb_type=gb_type)
    gnu_plot_twist_gb(gbid=gbid, target_dir=target_dir)

def zeiner_info():
    print '\t Using Zeiner'
    print '\t Spanning Vectors of CSL'
    print '\t', csl_lattice_vecs(m,n)
    print ''
    if (m!=0 and n!=0):
        bcc_csl_nn0(m, n, grain_b)
# A rotation is a coincidence rotation if the rotation
# matrix is orthogonal and has rational entries.
    if (m!=0 and n !=0):
        rotm = zeiner_matrix(np.array([m,n,n,0]))
        print rotm
    else:
        rotm = np.identity(3)
    if m!=0 and n!=0:
        mn = float(m*m-2*n*n)/float(m*m+2*n*n)
    else:
        mn = 0
    angle_of_rotation = np.arccos(mn)
    deg = round(angle_of_rotation*(180./np.pi),2)
    print m, n
    print '\n Angle of rotation: ', deg, 'or ', 180.-deg, ' Orientation Axis: \n', (n, n, 0), '\n'


def csl_tilt_factory(orientation_axis, boundary_plane, m, n, gbid, grain_a, grain_b,
                     theta=0, target_dir='./', gb_type='tilt'):
    """ Builds a plot of the coincident site lattice oriented along a suitable
    plane for viewing purposes. In the gnuplot plane we think of x,y,z orientation.
    so that the orientation axis = N = z is oriented 'into' the page.
    Vector specifying 0 Angle (for the case of tilt) = v = x
    y = Nxv.

    :func:`rotate_plane_z` takes a grain and rotates it
    so that the orientation axis of the grain boundary with
    respect to grain a is orthogonal to the x-y plane. The
    quaternion to accomplish this rotation is stored
    in plane_quaternion_z.

    Args:
      orientation_axis(list): orientation axis
      boundary_plane(list): boundary plane
      m(int): slope of boundary plane
      n(int): integer of orientation axis.
      gbid(str): grain boundary identifying label.
      grain_a(:class:`ase.Atoms`): blue crystal.
      grain_b(:class:`ase.Atoms`): red crystal.
      theta(float): angle.
      target_dir(str): target directory to write plotting scripts.
      gb_type(str): grain boundary type.
    """

    f = open(os.path.join(target_dir, 'grainaT.dat'), 'w')
    g = open(os.path.join(target_dir, 'grainaB.dat'), 'w')
    if np.allclose(orientation_axis, [0,0,1]):
        print 'Axis already aligned along z'
        plane_quaternion_z = np.array([1,0,0,0])
    else:
        plane_quaternion_z = rotate_plane_z(grain_a, (-1.*np.array(orientation_axis)))

    print '\t boundary plane', boundary_plane.round(2)
# plane_quaternion_x stores the rotation quaternion that rotates the
# boundary plane normal in grain_a coordinates so that it is perpendicular to the x-axis.
    plane_quaternion_x = rotate_plane_y(grain_a, rotate_vec(plane_quaternion_z, boundary_plane))
    print '\t boundary plane', boundary_plane.round(2)
# The following code finds the planes with largest number
# of points for viewing purposes:
# First we sort grain_a by x, y, z
    grain_a = sorted(grain_a, key=lambda x: (x.position[2], x.position[0], x.position[1]))
    z_vals = []
# The loop over each position in grain_a and store the z-value
# in a list.
    for grain in grain_a:
        if round(grain.position[2], 2) not in [round(z,2) for z in z_vals[:]]:
            z_vals.append(round(grain.position[2],2))
    grain_dict = {}
# initialize dictionary of lists
    for z in z_vals:
        grain_dict[z] = []
# append the atoms to each position in the dictionary.
    for grain in grain_a:
        grain_dict[round(grain.position[2], 2)].append(grain)
# create a list of z-values and the number of points for each z value
    len_list = []
    for z in z_vals:
        len_list.append((z, len(grain_dict[z])))

    key1, key2 = find_densest_plane(grain_dict)
    print_points(grain_dict[key1], f, top_grain=True, gb_type=gb_type)
    print_points(grain_dict[key2], g, top_grain=True, gb_type=gb_type)
    f.close()
    g.close()
# Setup grain b and project into chosen plane
    f = open(os.path.join(target_dir, 'grainbT.dat'),'w')
    g = open(os.path.join(target_dir, 'grainbB.dat'),'w')
#Theta = np.arccos(float(m-2*n*n)/float(m+2*n*n))
    rotate_grain(grain_b, -theta, orientation_axis)
    deg = round(theta*(180./np.pi), 2)
    print '\n Angle of rotation: ', deg, 'or ', 180.-deg, ' Orientation Axis: ', orientation_axis, '\n'

    boundary_plane = rotate_vec(plane_quaternion_z, boundary_plane)
    boundary_plane = rotate_vec(plane_quaternion_x, boundary_plane)
    print ''
    print 'Boundary plane: ', boundary_plane.round(3)
# Draw a line on the graph which is the
# equation of the line defining the miller index
# and a line perpendicular to this which corresponds
# to a line in the plane.
# m = (boundary_plane[0]/boundary_plane[1])
# invm = -(1./(boundary_plane[0]/boundary_plane[1]))
# Grain boundary should be parallel to \hat{x}
    m    = 0.
    invm = 0.
#
# The boundary_plane vector is the miller index
# of the grain boundary with respect to initial crystal
# for grain_b (blue crystal) we rotate this around orientation axis
# by the orientation angle.
#
    boundary_plane_b = boundary_plane
    rotm = zeiner_matrix(np.array([m,n,n,0]))
    boundary_plane_b = np.dot(rotm, boundary_plane_b)
    print '\t Rotated Grain Boundary:'
    boundary_plane_b = rotate_vec(plane_quaternion_z, boundary_plane_b)
    boundary_plane_b = rotate_vec(plane_quaternion_x, boundary_plane_b)
    print '\t', boundary_plane_b
    mb    = 0.
    invmb = 0.
    gnu_plot_gb('Plane', m, invm, gbid, mb=mb, invmb=invmb, target_dir=target_dir)
    rotate_grain(grain_b, q=plane_quaternion_z)
    rotate_grain(grain_b, q=plane_quaternion_x)
    grain_b = sorted(grain_b, key=lambda x: (x.position[2], x.position[0], x.position[1]))
    z_vals  = []
    for grain in grain_b:
        if round(grain.position[2], 2) not in [round(z,2) for z in z_vals[:]]:
            z_vals.append(round(grain.position[2],2))
    grain_dict = {}
    for z in z_vals:
        grain_dict[z] = []
    for grain in grain_b:
        grain_dict[round(grain.position[2], 2)].append(grain)
    len_list = []
    for z in z_vals:
        len_list.append((z,len(grain_dict[z])))
    key1, key2 = find_densest_plane(grain_dict)
    print_points(grain_dict[key1], f, top_grain=False, gb_type=gb_type)
    print_points(grain_dict[key2], g, top_grain=False, gb_type=gb_type)
    f.close()
    g.close()

def gen_canonical_grain_dir(angle, orientation_axis, boundary_plane, material='alphaFe', target_dir='./', gb_type="tilt"):
    """Generate a canonical grain boundary directory.
    Args:
      angle(float): misorientation angle in radians.
      orienation_axis(list): orientation axis.
      boundary_plane(list): boundary plane
      material(str, optional): material
      target_dir(str, optional): target directory to deposit canonical grain structure.
      gb_type(str): grain boundary type options are 'tilt', 'twist'.
    """

    angle_str = str(round((angle*180./np.pi),2)).replace('.', '')
    if len(angle_str) > 4:
        angle_str = angle_str[:-1]
    elif len(angle_str) < 4:
        angle_str = angle_str + '0'

    if gb_type == 'tilt':
        gbid = '{0}{1}{2}'.format(orientation_axis[0], orientation_axis[1], orientation_axis[2])\
                                + angle_str \
                                + '{0}{1}{2}'.format(int(abs(boundary_plane[0])),
                                                     int(abs(boundary_plane[1])),
                                                     int(abs(boundary_plane[2])))
    elif gb_type == 'twist':
        gbid = '{0}{1}{2}'.format(orientation_axis[0], orientation_axis[1], orientation_axis[2])\
                                + angle_str \
                                + '{0}{1}{2}'.format(int(abs(orientation_axis[0])),
                                                     int(abs(orientation_axis[1])),
                                                     int(abs(orientation_axis[2])))
    else:
        sys.exit("gb_type not recognized.")

    print '\t Grain Boundary ID',  gbid
    target_dir = os.path.join(target_dir, gbid)

    print '\t Grain Boundary Dir', target_dir
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    else:
        'File already exists.'

    gen_csl(angle, orientation_axis, boundary_plane, target_dir=target_dir, gbid=gbid, gb_type=gb_type)

    if gb_type == 'tilt':
        zplanes, dups, nunitcell, grain = build_tilt_sym_gb(gbid, bp=boundary_plane, v=orientation_axis,
                                                            c_space=None, target_dir=target_dir,
                                                            rbt=[0.0, 0.0])
        sigma = float(nunitcell)/float(dups)
    elif gb_type =='twist':
    #swap or_axis, and bp_plane for twist boundary:
        zplanes, dups, nunitcell, grain = build_twist_sym_gb(gbid, bp=orientation_axis, v=boundary_plane,
                                                             c_space=None, target_dir=target_dir,
                                                             rbt=[0.0, 0.0])
        n_at, dups, sigma = calc_twist_sigma_csl(orientation_axis, boundary_plane, chem_symbol="Fe")
    else:
        sys.exit("Unknown gb_type.")

    cell = grain.get_cell()
    A = cell[0][0]*cell[1][1]
    H = cell[2][2]
    if gb_type == 'tilt':
        gb_dict = {"gbid"  : gbid, "boundary_plane" : list(boundary_plane),
                   "orientation_axis" : list(orientation_axis),
                   "type": "symmetric {} boundary".format(gb_type),
                   "angle": angle, "zplanes" : zplanes, "coincident_sites": dups,
                   "sigma_csl": sigma, "n_at" : nunitcell, 'A':A, 'area':A, 'H':H} #area should become dominant keyword
    elif gb_type == 'twist':
        gb_dict = {"gbid"  : gbid, "boundary_plane" : list(orientation_axis),
                   "orientation_axis" : list(orientation_axis),
                   "type": "symmetric {} boundary".format(gb_type),
                   "angle": angle, "zplanes" : zplanes, "coincident_sites": dups,
                   "sigma_csl": sigma, "n_at" : nunitcell, 'A':A, 'area':A, 'H':H} #area should become dominant keyword
    else:
        sys.exit("unrecognized grain boundary type")

    with open(os.path.join(target_dir, 'gb.json'), 'w') as outfile:
        json.dump(gb_dict, outfile, indent=2)

if __name__=='__main__':
# BCC Iron unit cell
# These lists shouldn't be necessary if we exploit
# the relationship between the rotation quaternion
# and the tilt symmetric mirror plane.
# 0 Degrees is measured from 001
    sym_tilt_100 = [[np.pi*(7.15/180.), np.array([1.0, 16.0, 0.0])],
                    [np.pi*(7.63/180.), np.array([1.0, 15.0, 0.0])],
                    [np.pi*(8.17/180.), np.array([1.0, 14.0, 0.0])],
                    [np.pi*(8.8/180.), np.array([1.0, 13.0, 0.0])],
                    [np.pi*(9.53/180.), np.array([1.0, 12.0, 0.0])],
                    [np.pi*(10.39/180.), np.array([1.0, 11.0, 0.0])],
                    [np.pi*(11.42/180.), np.array([1.0, 10.0, 0.0])],
                    [np.pi*(12.68/180.), np.array([1.0, 9.0, 0.0])],
                    [np.pi*(14.25/180.), np.array([1.0, 8.0, 0.0])],
                    [np.pi*(15.19/180.), np.array([2.0, 15.0, 0.0])],
                    [np.pi*(16.26/180.), np.array([1.0, 7.0, 0.0])],
                    [np.pi*(17.49/180.), np.array([2.0, 13.0, 0.0])],
                    [np.pi*(18.92/180.), np.array([1.0, 6.0, 0.0])],
                    [np.pi*(20.61/180.), np.array([2.0, 11.0, 0.0])],
                    [np.pi*(21.24/180.), np.array([3.0, 16.0, 0.0])],
                    [np.pi*(22.62/180.), np.array([1.0, 5.0, 0.0])],
                    [np.pi*(24.19/180.), np.array([3.0, 14.0, 0.0])],
                    [np.pi*(25.06/180.), np.array([2.0, 9.0, 0.0])],
                    [np.pi*(25.99/180.), np.array([3.0, 13.0, 0.0])],
                    [np.pi*(28.07/180.), np.array([1.0, 4.0, 0.0])],
                    [np.pi*(30.51/180.), np.array([3.0, 11.0, 0.0])],
                    [np.pi*(31.89/180.), np.array([2.0, 7.0, 0.0])],
                    [np.pi*(33.4/180.), np.array([3.0, 10.0, 0.0])],
                    [np.pi*(34.21/180.), np.array([4.0, 13.0, 0.0])],
                    [np.pi*(34.71/180.), np.array([5.0, 16.0, 0.0])],
                    [np.pi*(36.87/180.), np.array([1.0, 3.0, 0.0])],
                    [np.pi*(39.31/180.), np.array([5.0, 14.0, 0.0])],
                    [np.pi*(41.11/180.), np.array([3.0, 8.0, 0.0])],
                    [np.pi*(43.6/180.), np.array([2.0, 5.0, 0.0])],
                    [np.pi*(45.24/180.), np.array([5.0, 12.0, 0.0])],
                    [np.pi*(46.4/180.), np.array([3.0, 7.0, 0.0])],
                    [np.pi*(47.26/180.), np.array([7.0, 16.0, 0.0])],
                    [np.pi*(47.92/180.), np.array([4.0, 9.0, 0.0])],
                    [np.pi*(48.89/180.), np.array([5.0, 11.0, 0.0])],
                    [np.pi*(49.55/180.), np.array([6.0, 13.0, 0.0])],
                    [np.pi*(50.03/180.), np.array([7.0, 15.0, 0.0])],
                    [np.pi*(53.13/180.), np.array([1.0, 2.0, 0.0])],
                    [np.pi*(58.11/180.), np.array([5.0, 9.0, 0.0])],
                    [np.pi*(61.93/180.), np.array([3.0, 5.0, 0.0])],
                    [np.pi*(64.94/180.), np.array([7.0, 11.0, 0.0])],
                    [np.pi*(67.38/180.), np.array([2.0, 3.0, 0.0])],
                    [np.pi*(69.39/180.), np.array([9.0, 13.0, 0.0])],
                    [np.pi*(71.08/180.), np.array([5.0, 7.0, 0.0])],
                    [np.pi*(72.51/180.), np.array([11.0, 15.0, 0.0])],
                    [np.pi*(73.74/180.), np.array([3.0, 4.0, 0.0])],
                    [np.pi*(75.75/180.), np.array([7.0, 9.0, 0.0])],
                    [np.pi*(77.32/180.), np.array([4.0, 5.0, 0.0])],
                    [np.pi*(78.58/180.), np.array([9.0, 11.0, 0.0])],
                    [np.pi*(79.61/180.), np.array([5.0, 6.0, 0.0])],
                    [np.pi*(80.47/180.), np.array([11.0, 13.0, 0.0])],
                    [np.pi*(81.2/180.), np.array([6.0, 7.0, 0.0])],
                    [np.pi*(81.83/180.), np.array([13.0, 15.0, 0.0])],
                    [np.pi*(82.37/180.), np.array([7.0, 8.0, 0.0])],
                    [np.pi*(83.27/180.), np.array([8.0, 9.0, 0.0])],
                    [np.pi*(83.97/180.), np.array([9.0, 10.0, 0.0])],
                    [np.pi*(84.55/180.), np.array([10.0, 11.0, 0.0])],
                    [np.pi*(85.02/180.), np.array([11.0, 12.0, 0.0])],
                    [np.pi*(85.42/180.), np.array([12.0, 13.0, 0.0])],
                    [np.pi*(85.76/180.), np.array([13.0, 14.0, 0.0])],
                    [np.pi*(86.05/180.), np.array([14.0, 15.0, 0.0])],
                    [np.pi*(86.3/180.), np.array([15.0, 16.0, 0.0])],
                    [np.pi*(90.0/180.), np.array([1.0, 1.0, 0.0])]]

# 0 degrees is at [1-10] rotation by 180 would set
# the zero of the angle as [-110].
    sym_tilt_110 = [[np.pi*(10.1/180.), np.array([-1.0, 1.0, 16.0])],
                    [np.pi*(10.77/180.), np.array([-1.0, 1.0, 15.0])],
                    [np.pi*(11.54/180.), np.array([-1.0, 1.0, 14.0])],
                    [np.pi*(12.42/180.), np.array([-1.0, 1.0, 13.0])],
                    [np.pi*(13.44/180.), np.array([-1.0, 1.0, 12.0])],
                    [np.pi*(14.65/180.), np.array([-1.0, 1.0, 11.0])],
                    [np.pi*(16.1/180.), np.array([-1.0, 1.0, 10.0])],
                    [np.pi*(17.86/180.), np.array([-1.0, 1.0, 9.0])],
                    [np.pi*(20.05/180.), np.array([-1.0, 1.0, 8.0])],
                    [np.pi*(21.36/180.), np.array([-2.0, 2.0, 15.0])],
                    [np.pi*(22.84/180.), np.array([-1.0, 1.0, 7.0])],
                    [np.pi*(24.55/180.), np.array([-2.0, 2.0, 13.0])],
                    [np.pi*(26.53/180.), np.array([-1.0, 1.0, 6.0])],
                    [np.pi*(28.84/180.), np.array([-2.0, 2.0, 11.0])],
                    [np.pi*(29.7/180.), np.array([-3.0, 3.0, 16.0])],
                    [np.pi*(31.59/180.), np.array([-1.0, 1.0, 5.0])],
                    [np.pi*(33.72/180.), np.array([-3.0, 3.0, 14.0])],
                    [np.pi*(34.89/180.), np.array([-2.0, 2.0, 9.0])],
                    [np.pi*(36.15/180.), np.array([-3.0, 3.0, 13.0])],
                    [np.pi*(38.94/180.), np.array([-1.0, 1.0, 4.0])],
                    [np.pi*(42.18/180.), np.array([-3.0, 3.0, 11.0])],
                    [np.pi*(44.0/180.), np.array([-2.0, 2.0, 7.0])],
                    [np.pi*(45.98/180.), np.array([-3.0, 3.0, 10.0])],
                    [np.pi*(47.03/180.), np.array([-4.0, 4.0, 13.0])],
                    [np.pi*(47.69/180.), np.array([-5.0, 5.0, 16.0])],
                    [np.pi*(50.48/180.), np.array([-1.0, 1.0, 3.0])],
                    [np.pi*(53.59/180.), np.array([-5.0, 5.0, 14.0])],
                    [np.pi*(55.88/180.), np.array([-3.0, 3.0, 8.0])],
                    [np.pi*(58.99/180.), np.array([-2.0, 2.0, 5.0])],
                    [np.pi*(61.02/180.), np.array([-5.0, 5.0, 12.0])],
                    [np.pi*(62.44/180.), np.array([-3.0, 3.0, 7.0])],
                    [np.pi*(63.49/180.), np.array([-7.0, 7.0, 16.0])],
                    [np.pi*(64.3/180.), np.array([-4.0, 4.0, 9.0])],
                    [np.pi*(65.47/180.), np.array([-5.0, 5.0, 11.0])],
                    [np.pi*(66.27/180.), np.array([-6.0, 6.0, 13.0])],
                    [np.pi*(66.85/180.), np.array([-7.0, 7.0, 15.0])],
                    [np.pi*(70.53/180.), np.array([-1.0, 1.0, 2.0])],
                    [np.pi*(76.31/180.), np.array([-5.0, 5.0, 9.0])],
                    [np.pi*(80.63/180.), np.array([-3.0, 3.0, 5.0])],
                    [np.pi*(83.97/180.), np.array([-7.0, 7.0, 11.0])],
                    [np.pi*(86.63/180.), np.array([-2.0, 2.0, 3.0])],
                    [np.pi*(88.79/180.), np.array([-9.0, 9.0, 13.0])],
                    [np.pi*(90.58/180.), np.array([-5.0, 5.0, 7.0])],
                    [np.pi*(92.09/180.), np.array([-11.0, 11.0, 15.0])],
                    [np.pi*(93.37/180.), np.array([-3.0, 3.0, 4.0])],
                    [np.pi*(95.45/180.), np.array([-7.0, 7.0, 9.0])],
                    [np.pi*(97.05/180.), np.array([-4.0, 4.0, 5.0])],
                    [np.pi*(98.33/180.), np.array([-9.0, 9.0, 11.0])],
                    [np.pi*(99.37/180.), np.array([-5.0, 5.0, 6.0])],
                    [np.pi*(100.23/180.), np.array([-11.0, 11.0, 13.0])],
                    [np.pi*(100.96/180.), np.array([-6.0, 6.0, 7.0])],
                    [np.pi*(101.58/180.), np.array([-13.0, 13.0, 15.0])],
                    [np.pi*(102.12/180.), np.array([-7.0, 7.0, 8.0])],
                    [np.pi*(103.0/180.), np.array([-8.0, 8.0, 9.0])],
                    [np.pi*(103.69/180.), np.array([-9.0, 9.0, 10.0])],
                    [np.pi*(104.25/180.), np.array([-10.0, 10.0, 11.0])],
                    [np.pi*(104.71/180.), np.array([-11.0, 11.0, 12.0])],
                    [np.pi*(105.09/180.), np.array([-12.0, 12.0, 13.0])],
                    [np.pi*(105.42/180.), np.array([-13.0, 13.0, 14.0])],
                    [np.pi*(105.7/180.), np.array([-14.0, 14.0, 15.0])],
                    [np.pi*(105.95/180.), np.array([-15.0, 15.0, 16.0])],
                    [np.pi*(109.47/180.), np.array([-1.0, 1.0, 1.0])],
                    [np.pi*(112.92/180.), np.array([-16.0, 16.0, 15.0])],
                    [np.pi*(113.15/180.), np.array([-15.0, 15.0, 14.0])],
                    [np.pi*(113.42/180.), np.array([-14.0, 14.0, 13.0])],
                    [np.pi*(113.73/180.), np.array([-13.0, 13.0, 12.0])],
                    [np.pi*(114.1/180.), np.array([-12.0, 12.0, 11.0])],
                    [np.pi*(114.53/180.), np.array([-11.0, 11.0, 10.0])],
                    [np.pi*(115.05/180.), np.array([-10.0, 10.0, 9.0])],
                    [np.pi*(115.7/180.), np.array([-9.0, 9.0, 8.0])],
                    [np.pi*(116.51/180.), np.array([-8.0, 8.0, 7.0])],
                    [np.pi*(117.0/180.), np.array([-15.0, 15.0, 13.0])],
                    [np.pi*(117.56/180.), np.array([-7.0, 7.0, 6.0])],
                    [np.pi*(118.21/180.), np.array([-13.0, 13.0, 11.0])],
                    [np.pi*(118.98/180.), np.array([-6.0, 6.0, 5.0])],
                    [np.pi*(119.9/180.), np.array([-11.0, 11.0, 9.0])],
                    [np.pi*(121.01/180.), np.array([-5.0, 5.0, 4.0])],
                    [np.pi*(122.38/180.), np.array([-9.0, 9.0, 7.0])],
                    [np.pi*(124.12/180.), np.array([-4.0, 4.0, 3.0])],
                    [np.pi*(125.18/180.), np.array([-15.0, 15.0, 11.0])],
                    [np.pi*(126.41/180.), np.array([-7.0, 7.0, 5.0])],
                    [np.pi*(127.83/180.), np.array([-13.0, 13.0, 9.0])],
                    [np.pi*(129.52/180.), np.array([-3.0, 3.0, 2.0])],
                    [np.pi*(131.55/180.), np.array([-11.0, 11.0, 7.0])],
                    [np.pi*(134.02/180.), np.array([-5.0, 5.0, 3.0])],
                    [np.pi*(137.11/180.), np.array([-9.0, 9.0, 5.0])],
                    [np.pi*(141.06/180.), np.array([-2.0, 2.0, 1.0])],
                    [np.pi*(143.48/180.), np.array([-15.0, 15.0, 7.0])],
                    [np.pi*(143.85/180.), np.array([-13.0, 13.0, 6.0])],
                    [np.pi*(144.36/180.), np.array([-11.0, 11.0, 5.0])],
                    [np.pi*(145.11/180.), np.array([-9.0, 9.0, 4.0])],
                    [np.pi*(145.62/180.), np.array([-16.0, 16.0, 7.0])],
                    [np.pi*(146.28/180.), np.array([-7.0, 7.0, 3.0])],
                    [np.pi*(147.17/180.), np.array([-12.0, 12.0, 5.0])],
                    [np.pi*(148.41/180.), np.array([-5.0, 5.0, 2.0])],
                    [np.pi*(150.3/180.), np.array([-8.0, 8.0, 3.0])],
                    [np.pi*(151.65/180.), np.array([-14.0, 14.0, 5.0])],
                    [np.pi*(153.47/180.), np.array([-3.0, 3.0, 1.0])],
                    [np.pi*(155.08/180.), np.array([-16.0, 16.0, 5.0])],
                    [np.pi*(155.45/180.), np.array([-13.0, 13.0, 4.0])],
                    [np.pi*(156.05/180.), np.array([-10.0, 10.0, 3.0])],
                    [np.pi*(157.16/180.), np.array([-7.0, 7.0, 2.0])],
                    [np.pi*(158.17/180.), np.array([-11.0, 11.0, 3.0])],
                    [np.pi*(159.95/180.), np.array([-4.0, 4.0, 1.0])],
                    [np.pi*(161.46/180.), np.array([-13.0, 13.0, 3.0])],
                    [np.pi*(162.14/180.), np.array([-9.0, 9.0, 2.0])],
                    [np.pi*(162.77/180.), np.array([-14.0, 14.0, 3.0])],
                    [np.pi*(163.9/180.), np.array([-5.0, 5.0, 1.0])],
                    [np.pi*(164.9/180.), np.array([-16.0, 16.0, 3.0])],
                    [np.pi*(165.35/180.), np.array([-11.0, 11.0, 2.0])],
                    [np.pi*(166.56/180.), np.array([-6.0, 6.0, 1.0])],
                    [np.pi*(167.58/180.), np.array([-13.0, 13.0, 2.0])],
                    [np.pi*(168.46/180.), np.array([-7.0, 7.0, 1.0])],
                    [np.pi*(169.23/180.), np.array([-15.0, 15.0, 2.0])],
                    [np.pi*(169.9/180.), np.array([-8.0, 8.0, 1.0])],
                    [np.pi*(171.02/180.), np.array([-9.0, 9.0, 1.0])],
                    [np.pi*(171.91/180.), np.array([-10.0, 10.0, 1.0])],
                    [np.pi*(172.64/180.), np.array([-11.0, 11.0, 1.0])],
                    [np.pi*(173.26/180.), np.array([-12.0, 12.0, 1.0])],
                    [np.pi*(173.77/180.), np.array([-13.0, 13.0, 1.0])],
                    [np.pi*(174.22/180.), np.array([-14.0, 14.0, 1.0])],
                    [np.pi*(174.6/180.), np.array([-15.0, 15.0, 1.0])],
                    [np.pi*(174.94/180.), np.array([-16.0, 16.0, 1.0])]]


    sym_tilt_111 = [[np.pi*(0.0/180.), np.array([2.0, -1.0, -1.0])],
                    [np.pi*(2.65/180.), np.array([25.0, -13.0, -12.0])],
                    [np.pi*(2.88/180.), np.array([35.0, -1.0, -34.0])],
                    [np.pi*(3.15/180.), np.array([21.0, -11.0, -10.0])],
                    [np.pi*(3.89/180.), np.array([17.0, -9.0, -8.0])],
                    [np.pi*(5.09/180.), np.array([13.0, -7.0, -6.0])],
                    [np.pi*(6.01/180.), np.array([11.0, -6.0, -5.0])],
                    [np.pi*(7.34/180.), np.array([9.0, -5.0, -4.0])],
                    [np.pi*(9.43/180.), np.array([7.0, -4.0, -3.0])],
                    [np.pi*(10.42/180.), np.array([11.0, 8.0, -19.0])],
                    [np.pi*(10.99/180.), np.array([7.0, 5.0, -12.0])],
                    [np.pi*(11.64/180.), np.array([10.0, 7.0, -17.0])],
                    [np.pi*(12.36/180.), np.array([19.0, 13.0, -32.0])],
                    [np.pi*(13.17/180.), np.array([3.0, 2.0, -5.0])],
                    [np.pi*(14.11/180.), np.array([17.0, 11.0, -28.0])],
                    [np.pi*(15.18/180.), np.array([8.0, 5.0, -13.0])],
                    [np.pi*(16.43/180.), np.array([5.0, 3.0, -8.0])],
                    [np.pi*(17.9/180.), np.array([7.0, 4.0, -11.0])],
                    [np.pi*(19.65/180.), np.array([13.0, 7.0, -20.0])],
                    [np.pi*(20.67/180.), np.array([25.0, 13.0, -38.0])],
                    [np.pi*(21.79/180.), np.array([2.0, 1.0, -3.0])],
                    [np.pi*(23.04/180.), np.array([23.0, 11.0, -34.0])],
                    [np.pi*(24.43/180.), np.array([11.0, 5.0, -16.0])],
                    [np.pi*(26.01/180.), np.array([7.0, 3.0, -10.0])],
                    [np.pi*(27.8/180.), np.array([5.0, 2.0, -7.0])],
                    [np.pi*(29.84/180.), np.array([19.0, 7.0, -26.0])],
                    [np.pi*(30.59/180.), np.array([14.0, 5.0, -19.0])],
                    [np.pi*(32.2/180.), np.array([3.0, 1.0, -4.0])],
                    [np.pi*(33.99/180.), np.array([13.0, 4.0, -17.0])],
                    [np.pi*(34.96/180.), np.array([17.0, 5.0, -22.0])],
                    [np.pi*(35.98/180.), np.array([25.0, 7.0, -32.0])],
                    [np.pi*(38.21/180.), np.array([4.0, 1.0, -5.0])],
                    [np.pi*(40.07/180.), np.array([31.0, 7.0, -38.0])],
                    [np.pi*(40.73/180.), np.array([23.0, 5.0, -28.0])],
                    [np.pi*(42.1/180.), np.array([5.0, 1.0, -6.0])],
                    [np.pi*(43.57/180.), np.array([11.0, 2.0, -13.0])],
                    [np.pi*(44.35/180.), np.array([29.0, 5.0, -34.0])],
                    [np.pi*(46.83/180.), np.array([7.0, 1.0, -8.0])],
                    [np.pi*(49.58/180.), np.array([9.0, 1.0, -10.0])],
                    [np.pi*(50.57/180.), np.array([10.0, 1.0, -11.0])],
                    [np.pi*(51.39/180.), np.array([11.0, 1.0, -12.0])],
                    [np.pi*(52.66/180.), np.array([13.0, 1.0, -14.0])],
                    [np.pi*(53.99/180.), np.array([16.0, 1.0, -17.0])],
                    [np.pi*(54.91/180.), np.array([19.0, 1.0, -20.0])],
                    [np.pi*(56.11/180.), np.array([25.0, 1.0, -26.0])],
                    [np.pi*(56.85/180.), np.array([31.0, 1.0, -32.0])],
                    [np.pi*(57.35/180.), np.array([37.0, 1.0, -38.0])],
                    [np.pi*(60.0/180.), np.array([1.0, 0.0, -1.0])]]

    if args.or_axis == '001':
        orientation_axis = np.array([0, 0, 1])
        boundaries       = sym_tilt_100[:]
    elif args.or_axis == '110':
        orientation_axis = np.array([1, 1, 0])
        boundaries       = sym_tilt_110[:]
    elif args.or_axis == '111':
        orientation_axis = np.array([1, 1, 1])
        boundaries       = sym_tilt_111[:]
    else:
        sys.exit('Invalid orientation axis please generate the boundary planes and misorientation angles')

    for gb in boundaries:
        angle_str      = str(round((gb[0]*180./np.pi),2)).replace('.', '')
        if len(angle_str) > 4:
            angle_str = angle_str[:-1]
        elif len(angle_str) < 4:
            angle_str = angle_str + '0'

        gbid = '{0}{1}{2}'.format(orientation_axis[0], orientation_axis[1], orientation_axis[2]) \
                                  + angle_str  \
                                  + '{0}{1}{2}'.format(int(abs(gb[1][0])), int(abs(gb[1][1])), int(abs(gb[1][2])))

        print '\t Grain Boundary ID',  gbid

        gb_dir     = os.path.join('./ada','111')
        target_dir = os.path.join(gb_dir, gbid)
        print '\t Grain Boundary Dir', gb_dir
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)
        else:
            'directory already exists'
# Drop the coincident site lattice stuff in the directory
        gen_csl(angle, orientation_axis, gb, target_dir=target_dir, gbid=gbid)
# Drop the grain.xyz file in the target directory
        zplanes, dups, nunitcell, grain_c = build_tilt_sym_gb(gbid, bp=gb[1], v = orientation_axis,
                                                     c_space=None,
                                                     target_dir=target_dir,
                                                     rbt=[0.0, 0.0])
# json id file contains, gbid, boundaryplane, zplanes(coordinates of center of
# grain boundary
        cell = grain_c.get_cell()
        A    = cell[0][0]*cell[1][1]
        H    = cell[2][2]
        gb_dict = {"gbid"  : gbid, "boundary_plane" : list(gb[1]),
                   "orientation_axis" : list(orientation_axis),
                   "type": "symmetric tilt boundary",
                   "angle": gb[0], "zplanes" : zplanes, "coincident_sites": dups,
                   "n_at" : nunitcell, 'A':A, 'area':A, 'H':H} #area should become dominant keyword

        with open(os.path.join(target_dir, 'gb.json'), 'w') as outfile:
            json.dump(gb_dict, outfile, indent=2)
