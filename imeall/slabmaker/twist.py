import os
import sys
import json
import time
import operator
import numpy as np
import transformations as quat

from ase.io import read
from ase.io import write
from ase.lattice.spacegroup import crystal
from ase.lattice.surface import surface, bcc111,bcc110
from ase.lattice.cubic   import BodyCenteredCubic
from ase.geometry  import get_duplicate_atoms

from slabmaker import gen_csl
from fractions import gcd


def integralize_quaternion(n1, n2, angle):
    """Integralize quaternion.

    Returns:
      Formatted string of angle and appropriate x vector.
    """
    comm_denom = []
    for a in n1:
        if (a%1).round(1) != 0:
            comm_denom.append(1./(np.abs(a)%1))
    comm_denom = np.array(comm_denom)
    if len(comm_denom)!=0:
        bp1 = (comm_denom.max()*n1).round(2)
        bp2 = (comm_denom.max()*n2).round(2)
    else:
        bp1 = n1
        bp2 = n2
    abs_bp1 = map(np.abs, bp1)
    grcd  = reduce(gcd, abs_bp1)
    bp1     = bp1/grcd
    rad  = '\t [{0}, np.array([{1} {2} {3}])],'.format(round(angle, 2), bp1[1].round(0), bp1[2].round(0), bp1[3].round(0))
    deg  = '\t [np.pi*({0}/180.), np.array([{1}, {2}, {3}])],'.format(round(180.*angle/np.pi,2), bp1[1].round(2), bp1[2].round(2), bp1[3].round(2))
    return [round(180.*angle/np.pi, 2),  [deg]]

if __name__=='__main__':
    sym_twist_111 = [
                   [np.pi*(0.0/180.), np.array([2.0, -1.0, -1.0])],
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

    sym_twist_001 = [
                   [np.pi*(7.15/180.), np.array([1.0, 16.0, 0.0])],
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

    sym_twist_110 = [[np.pi*(10.1/180.), np.array([-1.0, 1.0, 16.0])],
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

    #gen_sym_twist_gb()
    #build_twist_sym_gb(target_dir='./')
    #orientation_axis  = np.array([0,0,1])
    orientation_axis  = np.array([1,1,0])
    #orientation_axis  = np.array([1,1,1])
    #surfaces = [[np.pi*(0.0), np.array([0,0,1])]]
    #surfaces = [[np.pi*(61.93/180.), np.array([3.0, 5.0, 0.0])]]
    #surfaces = sym_twist_001
    #surfaces = [[np.pi*(174.6/180.), np.array([-15.0, 15.0, 1.0])]]
    surfaces = sym_twist_110
    #surfaces = sym_twist_111
    for gb in surfaces:
        angle_str      = str(round((gb[0]*180./np.pi),2)).replace('.', '')
        if len(angle_str) > 4:
            angle_str = angle_str[:-1]
        elif len(angle_str) < 4:
            angle_str = angle_str + '0'
        gbid = '{0}{1}{2}'.format(orientation_axis[0], orientation_axis[1],orientation_axis[2]) \
               + angle_str  \
               + '{0}{1}{2}'.format(orientation_axis[0],orientation_axis[1], orientation_axis[2])
        print '\t Grain Boundary ID',  gbid
        #gb_dir     = os.path.join('./boundaries/twist', '111')
        #gb_dir     = os.path.join('./boundaries/twist', '001')
        gb_dir     = os.path.join('./boundaries/twist', '110')
        target_dir = os.path.join(gb_dir, gbid)
        print '\t Grain Boundary Dir', gb_dir
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)
        else:
            print 'directory already exists'
        gen_csl(orientation_axis, gb, target_dir = target_dir, gbid=gbid, gb_type="twist")
        zplanes, sigma_csl, nunitcell, grain_c = build_twist_sym_gb(gbid, bp=orientation_axis, v=gb[1],
                                                               c_space=None, target_dir=target_dir,
                                                               rbt=[0.0, 0.0])
        cell    = grain_c.get_cell()
        A       = cell[0][0]*cell[1][1]
        H       = cell[2][2]
        gb_dict = {"gbid": gbid, "boundary_plane": list(orientation_axis),
                   "orientation_axis": list(orientation_axis),
                   "type": "symmetric twist boundary",
                   "angle": gb[0], "zplanes": zplanes, "sigma_csl": sigma_csl,
                   "n_at": nunitcell, 'A': A, 'area':A, 'H': H}

        with open(os.path.join(target_dir, 'gb.json'), 'w') as outfile:
            json.dump(gb_dict, outfile, indent=2)

        grain_c.write(os.path.join(target_dir, '{}.xyz'.format(gbid)))
