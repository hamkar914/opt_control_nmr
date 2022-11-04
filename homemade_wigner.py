'''Written by Hampus Karlsson October 2022, hamka@chalmers.se/hkarlsson914@gmail.com. Python script
   that provides function for calculating 2nd rank Wigner matrix for rotating 2nd rank irreducible
   spherical tensors.

   Based on:
   1. A.J. Pell et al., Progress in Nuclear Magnetic Resonance Spectroscopy 111 (2019) 1–271.
   2. Eden, M. Concepts in Magnetic Resonance Part A, Vol. 17A(1)117-154 (2003)'''


import numpy as np
from numpy import sin,cos,sqrt

def generate_wigner(eul_ang_triplet):

    '''Returns a 2nd rank, 5x5 Wigner matrix with euler angles
       supplied in tuple "eul_ang_triplet" (angles in radians).
       Based on eq. 2.98 and table 2.4 Ref.1 and eq.49 Ref.2.
       Output confirmed with SpinDynamica WignerD function.'''

    alpha = eul_ang_triplet[0]
    beta = eul_ang_triplet[1]
    gamma = eul_ang_triplet[2]

    # Empty matrix to be filled
    Wigner = np.zeros((5,5),dtype=np.complex128)

    # the m prime and m values...
    plus_to_minus_two = np.array([2, 1, 0, -1, -2])

    # The reduced Wigner functions taken from table 2.4 in in Ref.1  and arranged in 5x5 matrix.
    red_wig = np.array([[           cos(beta/2)**4,        -sin(beta)*(1+cos(beta))/2,           sqrt(3/8)*sin(beta)**2,        -sin(beta)*(1-cos(beta))/2,               sin(beta/2)**4],

                        [sin(beta)*(1+cos(beta))/2,   (2*cos(beta)-1)*(1+cos(beta))/2,   -sqrt(3/2)*sin(beta)*cos(beta),   (2*cos(beta)+1)*(1-cos(beta))/2,   -sin(beta)*(1-cos(beta))/2],

                        [   sqrt(3/8)*sin(beta)**2,     sqrt(3/2)*sin(beta)*cos(beta),             (3*cos(beta)**2-1)/2,    -sqrt(3/2)*sin(beta)*cos(beta),       sqrt(3/8)*sin(beta)**2],

                        [sin(beta)*(1-cos(beta))/2,   (2*cos(beta)+1)*(1-cos(beta))/2,    sqrt(3/2)*sin(beta)*cos(beta),   (2*cos(beta)-1)*(1+cos(beta))/2,   -sin(beta)*(1+cos(beta))/2],

                        [           sin(beta/2)**4,         sin(beta)*(1-cos(beta))/2,           sqrt(3/8)*sin(beta)**2,         sin(beta)*(1+cos(beta))/2,               cos(beta/2)**4]])

    for row in range(5):
        mp = plus_to_minus_two[row]

        for col in range(5):

            m = plus_to_minus_two[col]
            Wigner[row,col] = np.exp(-1j*mp*alpha)*red_wig[row,col]*np.exp(-1j*m*gamma)

    return Wigner
