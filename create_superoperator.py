'''Written by Hampus Karlsson (hamka@chalmers.se/hkarlsson914@gmail.com) October 2022.
   Python script for creating superoperator versions of NMR spin Hamiltonians. The script
   starts with with a Hilbert space version of the spin Hamiltonian for a spin system
   consisting of two coupled spin 1/2 nuclei. It then calculates and prints out the Liouville
   space superoperator version of the Hamiltonian to be used in other scripts.

   Based on:
   1. Helgstrand, M., Structure determination of ribosomal proteins and development of new
      methods in biomolecular NMR. PhD thesis, Royal Institute of Technology, Stockholm (2001).
   2. Allard, Helgstrand and Hard, Journal of magnetic resonance 129,19-29 (1997)
   3. Allard, P., Helgstrand, M. and Hard, T. (1998) Journal of Magnetic Resonance 134, pp.7-16.
   '''


import numpy as np


#------------------------------------------------------------------------------
#                         1. Spin operators
#------------------------------------------------------------------------------

# Spin matrices for one spin 1/2
# see eq.25 ref.1 and many others

uni =  np.array([[1., 0.],
                 [0., 1.]])

Ix = 0.5 * np.array([[0., 1.],
                     [1., 0.]])

Iy = 0.5j * np.array([[ 0., -1.],
                      [1.,   0.]])

Iz = 0.5 * np.array([[1., 0.],
                     [0.,-1.]])


# Create product operators for
# two coupled spin 1/2 nuclei

E = np.kron(uni,uni)
pIz = np.kron(Iz,uni)
pSz = np.kron(uni,Iz)
pIzSz = 2 * np.kron(Iz,Iz)

pIx = np.kron(Ix,uni)
pIy = np.kron(Iy,uni)
pIxSz = 2 * np.kron(Ix,Iz)
pIySz = 2 * np.kron(Iy,Iz)

pSx = np.kron(uni,Ix)
pSy = np.kron(uni,Iy)
pIzSx = 2 * np.kron(Iz,Ix)
pIzSy = 2 * np.kron(Iz,Iy)

pIxSx = 2 * np.kron(Ix,Ix)
pIySy = 2 * np.kron(Iy,Iy)
pIxSy = 2 * np.kron(Ix,Iy)
pIySx = 2 * np.kron(Iy,Ix)


# Flatten the product operator matrices
# to row vectors see eq.25/26 ref.1

vE = np.copy(E.flatten())
vIz = np.copy(pIz.flatten())
vSz = np.copy(pSz.flatten())
vIzSz = np.copy(pIzSz.flatten())

vIx = np.copy(pIx.flatten())
vIy = np.copy(pIy.flatten())
vIxSz = np.copy(pIxSz.flatten())
vIySz =np.copy(pIySz.flatten())

vSx = np.copy(pSx.flatten())
vSy = np.copy(pSy.flatten())
vIzSx = np.copy(pIzSx.flatten())
vIzSy = np.copy(pIzSy.flatten())

vIxSx = np.copy(pIxSx.flatten())
vIySy = np.copy(pIySy.flatten())
vIxSy = np.copy(pIxSy.flatten())
vIySx = np.copy(pIySx.flatten())


#------------------------------------------------------------------------------
#                     2. Hilbert space Hamiltonian
#------------------------------------------------------------------------------

# Write a Hilbert space Hamiltonian using the matrix operators above
H = 1*pIz + 2*pSz + 3*(pIxSx+pIySy+pIzSz) + 4*(pIx+pSx) + 5*(pIy+pSy)


#------------------------------------------------------------------------------
#                    3. Liouville space transformation
#------------------------------------------------------------------------------

# Use eq.28 Ref.1 to create super operator
Hs = np.kron(H, E.T)-np.kron(E, H.T)

# put vector versions of product operators in a list
vector_operators = [0.5*vE, vIx, vIy, vIz, vSx, vSy, vSz, vIxSz, vIySz,
                    vIzSx, vIzSy, vIxSx, vIxSy, vIySx, vIySy, vIzSz]

# Create empty array to hold the final super operator
Hsp = np.zeros((16, 16), dtype=np.complex128)

for r in range(16):

    # Levitt, section 6.2.2 also elaborates on this, <r|Hs|c>
    for c in range(16):

        rop = vector_operators[r]
        cop = vector_operators[c]

        # eq.34/35 in Ref.1 elaborates on this
        Hsp[r,c] = (np.dot(rop, np.dot(Hs, cop.T))/np.dot(rop,rop.T))


#------------------------------------------------------------------------------
#                    4. Print out the Liouvillian
#------------------------------------------------------------------------------

# Print out the superoperator matrix that can
# then be modified and used in other programs.

m, n = 0, 0

while m < 16:

    outstring = '{0:7.0f}'.format(m + 1)
    n = 0

    while n < 16:
        outstring += '{0:7.1f}'.format(Hsp.imag[m, n])

        n += 1

    print(outstring + "\n")

    m += 1

print(Hsp.imag)
