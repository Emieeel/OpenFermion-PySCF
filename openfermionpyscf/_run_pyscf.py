#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Driver to initialize molecular object from pyscf program."""

from __future__ import absolute_import

from functools import reduce

import os

import numpy as np 

from pyscf import gto, scf, ao2mo, ci, cc, fci, mp, lo, tools, mcscf

from openfermion import MolecularData
from openfermionpyscf import PyscfMolecularData

import time


def prepare_pyscf_molecule(molecule, verbose_lvl=None, output_file=None):
    """
    This function creates and saves a pyscf input file.

    Args:
        molecule: An instance of the MolecularData class.

    Returns:
        pyscf_molecule: A pyscf molecule instance.
    """
    pyscf_molecule = gto.Mole()
    pyscf_molecule.atom = molecule.geometry
    pyscf_molecule.basis = molecule.basis
    pyscf_molecule.spin = molecule.multiplicity - 1
    pyscf_molecule.charge = molecule.charge
    pyscf_molecule.symmetry = False
    if verbose_lvl is not None:
        pyscf_molecule.verbose = verbose_lvl
        if output_file is None:
            raise ValueError("Output file must be specified if setting pyscf verbose level")
        pyscf_molecule.output = os.getcwd() + '/output_files/' + output_file + 'verb' + verbose_lvl
    pyscf_molecule.build()

    return pyscf_molecule


def compute_scf(pyscf_molecule):
    """
    Perform a Hartree-Fock calculation.

    Args:
        pyscf_molecule: A pyscf molecule instance.

    Returns:
        pyscf_scf: A PySCF "SCF" calculation object.
    """
    if pyscf_molecule.spin:
        pyscf_scf = scf.ROHF(pyscf_molecule)
    else:
        pyscf_scf = scf.RHF(pyscf_molecule)
    return pyscf_scf


def compute_integrals(pyscf_molecule, pyscf_scf, C, threshold):
    """
    Compute the 1-electron and 2-electron integrals.

    Args:
        pyscf_molecule: A pyscf molecule instance.
        pyscf_scf: A PySCF "SCF" calculation object.

    Returns:
        one_electron_integrals: An N by N array storing h_{pq}
        two_electron_integrals: An N by N by N by N array storing h_{pqrs}.
    """
    # Get one electrons integrals.
    n_orbitals = C.shape[1]
    one_electron_compressed = reduce(np.dot, (C.T,
                                              pyscf_scf.get_hcore(),
                                              C))
    # print(one_electron_compressed.shape, 'norb', n_orbitals)
    one_electron_integrals = one_electron_compressed.reshape(
        n_orbitals, n_orbitals).astype(float)

    # Get two electron integrals in compressed format.
    two_electron_compressed = ao2mo.kernel(pyscf_molecule,
                                           C)

    two_electron_integrals = ao2mo.restore(
        1, # no permutation symmetry
        two_electron_compressed, n_orbitals)
    low_indices = abs(two_electron_integrals) < threshold
    two_electron_integrals[low_indices] = 0.
    # print('2ec shape', two_electron_compressed.shape, '2ei shape', two_electron_integrals.shape)
    # See PQRS convention in OpenFermion.hamiltonians._molecular_data
    # h[p,q,r,s] = (ps|qr)
    two_electron_integrals = np.asarray(
        two_electron_integrals.transpose(0, 2, 3, 1), order='C')

    # Return.
    return one_electron_integrals, two_electron_integrals

def transpose_integrals(h1e, mo_ints, nmo):
    """
    Compute the 1-electron and 2-electron integrals.

    Args:
        pyscf_molecule: A pyscf molecule instance.
        pyscf_scf: A PySCF "SCF" calculation object.

    Returns:
        one_electron_integrals: An N by N array storing h_{pq}
        two_electron_integrals: An N by N by N by N array storing h_{pqrs}.
    """
    # Get one electrons integrals.
    one_electron_compressed = h1e
    one_electron_integrals = one_electron_compressed.reshape(
        nmo, nmo).astype(float)

    # Get two electron integrals in compressed format.
    two_electron_integrals = mo_ints.reshape(nmo,nmo,nmo,nmo)
    # See PQRS convention in OpenFermion.hamiltonians._molecular_data
    # h[p,q,r,s] = (ps|qr)
    two_electron_integrals = np.asarray(
        two_electron_integrals.transpose(0, 2, 3, 1), order='C')

    # Return.
    return one_electron_integrals, two_electron_integrals

def run_pyscf(molecule,
              threshold=0.,
              also_h1e=False,
              run_scf=True,
              run_mp2=False,
              run_cisd=False,
              run_ccsd=False,
              run_fci=False,
              run_casci=False,
              n_electrons=0,
              n_orbitals=0,
              verbose=False,
              localize=False,
              visualize=False,
              localizemethod='Pipek-Mezey',
              localize_virt=False,
              localize_cas=False,
              only_cas=False,
              also_occ=True,
              localize_sep=True,
              do_cholesky=False,
              do_svd=False,
              Orth_AO=False,
              save_tohdf5=True,
              max_memory=8000,
              verbose_lvl=None):
    """
    This function runs a pyscf calculation.

    Args:
        molecule: An instance of the MolecularData or PyscfMolecularData class.
        threshold: Threshold on the two_body_integrals
        also_h1e: Also oppose threshold on the one_body_integrals
        run_scf: Optional boolean to run SCF calculation.
        run_mp2: Optional boolean to run MP2 calculation.
        run_cisd: Optional boolean to run CISD calculation.
        run_ccsd: Optional boolean to run CCSD calculation.
        run_fci: Optional boolean to FCI calculation.
        verbose: Boolean whether to print calculation results to screen.
        localize: Localize the MOs after Hartree-Fock
        visualize: Print the MOs to cube files
        localizemethod: Localization method to use
        localize_virt: Localize the virtual MOs as well
        do_svd: Impose the threshold on the singular values of the MO_ints
        max_memory: Maximal memory the calculation should take

    Returns:
        molecule: The updated PyscfMolecularData object. Note the attributes
        of the input molecule are also updated in this function.
    """
    #print('Test!!!!!!!!!')
    # Prepare pyscf molecule.
    pyscf_molecule = prepare_pyscf_molecule(molecule, verbose_lvl, molecule.description)
    pyscf_molecule.max_memory = max_memory
    molecule.n_orbitals = int(pyscf_molecule.nao_nr())
    molecule.n_qubits = 2 * molecule.n_orbitals
    molecule.nuclear_repulsion = float(pyscf_molecule.energy_nuc())

    t15 = time.time()
    # Run SCF.
    pyscf_scf = compute_scf(pyscf_molecule)
    pyscf_scf.verbose = 0
    pyscf_scf.run()
    if verbose: print('Doing HF took', time.time()-t15)
    
    # Initialize MOs
    C_nonloc = np.copy(pyscf_scf.mo_coeff)
    nmo = C_nonloc.shape[1] # Number of MOs
    
    t0 = time.time()
    # Localize orbitals
    if localize:
        ndocc = np.count_nonzero(pyscf_scf.mo_occ) # Number of (doubly) occupied MOs
        if localize_cas:
            # Localize MOs in active space
            n_core_orbitals = (pyscf_molecule.nelectron - n_electrons) // 2
            if localize_virt:
                ntot = n_core_orbitals + n_orbitals
            else:
                ntot = ndocc
                localize_sep = 1
        else:
            # Localize every MO
            n_core_orbitals = 0
            if localize_virt:
                ntot = len(pyscf_scf.mo_occ)
            else:
                localize_sep = 1
                ntot = ndocc
                
        C = C_nonloc
        
        if localize_sep:
            print ('LOCALIZING SEPERATELY')
            if localizemethod == 'Pipek-Mezey':
                orb = lo.PipekMezey(pyscf_molecule).kernel(C[:,n_core_orbitals:ndocc])
                if localize_virt:
                    orbvirt = lo.PipekMezey(pyscf_molecule).kernel(C[:,ndocc:ntot])
            elif localizemethod == 'Boys':
                orb = lo.Boys(pyscf_molecule).kernel(C[:,n_core_orbitals:ndocc])
                if localize_virt:
                    orbvirt = lo.Boys(pyscf_molecule).kernel(C[:,ndocc:ntot])
            elif localizemethod == 'ibo':
                orb = lo.ibo.ibo(pyscf_molecule, C[:,n_core_orbitals:ndocc])
                if localize_virt:
                    orbvirt = lo.ibo.ibo(pyscf_molecule, C[:,ndocc:ntot])
            elif localizemethod == 'ER':
                orb = lo.EdmistonRuedenberg(pyscf_molecule).kernel(C[:,n_core_orbitals:ndocc])
                if localize_virt:
                    orbvirt = lo.EdmistonRuedenberg(pyscf_molecule).kernel(C[:,ndocc:ntot])
            else:
                raise ValueError('Localization method not recognized')
        else:
            print('LOCALIZING TOGETHER','n_core',n_core_orbitals,'n_active',n_orbitals, \
                  'n_virtual', (nmo-(n_core_orbitals + n_orbitals)))
            if localizemethod == 'Pipek-Mezey':
                orb = lo.PipekMezey(pyscf_molecule).kernel(C[:,n_core_orbitals:ntot])
            elif localizemethod == 'Boys':
                orb = lo.Boys(pyscf_molecule).kernel(C[:,n_core_orbitals:ntot])
            elif localizemethod == 'ibo':
                orb = lo.ibo.ibo(pyscf_molecule, C[:,n_core_orbitals:ntot])
            elif localizemethod == 'ER':
                orb = lo.EdmistonRuedenberg(pyscf_molecule).kernel(C[:,n_core_orbitals:ntot])
            else:
                raise ValueError('Localization method not recognized')
        
        
        if localize_virt:
            if localize_sep:
                if localize_cas:
                    C = np.hstack((C_nonloc[:,:n_core_orbitals],orb,orbvirt,C_nonloc[:,ntot:nmo]))
                else:
                    C = np.hstack((orb,orbvirt))
            else:
                if localize_cas:
                    C = np.hstack((C_nonloc[:,:n_core_orbitals], orb, C_nonloc[:,ntot:nmo]))
                else:
                    C = orb
        else:
            if localize_cas:
                C = np.hstack((C_nonloc[:,:n_core_orbitals], orb, C_nonloc[:,ndocc:]))
            else:                
                C = np.hstack((orb,C_nonloc[:,ndocc:]))
        pyscf_scf.mo_coeff = C 
        if verbose: print('core',n_core_orbitals,'ndocc',ndocc,'n_orb',n_orbitals,'nmo',nmo)
    
    # If you want to return orthogonal AOs as the MOS
    elif Orth_AO:
        t5 = time.time()
        S = pyscf_scf.get_ovlp()
        S_eigval, S_eigvec = np.linalg.eigh(S)
        S_sqrt_inv = S_eigvec @ np.diag((S_eigval)**(-1./2.)) @ np.linalg.inv(S_eigvec)
        C = S_sqrt_inv
        
        pyscf_scf.mo_coeff = C
        if verbose: print("Computing inverse overlap took", time.time()-t5)
    else: 
        C = C_nonloc
    
    
    if verbose: print('localizing took', time.time()-t0)


    # Visualize the orbitals by preparing cube files    
    if visualize:
        t13 = time.time()
        ndocc = np.count_nonzero(pyscf_scf.mo_occ)
        nmo = len(pyscf_scf.mo_occ)
        print('ndocc', ndocc, 'nmo', nmo)
        for i in range(nmo):    
            tools.cubegen.orbital(pyscf_molecule, os.getcwd() + '/CUBE_FILES/pyscfcube'\
                                  + molecule.description + localizemethod + str(localize)\
                                  + str(i) , pyscf_scf.mo_coeff[:,i])
        print('Cube files of molecule', molecule.name,'created in', os.getcwd() + '/CUBE_FILES/')
        if verbose: print('extracting cube files took', time.time()-t13)
  
    # Now set a threshold.
    # Set it also on the one electron integrals?
    if also_h1e:
        h1e = C.T.dot(pyscf_scf.get_hcore()).dot(C)
        low_indices_h1e = abs(h1e) < threshold
        h1e[low_indices_h1e] = .0

    # Or we could use the cholesky decomposition with convergence threshold
    # If we run an FCI, we do the cholesky decomposition here. If we run CASCI,
    # we only do cholesky decomposition inside the CAS.
    elif do_cholesky and run_fci:
        t11 = time.time()
        h1e = C.T.dot(pyscf_scf.get_hcore()).dot(C)
        mo_ints = ao2mo.full(pyscf_molecule, C, compact=False)
        mo_ints[ abs(mo_ints) < 1e-16] = 0.
        if verbose: print("extracting mo_ints took", time.time()-t11)
        t20 = time.time()
        mo_ints_ch = cholesky(mo_ints.reshape(nmo,nmo,nmo,nmo), nmo, threshold, verbose=0)
        if verbose: print("doing cholesky decomp took", time.time()-t20)
        molecule.general_calculations['n_ch'] = mo_ints_ch.shape[-1]
        t10 = time.time()
        mo_ints = np.einsum('pqt,rst->pqrs',mo_ints_ch,mo_ints_ch).reshape(nmo**2,nmo**2)
        if verbose: print("recalculating moints took", time.time()-t10)
    
    # Otherwise we just set the threshold directly
    elif run_fci:
        t1 = time.time()
        # Extract integrals and set a threshold 
        h1e = C.T.dot(pyscf_scf.get_hcore()).dot(C) # One electron integrals
        mo_ints = ao2mo.full(pyscf_molecule, C, aosym=1) # Two electron integrals
        if verbose: print('extracting integrals took', time.time() - t1)
        t2 = time.time()
        low_indices = abs(mo_ints) < threshold
        mo_ints[low_indices] = .0       
        if verbose: print('setting threshold took', time.time()-t2)
    
    # Pyscf module
    molecule.hf_energy = float(pyscf_scf.e_tot)
    if verbose:
        print('Hartree-Fock energy for {} ({} electrons) is {}.'.format(
            molecule.name, molecule.n_electrons, molecule.hf_energy))
    
    t7 = time.time()
    # Hold pyscf data in molecule. They are required to compute density
    # matrices and other quantities.
    molecule._pyscf_data = pyscf_data = {}
    pyscf_data['mol'] = pyscf_molecule
    pyscf_data['scf'] = pyscf_scf

    # Populate fields.
    molecule.canonical_orbitals = pyscf_scf.mo_coeff.astype(float)
    molecule.orbital_energies = pyscf_scf.mo_energy.astype(float)
    if verbose: print('populating data fields in molecule class took', time.time()-t7)
    
    
    # Now we compute the integrals. We can compute them only inside the active
    # space (used for saving memory), or we can compute them for every MO. 
    if only_cas:
        t3 = time.time()
        ncore = (pyscf_molecule.nelectron - n_electrons) // 2
        nocc = ncore + n_orbitals
        if also_occ:
            C_copy = C[:,:nocc]
        else:
            C_copy = C[:,ncore:nocc]
        one_body_integrals, two_body_integrals = compute_integrals(
                pyscf_molecule, pyscf_scf, C_copy, threshold)
        molecule.one_body_integrals = one_body_integrals
        molecule.two_body_integrals = two_body_integrals
        molecule.overlap_integrals = pyscf_scf.get_ovlp()
        if verbose:
            #print('shape 2bodyints is', two_body_integrals.shape)
            print('computing and storing ONLY cas integrals (with ncore',ncore,'ncas',\
                  n_orbitals, 'and nocc',nocc,') took', time.time()-t3)
    else:
        t3 = time.time()
        # If we do run_fci, we already computed the integrals with setting
        # the treshold, in which case we only need to transpose them.
        if run_fci:
            one_body_integrals, two_body_integrals = transpose_integrals(
                h1e, mo_ints, nmo)
        else:
            one_body_integrals, two_body_integrals = compute_integrals(
                pyscf_molecule, pyscf_scf, C, threshold)
        molecule.one_body_integrals = one_body_integrals
        molecule.two_body_integrals = two_body_integrals
        molecule.overlap_integrals = pyscf_scf.get_ovlp()
        if verbose: print(f'{"transposing" if run_fci else "computing"} integrals and storing them took', time.time() - t3)
    
    # Run MP2.
    if run_mp2:
        if molecule.multiplicity != 1:
            print("WARNING: RO-MP2 is not available in PySCF.")
        else:
            pyscf_mp2 = mp.MP2(pyscf_scf)
            pyscf_mp2.verbose = 0
            pyscf_mp2.run()
            molecule.mp2_energy = pyscf_mp2.e_tot  # pyscf-1.4.4 or higher
            #molecule.mp2_energy = pyscf_scf.e_tot + pyscf_mp2.e_corr
            pyscf_data['mp2'] = pyscf_mp2
            if verbose:
                print('MP2 energy for {} ({} electrons) is {}.'.format(
                    molecule.name, molecule.n_electrons, molecule.mp2_energy))

    # Run CISD.
    if run_cisd:
        pyscf_cisd = ci.CISD(pyscf_scf)
        pyscf_cisd.verbose = 0
        pyscf_cisd.run()
        molecule.cisd_energy = pyscf_cisd.e_tot
        pyscf_data['cisd'] = pyscf_cisd
        if verbose:
            print('CISD energy for {} ({} electrons) is {}.'.format(
                molecule.name, molecule.n_electrons, molecule.cisd_energy))

    # Run CCSD.
    if run_ccsd:
        pyscf_ccsd = cc.CCSD(pyscf_scf)
        pyscf_ccsd.verbose = 0
        t8 = time.time()
        pyscf_ccsd.run()
        if verbose: print('ccsd calculation took', time.time() - t8)
        molecule.ccsd_energy = pyscf_ccsd.e_tot
        pyscf_data['ccsd'] = pyscf_ccsd
        if verbose:
            print('CCSD energy for {} ({} electrons) is {}.'.format(
                molecule.name, molecule.n_electrons, molecule.ccsd_energy))

    # Run FCI.
    if run_fci:
        pyscf_fci = fci.FCI(pyscf_molecule, C)
        pyscf_fci.verbose = 0
        t9 = time.time()
        molecule.fci_energy = pyscf_fci.kernel(h1e = h1e, 
                                               eri = mo_ints, 
                                               norb = h1e.shape[1], 
                                               nelec = pyscf_molecule.nelec, 
                                               ecore=pyscf_molecule.energy_nuc())[0]
        if verbose: print('fci calculation took', time.time() - t9)
        pyscf_data['fci'] = pyscf_fci
        if verbose:
            print('FCI energy for {} ({} electrons) is {}.'.format(
                molecule.name, molecule.n_electrons, molecule.fci_energy))
    
    # Run CASCI
    if run_casci:
        t4 = time.time()
        print('Computing CASCI(' + str(n_orbitals) + ',' + str(n_electrons) + ')')
        pyscf_cas = pyscf_scf.CASCI(n_orbitals, n_electrons)
        pyscf_cas.verbose = 0
        pyscf_cas.canonicalization = 0
        ncore = pyscf_cas.ncore
        ncas = pyscf_cas.ncas
        nocc = ncore + ncas
        print('ncore',ncore,'ncas',ncas,'nocc',nocc)
        C = C[:,ncore:nocc]
        mo_intscas = ao2mo.full(pyscf_molecule, C, compact=False)
        if verbose: print('extracting cas integrals took', time.time()-t4)
        
        if do_cholesky:
            # if verbose: print('ncas=',ncas,'mo_intscas.shape=',mo_intscas.shape)
            mo_intscas[ abs(mo_intscas) < 1e-16] = 0.
            t2 = time.time()
            mo_ints_ch = cholesky(mo_intscas.reshape(ncas,ncas,ncas,ncas), ncas, threshold, verbose=0)
            if verbose: print("doing cholesky decomp took", time.time()-t2)
            molecule.general_calculations['n_ch'] = mo_ints_ch.shape[-1]
            t10 = time.time()
            # if verbose: print('SHAPE mo_ints_ch=',mo_ints_ch.shape)
            if len(mo_ints_ch.shape) > 2:
                mo_intscas = np.einsum('pqt,rst->pqrs',mo_ints_ch,mo_ints_ch).reshape(ncas**2,ncas**2)
            else:
                mo_intscas = np.einsum('pq,rs->pqrs',mo_ints_ch,mo_ints_ch).reshape(ncas**2,ncas**2)
            if verbose: print("recalculating moints took", time.time()-t10)
        else:
            t12 = time.time()
            low_indices = abs(mo_intscas) < threshold
            mo_intscas[low_indices] = .0
            if verbose: print("setting threshold on mo_ints took", time.time()-t12)
        
        pyscf_cas.eri = mo_intscas
        t5 = time.time()
        pyscf_cas.kernel()
        molecule.general_calculations['casci_energy'] = pyscf_cas.e_tot
        molecule.general_calculations['nterms'] = np.count_nonzero(mo_intscas)
        if verbose: print('doing casci took', time.time()-t5)
        
        
        if verbose:
            print('CASCI energy for {} ({} electrons in {} orbitals) is {}.'.format(molecule.name,
                n_electrons, n_orbitals, molecule.general_calculations['casci_energy']))
       
    t6 = time.time()
    # Return updated molecule instance.
    pyscf_molecular_data = PyscfMolecularData.__new__(PyscfMolecularData)
    pyscf_molecular_data.__dict__.update(molecule.__dict__)
    if verbose: print('saving data to molecular data structure took', time.time()-t6)
    
    if save_tohdf5:
        t7 = time.time()
        pyscf_molecular_data.save()
        if verbose: print('saving data in moldata to hdf5 took', time.time()-t7)
    return pyscf_molecular_data


def generate_molecular_hamiltonian( geometry,
                                    basis,
                                    multiplicity,
                                    threshold=0.,
                                    charge=0,
                                    n_active_electrons=None,
                                    n_active_orbitals=None,
                                    also_h1e=False,
                                    run_scf=True,
                                    run_mp2=False,
                                    run_cisd=False,
                                    run_ccsd=False,
                                    run_fci=False,
                                    run_casci=False,
                                    verbose=False,
                                    localize=False,
                                    visualize=False,
                                    localizemethod='Pipek-Mezey',
                                    localize_virt=False,
                                    do_svd=False,
                                    max_memory=4000):
    """Generate a molecular Hamiltonian with the given properties.

    Args:
        geometry: A list of tuples giving the coordinates of each atom.
            An example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
            Distances in angstrom. Use atomic symbols to
            specify atoms.
        basis: A string giving the basis set. An example is 'cc-pvtz'.
            Only optional if loading from file.
        multiplicity: An integer giving the spin multiplicity.
        charge: An integer giving the charge.
        n_active_electrons: An optional integer specifying the number of
            electrons desired in the active space.
        n_active_orbitals: An optional integer specifying the number of
            spatial orbitals desired in the active space.

    Returns:
        The Hamiltonian as an InteractionOperator.
    """
    # Initialize molecule
    molecule = MolecularData(geometry, basis, multiplicity, charge)
    
    # Run electronic structure calculations
    molecule = run_pyscf(molecule,
                         threshold=threshold,
                         also_h1e=also_h1e,
                         run_scf=run_scf,
                         run_mp2=run_mp2,
                         run_cisd=run_cisd,
                         run_ccsd=run_ccsd,
                         run_fci=run_fci,
                         run_casci=run_casci,
                         verbose=verbose,
                         localize=localize,
                         visualize=visualize,
                         localizemethod=localizemethod,
                         localize_virt=localize_virt,
                         do_svd=do_svd,
                         max_memory=max_memory)

    # Freeze core orbitals and truncate to active space
    if n_active_electrons is None:
        n_core_orbitals = 0
        occupied_indices = None
    else:
        n_core_orbitals = (molecule.n_electrons - n_active_electrons) // 2
        occupied_indices = list(range(n_core_orbitals))

    if n_active_orbitals is None:
        active_indices = None
    else:
        active_indices = list(range(n_core_orbitals,
                                    n_core_orbitals + n_active_orbitals))

    return molecule.get_molecular_hamiltonian(
            occupied_indices=occupied_indices,
            active_indices=active_indices)

def cholesky(G, nao, threshold, realMO=True, verbose=1):
    """
    
    Parameters
    ----------
    G : numpy.array
        two-electron integrals.
    nao : numpy.array
        number of AOs/MOs.
    threshold : int
        convergence threshold.
    realMO : bool, optional
        Real (True) or complex (False) MOs. The default is True.
    verbose : int, optional
        verbosity (0, 1 or 2). The default is 1.

    Returns
    -------
    Cholesky matrix L

    """
    #starting with cholesky decomposition
    D=np.zeros((nao,nao))
    for i in range(nao):
        for j in range(nao):
            D[i,j]=G[i,j,i,j]
            
    i_max=np.zeros((2),dtype=np.int)
    i_help=np.where(np.max(abs(D))==abs(D))
    try:
        i_max[0]=i_help[0][0][0]
        i_max[1]=i_help[0][0][1]
    except:
        i_max[0]=i_help[0][0]
        i_max[1]=i_help[1][0]
    D_max=D[i_max[0],i_max[1]]
    if verbose == 1:
        print('ind = '+str(i_max[0])+' '+str(i_max[1])+'  D_max='+str(D_max))
    if verbose == 2:
        print('ind = '+str(i_max[0])+' '+str(i_max[1])+'  D_max='+str(D_max))
        print(D) 
    
    m=0
    Res=G
    while abs(D_max)>threshold:
        if verbose == 1 or verbose == 2:
            print('-------- iteration m = '+str(m)+'-------')
        
        #t1 = time.time()
        L_h=Res[:,:,i_max[0],i_max[1]]/np.sqrt(np.abs(D_max))
        L_sign=np.sign(D_max)
        #print('Computing L_h took', time.time()-t1)
        t2 = time.time()
        G_h=np.zeros((nao,nao,nao,nao)) 
        if m==0:
            L=L_h
            Ls=L_sign
            G_h=G_h+Ls*np.einsum('ij,kl->ijkl', L[ :, :], L[ :, :])
        else:
            L=np.dstack((L,L_h))  
            Ls=np.hstack((Ls,L_sign))
            for i in range(m+1):
                if realMO:
                    G_h=G_h+np.einsum('ij,kl->ijkl', L[ :, :, i], L[ :, :, i])
                    #G_h=G_h+prod_out(L[ :, :, i], L[ :, :, i])
                    # print("DIFFERENCE PROD_OUT AND EINSUM:", \
                    #       np.max(prod_out(L[ :, :, i], L[ :, :, i])-\
                    #       np.einsum('ij,kl->ijkl', L[ :, :, i], L[ :, :, i])))
                else:
                    G_h=G_h+Ls[i]*np.einsum('ij,kl->ijkl', L[ :, :, i], L[ :, :, i])
                    # G_h=G_h+Ls[i]*prod_out(L[ :, :, i], L[ :, :, i])
                    
        if verbose == 1 or verbose == 2:
            print('computing G_h took', time.time() - t2)
        #t3 = time.time()
        Res=G-G_h
        err_h=np.sum(np.diag(np.abs(Res.reshape(Res.shape[0]**2,Res.shape[1]**2))))
        # err_h = np.sum(np.abs(Res))
        #print('computing residue and error took', time.time()-t3)
        #t4 = time.time()
        for i in range(nao):
            for j in range(nao):
                D[i,j]=Res[i,j,i,j]
                
        m=m+1
        i_help=np.where(np.max(abs(D))==abs(D))
        try:
            i_max[0]=i_help[0][0][0]
            i_max[1]=i_help[0][0][1]
        except:
            i_max[0]=i_help[0][0]
            i_max[1]=i_help[1][0]
        D_max=D[i_max[0],i_max[1]]
        if verbose == 2:
            print(D)
        if verbose == 1 or verbose == 2:
            print('ind = '+str(i_max[0])+' '+str(i_max[1])+'  D_max='+str(D_max))
            print('current error = '+str(err_h))
        #print('computing new D and D_max took', time.time() - t4)
        if m>500: 
            break
    if verbose == 1 or verbose == 2:   
        print('_________________ decomposition done ___________________________')
    return(L)