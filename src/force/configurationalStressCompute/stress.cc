// ---------------------------------------------------------------------
//
// Copyright (c) 2017 The Regents of the University of Michigan and DFT-FE authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------
//
// @author Sambit Das(2018)
//

#ifdef ENABLE_PERIODIC_BC 
template<unsigned int FEOrder>
void forceClass<FEOrder>::computeStress()
{
  //reset to zero
  for (unsigned int idim=0; idim<C_DIM; idim++)
  {
    for (unsigned int jdim=0; jdim<C_DIM; jdim++)
    {
	d_stress[idim][jdim]=0.0;
	d_stressKPoints[idim][jdim]=0.0;
    }
  }

  //configurational stress contribution from all terms except those from nuclear self energy
  if (dftParameters::spinPolarized)
     computeStressSpinPolarizedEEshelbyEPSPEnlEk();  
  else
     computeStressEEshelbyEPSPEnlEk();

  //configurational stress contribution from nuclear self energy. This is handled separately as it involves
  // a surface integral over the vself ball surface
  computeStressEself();

  //Sum all processor contributions and distribute to all processors
  d_stress=Utilities::MPI::sum(d_stress,mpi_communicator);

  //Sum k point stress contribution over all processors
  //and k point pools and add to total stress
  d_stressKPoints=Utilities::MPI::sum(d_stressKPoints,mpi_communicator); 
  d_stressKPoints=Utilities::MPI::sum(d_stressKPoints,dftPtr->interpoolcomm);  
  d_stress+=d_stressKPoints;

  //Scale by inverse of domain volume
  d_stress=d_stress*(1.0/dftPtr->d_domainVolume);
}


template<unsigned int FEOrder>
void forceClass<FEOrder>::printStress()
{
    pcout<< "------------Configurational stress (Hartree/Bohr^3)--------------------- "<< std::endl;
    for (unsigned int idim=0; idim< 3; idim++)
        pcout<< d_stress[idim][0]<<"  "<<d_stress[idim][1]<<"  "<<d_stress[idim][2]<< std::endl;    
    pcout<< "------------------------------------------------------------------------"<<std::endl;    
}

#endif