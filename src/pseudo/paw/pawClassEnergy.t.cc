// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
// authors.
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
// @author Kartick Ramakrishnan
//

namespace dftfe
{
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::
    evaluateNonLocalHamiltonianElectrostaticsValue(
      const distributedCPUVec<double> &phiTotNodalValues,
      const unsigned int               dofHandlerId)
  {
    double alpha = 1.0;
    double beta  = 1.0;
    d_BasisOperatorElectroHostPtr->reinit(
      0, 0, d_compensationChargeQuadratureIdElectro);
    const unsigned int numberNodesPerElement =
      d_BasisOperatorElectroHostPtr->nDofsPerCell();
    const unsigned int numberQuadraturePoints =
      d_BasisOperatorElectroHostPtr->nQuadsPerCell();
    const unsigned int numberAtomsOfInterest =
      d_atomicShapeFnsContainer->getNumAtomCentersSize();
    const std::vector<unsigned int> &atomicNumber =
      d_atomicShapeFnsContainer->getAtomicNumbers();
    const std::vector<unsigned int> atomIdsInCurrentProcess =
      d_atomicShapeFnsContainer->getAtomIdsInCurrentProcess();
    d_nonLocalHamiltonianElectrostaticValue.clear();
    for (unsigned int iAtom = 0; iAtom < atomIdsInCurrentProcess.size();
         iAtom++)
      {
        const unsigned int atomId = atomIdsInCurrentProcess[iAtom];
        const unsigned int Znum   = atomicNumber[atomId];
        const unsigned int NumTotalSphericalFunctions =
          d_atomicShapeFnsContainer->getTotalNumberOfSphericalFunctionsPerAtom(
            Znum);
        d_nonLocalHamiltonianElectrostaticValue[atomId] =
          std::vector<double>(NumTotalSphericalFunctions, 0.0);
      }
    dealii::FEEvaluation<3, -1> feEvalObj(
      d_BasisOperatorElectroHostPtr->matrixFreeData(),
      dofHandlerId,
      d_compensationChargeQuadratureIdElectro);
    std::vector<double> phiValuesQuadPoints(numberQuadraturePoints, 0.0);

    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;
    for (std::set<unsigned int>::iterator it =
           d_atomicShapeFnsContainer->d_feEvaluationMap.begin();
         it != d_atomicShapeFnsContainer->d_feEvaluationMap.begin();
         ++it)
      {
        unsigned int cell = *it;
        feEvalObj.reinit(cell);
        feEvalObj.read_dof_values(phiTotNodalValues);
        feEvalObj.evaluate(true, false);

        for (unsigned int iSubCell = 0;
             iSubCell < d_BasisOperatorElectroHostPtr->matrixFreeData()
                          .n_active_entries_per_cell_batch(cell);
             ++iSubCell)
          {
            subCellPtr =
              d_BasisOperatorElectroHostPtr->matrixFreeData().get_cell_iterator(
                cell, iSubCell, dofHandlerId);
            dealii::CellId subCellId = subCellPtr->id();
            unsigned int   cellIndex =
              d_BasisOperatorElectroHostPtr->cellIndex(subCellId);

            if (d_atomicShapeFnsContainer->atomSupportInElement(cellIndex))
              {
                double *tempVec = phiValuesQuadPoints.data();


                for (unsigned int q_point = 0; q_point < numberQuadraturePoints;
                     ++q_point)
                  {
                    tempVec[q_point] = feEvalObj.get_value(q_point)[iSubCell];
                  }
                std::vector<int> atomIdsInElem =
                  d_atomicShapeFnsContainer->getAtomIdsInElement(cellIndex);
                for (int iAtom = 0; iAtom < atomIdsInElem.size(); iAtom++)
                  {
                    const unsigned int atomId = atomIdsInElem[iAtom];
                    const unsigned int Znum   = atomicNumber[atomId];
                    const unsigned int NumTotalSphericalFunctions =
                      d_atomicShapeFnsContainer
                        ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                    std::vector<double> gLValues =
                      d_gLValuesQuadPoints[std::make_pair(atomId, cellIndex)];
                    d_BLASWrapperHostPtr->xgemm(
                      'N',
                      'N',
                      1,
                      NumTotalSphericalFunctions,
                      numberQuadraturePoints,
                      &alpha,
                      phiValuesQuadPoints.data(),
                      1,
                      &gLValues[0],
                      numberQuadraturePoints,
                      &beta,
                      d_nonLocalHamiltonianElectrostaticValue[atomId].data(),
                      1);



                  } // iAtom
              }
          }
      }
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeNonLocalHamiltonianEntries(
    const bool flagEnergy)
  {}

} // namespace dftfe
