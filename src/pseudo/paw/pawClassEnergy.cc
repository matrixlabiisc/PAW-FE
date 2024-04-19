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
#include <pawClass.h>
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
    pcout << "DEBUG: Line 31 pawClassEnergy" << std::endl;
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
    pcout << "g_LPhi(bx): " << std::endl;
    int iElem = 0;
    for (std::set<unsigned int>::iterator it =
           d_atomicShapeFnsContainer->d_feEvaluationMap.begin();
         it != d_atomicShapeFnsContainer->d_feEvaluationMap.end();
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
                    for (int L = 0; L < NumTotalSphericalFunctions; L++)
                      {
                        for (int iQuadPoint = 0;
                             iQuadPoint < numberQuadraturePoints;
                             iQuadPoint++)
                          pcout
                            << iElem << " " << L << " "
                            << gLValues[L * numberQuadraturePoints + iQuadPoint]
                            << " " << tempVec[iQuadPoint] << " "
                            << *(phiValuesQuadPoints.data() + iQuadPoint)
                            << std::endl;
                      }
                    iElem++;
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
              }     // if
          }         // subcell
      }             // FEEval iterator
    for (std::map<unsigned int, std::vector<double>>::iterator it =
           d_nonLocalHamiltonianElectrostaticValue.begin();
         it != d_nonLocalHamiltonianElectrostaticValue.end();
         ++it)
      {
        unsigned int        atomId  = it->first;
        std::vector<double> entries = it->second;
        for (int i = 0; i < entries.size(); i++)
          pcout << entries[i] << " ";
        pcout << std::endl;
      }
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeNonLocalHamiltonianEntries(
    const bool flagEnergy)
  {}

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::computeDeltaExchangeCorrelationEnergy(
    double &DeltaExchangeCorrelationVal)
  {
    double     TotalDeltaExchangeCorrelationVal = 0.0;
    const bool isGGA =
      d_excManagerPtr->getDensityBasedFamilyType() == densityFamilyType::GGA;
    std::vector<double>              quad_weights;
    std::vector<std::vector<double>> quad_points;
    getSphericalQuadratureRule(quad_weights, quad_points);
    double TotalDeltaXC = 0.0;

    int          numberofSphericalValues = quad_weights.size();
    unsigned int atomId                  = 0;
    if (d_LocallyOwnedAtomId.size() > 0)
      {
        const std::map<std::pair<unsigned int, unsigned int>,
                       std::shared_ptr<AtomCenteredSphericalFunctionBase>>
          sphericalFunction =
            d_atomicProjectorFnsContainer->getSphericalFunctions();
        std::vector<unsigned int> atomicNumbers =
          d_atomicProjectorFnsContainer->getAtomicNumbers();

        for (int iAtomList = 0; iAtomList < d_LocallyOwnedAtomId.size();
             iAtomList++)
          {
            atomId                         = d_LocallyOwnedAtomId[iAtomList];
            std::vector<double> Dij        = D_ij[TypeOfField::Out][atomId];
            const unsigned int  Znum       = atomicNumbers[atomId];
            const unsigned int  RmaxIndex  = d_RmaxAugIndex[Znum];
            std::vector<double> radialGrid = d_radialMesh[Znum];
            std::vector<double> rab        = d_radialJacobianData[Znum];
            unsigned int        RadialMeshSize = radialGrid.size();
            const unsigned int  numberofValues =
              std::min(RmaxIndex + 5, RadialMeshSize);
            TotalDeltaExchangeCorrelationVal -= d_coreXC[Znum];
            const unsigned int numberOfProjectors =
              d_atomicProjectorFnsContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
            const unsigned int numberOfRadialProjectors =
              d_atomicProjectorFnsContainer
                ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);
            const unsigned int numberOfProjectorsSq =
              numberOfProjectors * numberOfProjectors;
            double Yi, Yj;
            if (!isGGA)
              {
                double Yi, Yj;
                for (int qpoint = 0; qpoint < numberofSphericalValues; qpoint++)
                  {
                    std::vector<double> atomDensityAllelectron =
                      d_atomTypeCoreFlagMap[Znum] ?
                        d_atomCoreDensityAE[Znum] :
                        std::vector<double>(numberofValues, 0.0);
                    std::vector<double> atomDensitySmooth =
                      d_atomTypeCoreFlagMap[Znum] ?
                        d_atomCoreDensityPS[Znum] :
                        std::vector<double>(numberofValues, 0.0);
                    std::vector<double> SphericalHarmonics(numberOfProjectors *
                                                             numberOfProjectors,
                                                           0.0);
                    // help me.. A better strategy to store this

                    std::vector<double> productOfAEpartialWfc =
                      d_productOfAEpartialWfc[Znum];
                    std::vector<double> productOfPSpartialWfc =
                      d_productOfPSpartialWfc[Znum];
                    double              quadwt = quad_weights[qpoint];
                    std::vector<double> DijYij(numberOfProjectors *
                                                 numberOfProjectors,
                                               0.0);
                    int                 projIndexI = 0;
                    for (int iProj = 0; iProj < numberOfRadialProjectors;
                         iProj++)
                      {
                        std::shared_ptr<AtomCenteredSphericalFunctionBase>
                          sphFn_i =
                            sphericalFunction.find(std::make_pair(Znum, iProj))
                              ->second;
                        const int lQuantumNo_i = sphFn_i->getQuantumNumberl();
                        for (int mQuantumNumber_i = -lQuantumNo_i;
                             mQuantumNumber_i <= lQuantumNo_i;
                             mQuantumNumber_i++)
                          {
                            sphericalHarmonicUtils::getSphericalHarmonicVal(
                              quad_points[qpoint][0],
                              quad_points[qpoint][1],
                              lQuantumNo_i,
                              mQuantumNumber_i,
                              Yi);

                            int projIndexJ = 0;
                            for (int jProj = 0;
                                 jProj < numberOfRadialProjectors;
                                 jProj++)
                              {
                                std::shared_ptr<
                                  AtomCenteredSphericalFunctionBase>
                                  sphFn_j = sphericalFunction
                                              .find(std::make_pair(Znum, jProj))
                                              ->second;
                                const int lQuantumNo_j =
                                  sphFn_j->getQuantumNumberl();
                                for (int mQuantumNumber_j = -lQuantumNo_j;
                                     mQuantumNumber_j <= lQuantumNo_j;
                                     mQuantumNumber_j++)
                                  {
                                    sphericalHarmonicUtils::
                                      getSphericalHarmonicVal(
                                        quad_points[qpoint][0],
                                        quad_points[qpoint][1],
                                        lQuantumNo_j,
                                        mQuantumNumber_j,
                                        Yi);

                                    SphericalHarmonics[projIndexI *
                                                         numberOfProjectors +
                                                       projIndexJ] = Yi * Yj;
                                    SphericalHarmonics[projIndexJ *
                                                         numberOfProjectors +
                                                       projIndexI] = Yi * Yj;
                                    DijYij[projIndexI * numberOfProjectors +
                                           projIndexJ] =
                                      Yi * Yj *
                                      Dij[projIndexI * numberOfProjectors +
                                          projIndexJ];
                                    DijYij[projIndexJ * numberOfProjectors +
                                           projIndexI] =
                                      Yi * Yj *
                                      Dij[projIndexJ * numberOfProjectors +
                                          projIndexI];

                                    projIndexJ++;
                                  } // mQuantumNumber_j

                              } // jProj
                            projIndexI++;
                          } // mQuantumNumber_i



                      } // iProj
                    const char         transA = 'N', transB = 'N';
                    const double       Alpha = 1, Beta = 0.0;
                    const unsigned int inc   = 1;
                    const double       Beta2 = 1.0;

                    dgemm_(&transA,
                           &transB,
                           &inc,
                           &numberofValues,
                           &numberOfProjectorsSq,
                           &Alpha,
                           &DijYij[0],
                           &inc,
                           &productOfAEpartialWfc[0],
                           &numberOfProjectorsSq,
                           &Beta2,
                           &atomDensityAllelectron[0],
                           &inc);
                    // pcout<<"Line 2678"<<std::endl;
                    dgemm_(&transA,
                           &transB,
                           &inc,
                           &numberofValues,
                           &numberOfProjectorsSq,
                           &Alpha,
                           &DijYij[0],
                           &inc,
                           &productOfPSpartialWfc[0],
                           &numberOfProjectorsSq,
                           &Beta2,
                           &atomDensitySmooth[0],
                           &inc);


                    std::vector<double> exchangeEnergyValAE(numberofValues);
                    std::vector<double> corrEnergyValAE(numberofValues);
                    std::vector<double> exchangeEnergyValPS(numberofValues);
                    std::vector<double> corrEnergyValPS(numberofValues);
                    std::map<rhoDataAttributes, const std::vector<double> *>
                      rhoDataAE, rhoDataPS;



                    rhoDataAE[rhoDataAttributes::values] =
                      &atomDensityAllelectron;
                    rhoDataPS[rhoDataAttributes::values] = &atomDensitySmooth;


                    d_excManagerPtr->getExcDensityObj()
                      ->computeDensityBasedEnergyDensity(numberofValues,
                                                         rhoDataAE,
                                                         exchangeEnergyValAE,
                                                         corrEnergyValAE);
                    d_excManagerPtr->getExcDensityObj()
                      ->computeDensityBasedEnergyDensity(numberofValues,
                                                         rhoDataPS,
                                                         exchangeEnergyValPS,
                                                         corrEnergyValPS);

                    std::function<double(const unsigned int &)> Integral =
                      [&](const unsigned int &rpoint) {
                        double Val1 = atomDensityAllelectron[rpoint] *
                                      (exchangeEnergyValAE[rpoint] +
                                       corrEnergyValAE[rpoint]);
                        double Val2 = atomDensitySmooth[rpoint] *
                                      (exchangeEnergyValPS[rpoint] +
                                       corrEnergyValPS[rpoint]);
                        double Value = rab[rpoint] * (Val1 - Val2) *
                                       pow(radialGrid[rpoint], 2);
                        // pcout<<i<<" "<<Value<<std::endl;
                        return (Value);
                      };

                    double RadialIntegral =
                      simpsonIntegral(0, RmaxIndex + 1, Integral);
                    TotalDeltaXC += RadialIntegral * quadwt * 4.0 * M_PI;
                    TotalDeltaExchangeCorrelationVal +=
                      RadialIntegral * quadwt * 4.0 * M_PI;
                    exchangeEnergyValAE.resize(0);
                    exchangeEnergyValPS.resize(0);
                    corrEnergyValAE.resize(0);
                    corrEnergyValPS.resize(0);
                    atomDensityAllelectron.resize(0);
                    atomDensitySmooth.resize(0);

                    exchangeEnergyValAE.clear();
                    exchangeEnergyValPS.clear();
                    corrEnergyValAE.clear();
                    corrEnergyValPS.clear();
                    atomDensityAllelectron.clear();
                    atomDensitySmooth.clear();


                  } // qpoint

              } // LDA case
            else
              {
                const std::vector<double> &productOfAEpartialWfc =
                  d_productOfAEpartialWfc[Znum];
                const std::vector<double> &productOfPSpartialWfc =
                  d_productOfPSpartialWfc[Znum];
                const std::vector<double> &productDerCoreDensityWfcDerWfcAE =
                  d_productDerCoreDensityWfcDerWfcAE[Znum];
                const std::vector<double> &productDerCoreDensityWfcDerWfcPS =
                  d_productDerCoreDensityWfcDerWfcPS[Znum];



                const std::vector<double> &TensorWfcAE = d_tensorWfcAE[Znum];
                const std::vector<double> &TensorWfcPS = d_tensorWfcPS[Znum];
                const std::vector<double> &TensorWfcDerAE =
                  d_tensorWfcDerAE[Znum];
                const std::vector<double> &TensorWfcDerPS =
                  d_tensorWfcDerPS[Znum];
                std::vector<double> productOfPSpartialWfcDer =
                  d_productOfPSpartialWfcDer[Znum];
                std::vector<double> productOfAEpartialWfcDer =
                  d_productOfAEpartialWfcDer[Znum];
                std::vector<double> productOfPSpartialWfcVals =
                  d_productOfPSpartialWfcValue[Znum];
                std::vector<double> productOfAEpartialWfcVals =
                  d_productOfAEpartialWfcValue[Znum];


                // numberofSphericalValues = 1;
                for (int qpoint = 0; qpoint < numberofSphericalValues; qpoint++)
                  {
                    std::vector<double> atomDensityAllelectron =
                      d_atomTypeCoreFlagMap[Znum] ?
                        d_atomCoreDensityAE[Znum] :
                        std::vector<double>(numberofValues, 0.0);
                    std::vector<double> atomDensitySmooth =
                      d_atomTypeCoreFlagMap[Znum] ?
                        d_atomCoreDensityPS[Znum] :
                        std::vector<double>(numberofValues, 0.0);
                    std::vector<double> sigmaAllElectron =
                      d_atomTypeCoreFlagMap[Znum] ?
                        d_gradCoreSqAE[Znum] :
                        std::vector<double>(numberofValues, 0.0);
                    std::vector<double> sigmaSmooth =
                      d_atomTypeCoreFlagMap[Znum] ?
                        d_gradCoreSqPS[Znum] :
                        std::vector<double>(numberofValues, 0.0);
                    std::vector<double> SphericalHarmonics(numberOfProjectors *
                                                             numberOfProjectors,
                                                           0.0);
                    std::vector<double> GradThetaSphericalHarmonics(
                      numberOfProjectors * numberOfProjectors, 0.0);
                    std::vector<double> GradPhiSphericalHarmonics(
                      numberOfProjectors * numberOfProjectors, 0.0);
                    // help me.. A better strategy to store this



                    double quadwt = quad_weights[qpoint];
                    // pcout << "Storing the Dij and its variants: " <<
                    // std::endl;
                    std::vector<double> DijYij(numberOfProjectors *
                                                 numberOfProjectors,
                                               0.0);
                    std::vector<double> DijGradThetaYij(numberOfProjectors *
                                                          numberOfProjectors,
                                                        0.0);
                    std::vector<double> DijGradPhiYij(numberOfProjectors *
                                                        numberOfProjectors,
                                                      0.0);
                    // pcout << "Dij and its other Values: " << std::endl;
                    int projIndexI = 0;
                    for (int iProj = 0; iProj < numberOfRadialProjectors;
                         iProj++)
                      {
                        std::shared_ptr<AtomCenteredSphericalFunctionBase>
                          sphFn_i =
                            sphericalFunction.find(std::make_pair(Znum, iProj))
                              ->second;
                        const int lQuantumNo_i = sphFn_i->getQuantumNumberl();
                        for (int mQuantumNumber_i = -lQuantumNo_i;
                             mQuantumNumber_i <= lQuantumNo_i;
                             mQuantumNumber_i++)
                          {
                            sphericalHarmonicUtils::getSphericalHarmonicVal(
                              quad_points[qpoint][0],
                              quad_points[qpoint][1],
                              lQuantumNo_i,
                              mQuantumNumber_i,
                              Yi);

                            int projIndexJ = 0;
                            for (int jProj = 0;
                                 jProj < numberOfRadialProjectors;
                                 jProj++)
                              {
                                std::shared_ptr<
                                  AtomCenteredSphericalFunctionBase>
                                  sphFn_j = sphericalFunction
                                              .find(std::make_pair(Znum, jProj))
                                              ->second;
                                const int lQuantumNo_j =
                                  sphFn_j->getQuantumNumberl();
                                for (int mQuantumNumber_j = -lQuantumNo_j;
                                     mQuantumNumber_j <= lQuantumNo_j;
                                     mQuantumNumber_j++)
                                  {
                                    sphericalHarmonicUtils::
                                      getSphericalHarmonicVal(
                                        quad_points[qpoint][0],
                                        quad_points[qpoint][1],
                                        lQuantumNo_j,
                                        mQuantumNumber_j,
                                        Yi);

                                    std::vector<double> gradYj =
                                      derivativeOfRealSphericalHarmonic(
                                        lQuantumNo_j,
                                        lQuantumNo_j,
                                        quad_points[qpoint][0],
                                        quad_points[qpoint][1]);
                                    SphericalHarmonics[projIndexI *
                                                         numberOfProjectors +
                                                       projIndexJ] = Yi * Yj;
                                    DijYij[projIndexI * numberOfProjectors +
                                           projIndexJ] =
                                      Yi * Yj *
                                      Dij[projIndexI * numberOfProjectors +
                                          projIndexJ];

                                    GradThetaSphericalHarmonics
                                      [projIndexI * numberOfProjectors +
                                       projIndexJ] = Yi * gradYj[0];
                                    double temp =
                                      std::abs(std::sin(
                                        quad_points[qpoint][0])) <= 1E-8 ?
                                        0.0 :
                                        Yi * gradYj[1] /
                                          std::sin(quad_points[qpoint][0]);
                                    GradPhiSphericalHarmonics
                                      [projIndexI * numberOfProjectors +
                                       projIndexJ] = temp;

                                    DijGradThetaYij[projIndexI *
                                                      numberOfProjectors +
                                                    projIndexJ] =
                                      Dij[projIndexI * numberOfProjectors +
                                          projIndexJ] *
                                      Yi * gradYj[0];
                                    DijGradPhiYij[projIndexI *
                                                    numberOfProjectors +
                                                  projIndexJ] =
                                      Dij[projIndexI * numberOfProjectors +
                                          projIndexJ] *
                                      temp;

                                    projIndexJ++;
                                  } // mQuantumNumber_j

                              } // jProj
                            projIndexI++;
                          } // mQuantumNumber_i



                      } // iProj

                    const char         transA = 'N', transB = 'N';
                    const double       Alpha = 1, Beta = 0.0;
                    const double       Alphasigma1 = 4.0;
                    const unsigned int inc         = 1;
                    const double       Beta2       = 1.0;
                    // Computing Density for Libxc
                    // pcout << "Starting LDA contribution term: " << std::endl;
                    // MPI_Barrier(d_mpiCommParent);
                    // double TimerLDAStart = MPI_Wtime();
                    dgemm_(&transA,
                           &transB,
                           &inc,
                           &numberofValues,
                           &numberOfProjectorsSq,
                           &Alpha,
                           &DijYij[0],
                           &inc,
                           &productOfAEpartialWfc[0],
                           &numberOfProjectorsSq,
                           &Beta2,
                           &atomDensityAllelectron[0],
                           &inc);
                    dgemm_(&transA,
                           &transB,
                           &inc,
                           &numberofValues,
                           &numberOfProjectorsSq,
                           &Alpha,
                           &DijYij[0],
                           &inc,
                           &productOfPSpartialWfc[0],
                           &numberOfProjectorsSq,
                           &Beta2,
                           &atomDensitySmooth[0],
                           &inc);
                    // MPI_Barrier(d_mpiCommParent);
                    // timerLDAContribution += (MPI_Wtime() - TimerLDAStart);

                    // MPI_Barrier(d_mpiCommParent);
                    // double TimerGGAStart = MPI_Wtime();

                    // MPI_Barrier(d_mpiCommParent);
                    // double TimerGGA0Start = MPI_Wtime();
                    if (d_atomTypeCoreFlagMap[Znum])
                      {
                        // pcout << "Starting Sigma Contribution part0" <<
                        // std::endl;
                        dgemm_(&transA,
                               &transB,
                               &inc,
                               &numberofValues,
                               &numberOfProjectorsSq,
                               &Alphasigma1,
                               &DijYij[0],
                               &inc,
                               &productDerCoreDensityWfcDerWfcAE[0],
                               &numberOfProjectorsSq,
                               &Beta2,
                               &sigmaAllElectron[0],
                               &inc);
                        dgemm_(&transA,
                               &transB,
                               &inc,
                               &numberofValues,
                               &numberOfProjectorsSq,
                               &Alphasigma1,
                               &DijYij[0],
                               &inc,
                               &productDerCoreDensityWfcDerWfcPS[0],
                               &numberOfProjectorsSq,
                               &Beta2,
                               &sigmaSmooth[0],
                               &inc);

                        // pcout << "Finished Sigma Contribution part0" <<
                        // std::endl;
                      }
                    // MPI_Barrier(d_mpiCommParent);
                    // timerGGA0Contribution += MPI_Wtime() - TimerGGA0Start;


                    std::vector<double> tempAEcontributionA(
                      numberOfProjectorsSq * numberofValues, 0.0);
                    std::vector<double> tempPScontributionA(
                      numberOfProjectorsSq * numberofValues, 0.0);
                    const unsigned int numValsTimesnpjsq =
                      numberofValues * numberOfProjectorsSq;
                    // Part1 of Tensor Contraction for A
                    std::vector<double> tempAETrialA(numberofValues, 0.0);
                    std::vector<double> tempPSTrialA(numberofValues, 0.0);
                    // MPI_Barrier(d_mpiCommParent);
                    // double TimerGGAAStart = MPI_Wtime();
                    dgemm_(&transA,
                           &transB,
                           &inc,
                           &numberofValues,
                           &numberOfProjectorsSq,
                           &Alpha,
                           &DijYij[0],
                           &inc,
                           &productOfAEpartialWfcDer[0],
                           &numberOfProjectorsSq,
                           &Beta,
                           &tempAETrialA[0],
                           &inc);
                    dgemm_(&transA,
                           &transB,
                           &inc,
                           &numberofValues,
                           &numberOfProjectorsSq,
                           &Alpha,
                           &DijYij[0],
                           &inc,
                           &productOfPSpartialWfcDer[0],
                           &numberOfProjectorsSq,
                           &Beta,
                           &tempPSTrialA[0],
                           &inc);
                    for (int iRad = 0; iRad < numberofValues; iRad++)
                      {
                        const double scaleAE = tempAETrialA[iRad];
                        const double scalePS = tempPSTrialA[iRad];
                        unsigned int index   = iRad * numberOfProjectorsSq;
                        daxpy_(&numberOfProjectorsSq,
                               &scaleAE,
                               &productOfAEpartialWfcDer[index],
                               &inc,
                               &tempAEcontributionA[index],
                               &inc);
                        daxpy_(&numberOfProjectorsSq,
                               &scalePS,
                               &productOfPSpartialWfcDer[index],
                               &inc,
                               &tempPScontributionA[index],
                               &inc);
                      }

                    // Part2 of TensorContraction  for A

                    dgemm_(&transA,
                           &transB,
                           &inc,
                           &numberofValues,
                           &numberOfProjectorsSq,
                           &Alphasigma1,
                           &DijYij[0],
                           &inc,
                           &tempAEcontributionA[0],
                           &numberOfProjectorsSq,
                           &Beta2,
                           &sigmaAllElectron[0],
                           &inc);
                    dgemm_(&transA,
                           &transB,
                           &inc,
                           &numberofValues,
                           &numberOfProjectorsSq,
                           &Alphasigma1,
                           &DijYij[0],
                           &inc,
                           &tempPScontributionA[0],
                           &numberOfProjectorsSq,
                           &Beta2,
                           &sigmaSmooth[0],
                           &inc);
                    // pcout << "Finished Sigma Contribution part1 B" <<
                    // std::endl;
                    // MPI_Barrier(d_mpiCommParent);
                    // timerGGAAContribution += (MPI_Wtime() - TimerGGAAStart);


                    std::vector<double> tempAEcontributionB(
                      numberOfProjectorsSq * numberofValues, 0.0);
                    std::vector<double> tempPScontributionB(
                      numberOfProjectorsSq * numberofValues, 0.0);
                    std::vector<double> tempAETrialB(numberofValues, 0.0);
                    std::vector<double> tempPSTrialB(numberofValues, 0.0);
                    // Part1 of Tensor Contraction for B
                    // MPI_Barrier(d_mpiCommParent);
                    // double TimerGGABStart = MPI_Wtime();
                    dgemm_(&transA,
                           &transB,
                           &inc,
                           &numberofValues,
                           &numberOfProjectorsSq,
                           &Alpha,
                           &DijGradThetaYij[0],
                           &inc,
                           &productOfAEpartialWfcVals[0],
                           &numberOfProjectorsSq,
                           &Beta,
                           &tempAETrialB[0],
                           &inc);
                    dgemm_(&transA,
                           &transB,
                           &inc,
                           &numberofValues,
                           &numberOfProjectorsSq,
                           &Alpha,
                           &DijGradThetaYij[0],
                           &inc,
                           &productOfPSpartialWfcVals[0],
                           &numberOfProjectorsSq,
                           &Beta,
                           &tempPSTrialB[0],
                           &inc);
                    for (int iRad = 0; iRad < numberofValues; iRad++)
                      {
                        const double scaleAE =
                          tempAETrialB[iRad] * pow(radialGrid[iRad], 2);
                        const double scalePS =
                          tempPSTrialB[iRad] * pow(radialGrid[iRad], 2);
                        unsigned int index = iRad * numberOfProjectorsSq;
                        daxpy_(&numberOfProjectorsSq,
                               &scaleAE,
                               &productOfAEpartialWfcVals[index],
                               &inc,
                               &tempAEcontributionB[index],
                               &inc);
                        daxpy_(&numberOfProjectorsSq,
                               &scalePS,
                               &productOfPSpartialWfcVals[index],
                               &inc,
                               &tempPScontributionB[index],
                               &inc);
                      }


                    // pcout << "Finished Sigma Contribution part2 A" <<
                    // std::endl;
                    // Part2 of TensorContraction  for B

                    dgemm_(&transA,
                           &transB,
                           &inc,
                           &numberofValues,
                           &numberOfProjectorsSq,
                           &Alphasigma1,
                           &DijGradThetaYij[0],
                           &inc,
                           &tempAEcontributionB[0],
                           &numberOfProjectorsSq,
                           &Beta2,
                           &sigmaAllElectron[0],
                           &inc);
                    dgemm_(&transA,
                           &transB,
                           &inc,
                           &numberofValues,
                           &numberOfProjectorsSq,
                           &Alphasigma1,
                           &DijGradThetaYij[0],
                           &inc,
                           &tempPScontributionB[0],
                           &numberOfProjectorsSq,
                           &Beta2,
                           &sigmaSmooth[0],
                           &inc);
                    // MPI_Barrier(d_mpiCommParent);
                    // timerGGABContribution += (MPI_Wtime() - TimerGGABStart);
                    // pcout << "Finished Sigma Contribution part2 B" <<
                    // std::endl;

                    std::vector<double> tempAEcontributionC(
                      numberOfProjectorsSq * numberofValues, 0.0);
                    std::vector<double> tempPScontributionC(
                      numberOfProjectorsSq * numberofValues, 0.0);

                    std::vector<double> tempAETrialC(numberofValues, 0.0);
                    std::vector<double> tempPSTrialC(numberofValues, 0.0);

                    // Part1 of Tensor Contraction for C
                    // MPI_Barrier(d_mpiCommParent);
                    // double TimerGGACStart = MPI_Wtime();
                    dgemm_(&transA,
                           &transB,
                           &inc,
                           &numberofValues,
                           &numberOfProjectorsSq,
                           &Alpha,
                           &DijGradPhiYij[0],
                           &inc,
                           &productOfAEpartialWfcVals[0],
                           &numberOfProjectorsSq,
                           &Beta2,
                           &tempAETrialC[0],
                           &inc);
                    dgemm_(&transA,
                           &transB,
                           &inc,
                           &numberofValues,
                           &numberOfProjectorsSq,
                           &Alpha,
                           &DijGradPhiYij[0],
                           &inc,
                           &productOfPSpartialWfcVals[0],
                           &numberOfProjectorsSq,
                           &Beta2,
                           &tempPSTrialC[0],
                           &inc);
                    for (int iRad = 0; iRad < numberofValues; iRad++)
                      {
                        const double scaleAE =
                          tempAETrialC[iRad] * pow(radialGrid[iRad], 2);
                        const double scalePS =
                          tempPSTrialC[iRad] * pow(radialGrid[iRad], 2);
                        unsigned int index = iRad * numberOfProjectorsSq;
                        daxpy_(&numberOfProjectorsSq,
                               &scaleAE,
                               &productOfAEpartialWfcVals[index],
                               &inc,
                               &tempAEcontributionC[index],
                               &inc);
                        daxpy_(&numberOfProjectorsSq,
                               &scalePS,
                               &productOfPSpartialWfcVals[index],
                               &inc,
                               &tempPScontributionC[index],
                               &inc);
                      }



                    dgemm_(&transA,
                           &transB,
                           &inc,
                           &numberofValues,
                           &numberOfProjectorsSq,
                           &Alphasigma1,
                           &DijGradPhiYij[0],
                           &inc,
                           &tempAEcontributionC[0],
                           &numberOfProjectorsSq,
                           &Beta2,
                           &sigmaAllElectron[0],
                           &inc);
                    dgemm_(&transA,
                           &transB,
                           &inc,
                           &numberofValues,
                           &numberOfProjectorsSq,
                           &Alphasigma1,
                           &DijGradPhiYij[0],
                           &inc,
                           &tempPScontributionC[0],
                           &numberOfProjectorsSq,
                           &Beta2,
                           &sigmaSmooth[0],
                           &inc);



                    std::map<rhoDataAttributes, const std::vector<double> *>
                                        rhoDataAE, rhoDataPS;
                    std::vector<double> exchangeEnergyValAE(numberofValues);
                    std::vector<double> corrEnergyValAE(numberofValues);
                    std::vector<double> exchangeEnergyValPS(numberofValues);
                    std::vector<double> corrEnergyValPS(numberofValues);



                    rhoDataAE[rhoDataAttributes::values] =
                      &atomDensityAllelectron;
                    rhoDataPS[rhoDataAttributes::values] = &atomDensitySmooth;
                    rhoDataAE[rhoDataAttributes::sigmaGradValue] =
                      &sigmaAllElectron;
                    rhoDataPS[rhoDataAttributes::sigmaGradValue] = &sigmaSmooth;



                    d_excManagerPtr->getExcDensityObj()
                      ->computeDensityBasedEnergyDensity(numberofValues,
                                                         rhoDataAE,
                                                         exchangeEnergyValAE,
                                                         corrEnergyValAE);
                    d_excManagerPtr->getExcDensityObj()
                      ->computeDensityBasedEnergyDensity(numberofValues,
                                                         rhoDataPS,
                                                         exchangeEnergyValPS,
                                                         corrEnergyValPS);



                    std::function<double(const unsigned int &)> Integral =
                      [&](const unsigned int &rpoint) {
                        double Val1 = atomDensityAllelectron[rpoint] *
                                      (exchangeEnergyValAE[rpoint] +
                                       corrEnergyValAE[rpoint]);
                        double Val2 = atomDensitySmooth[rpoint] *
                                      (exchangeEnergyValPS[rpoint] +
                                       corrEnergyValPS[rpoint]);
                        double Value = rab[rpoint] * (Val1 - Val2) *
                                       pow(radialGrid[rpoint], 2);
                        // pcout<<i<<" "<<Value<<std::endl;
                        return (Value);
                      };

                    double RadialIntegral =
                      simpsonIntegral(0, RmaxIndex + 1, Integral);
                    TotalDeltaXC += RadialIntegral * quadwt * 4.0 * M_PI;
                    TotalDeltaExchangeCorrelationVal +=
                      RadialIntegral * quadwt * 4.0 * M_PI;
                    exchangeEnergyValAE.resize(0);
                    exchangeEnergyValPS.resize(0);
                    corrEnergyValAE.resize(0);
                    corrEnergyValPS.resize(0);
                    atomDensityAllelectron.resize(0);
                    atomDensitySmooth.resize(0);

                    exchangeEnergyValAE.clear();
                    exchangeEnergyValPS.clear();
                    corrEnergyValAE.clear();
                    corrEnergyValPS.clear();
                    atomDensityAllelectron.clear();
                    atomDensitySmooth.clear();


                  } // qpoint
              }
          } // iAtomList
      }     // If locallyOwned
    DeltaExchangeCorrelationVal =
      (dealii::Utilities::MPI::sum(TotalDeltaExchangeCorrelationVal,
                                   d_mpiCommParent));

    return (dealii::Utilities::MPI::sum(TotalDeltaXC, d_mpiCommParent));
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeCoreDeltaExchangeCorrelationEnergy()
  {
    for (std::set<unsigned int>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        double     TotalDeltaXC = 0.0;
        const bool isGGA = d_excManagerPtr->getDensityBasedFamilyType() ==
                           densityFamilyType::GGA;

        if (d_atomTypeCoreFlagMap[*it])
          {
            unsigned int        Znum       = *it;
            std::vector<double> RadialMesh = d_radialMesh[Znum];
            unsigned int        RmaxIndex  = d_RmaxAugIndex[Znum];

            std::vector<double> rab            = d_radialJacobianData[Znum];
            unsigned int        RadialMeshSize = RadialMesh.size();
            const unsigned int  numberofValues = RadialMesh.size();
            // std::min(RadialMeshSize, d_refinedRmaxAugIndex[Zno] + 5);
            std::vector<double> atomDensityAllelectron =
              d_atomTypeCoreFlagMap[Znum] ?
                d_atomCoreDensityAE[Znum] :
                std::vector<double>(numberofValues, 0.0);
            std::vector<double> atomSigmaAllelectron =
              d_atomTypeCoreFlagMap[Znum] ?
                d_gradCoreSqAE[Znum] :
                std::vector<double>(numberofValues, 0.0);
            std::vector<double> exchangeEnergyValAE(numberofValues);
            std::vector<double> corrEnergyValAE(numberofValues);
            std::map<rhoDataAttributes, const std::vector<double> *> rhoDataAE;
            rhoDataAE[rhoDataAttributes::values] = &atomDensityAllelectron;
            if (isGGA)
              rhoDataAE[rhoDataAttributes::sigmaGradValue] =
                &atomSigmaAllelectron;

            d_excManagerPtr->getExcDensityObj()
              ->computeDensityBasedEnergyDensity(numberofValues,
                                                 rhoDataAE,
                                                 exchangeEnergyValAE,
                                                 corrEnergyValAE);
            double                                      RadialIntegral = 0.0;
            std::function<double(const unsigned int &)> Integral =
              [&](const unsigned int &rpoint) {
                double Val1 =
                  atomDensityAllelectron[rpoint] *
                  (exchangeEnergyValAE[rpoint] + corrEnergyValAE[rpoint]);

                double Value = rab[rpoint] * (Val1)*pow(RadialMesh[rpoint], 2);
                // pcout << rpoint << " " << RadialMesh[rpoint] << " "
                //       << Value << std::endl;
                return (Value);
              };

            RadialIntegral = simpsonIntegral(0, numberofValues - 2, Integral);
            exchangeEnergyValAE.resize(0);
            corrEnergyValAE.resize(0);
            atomDensityAllelectron.resize(0);
            exchangeEnergyValAE.clear();
            corrEnergyValAE.clear();
            atomDensityAllelectron.clear();
            TotalDeltaXC = RadialIntegral;
          } // if core
        d_coreXC[*it] = TotalDeltaXC * (4 * M_PI);
        pcout << "Core contribution: " << d_coreXC[*it] << std::endl;
      } // *it
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::computeTotalDeltaEnergy()
  {}
  template class pawClass<dataTypes::number, dftfe::utils::MemorySpace::HOST>;
#if defined(DFTFE_WITH_DEVICE)
  template class pawClass<dataTypes::number, dftfe::utils::MemorySpace::DEVICE>;
#endif

} // namespace dftfe
