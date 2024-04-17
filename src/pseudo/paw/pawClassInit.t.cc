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
  pawClass<ValueType, memorySpace>::pawClass(
    const MPI_Comm &                            mpi_comm_parent,
    const std::string &                         scratchFolderName,
    dftParameters *                             dftParamsPtr,
    const std::set<unsigned int> &              atomTypes,
    const bool                                  floatingNuclearCharges,
    const unsigned int                          nOMPThreads,
    const std::map<unsigned int, unsigned int> &atomAttributes,
    const bool                                  reproducibleOutput,
    const int                                   verbosity,
    const bool                                  useDevice)
    : d_mpiCommParent(mpi_comm_parent)
    , d_this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent))
    , d_n_mpi_processes(
        dealii::Utilities::MPI::n_mpi_processes(mpi_comm_parent))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
    , d_dftParamsPtr(dftParamsPtr)
  {
    d_dftfeScratchFolderName = scratchFolderName;
    d_atomTypes              = atomTypes;
    d_floatingNuclearCharges = floatingNuclearCharges;
    d_nOMPThreads            = nOMPThreads;
    d_reproducible_output    = reproducibleOutput;
    d_verbosity              = verbosity;
    d_atomTypeAtributes      = atomAttributes;
    d_useDevice              = useDevice;
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType,
           memorySpace>::createAtomCenteredSphericalFunctionsForShapeFunctions()
  {
    for (std::set<unsigned int>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        char         pseudoAtomDataFile[256];
        unsigned int cumulativeSplineId = 0;
        strcpy(pseudoAtomDataFile,
               (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                "/PseudoAtomDat")
                 .c_str());

        unsigned int  Znum = *it;
        std::ifstream readPseudoDataFileNames(pseudoAtomDataFile);
        for (int i = 0; i < 4; i++)
          readPseudoDataFileNames.ignore();
        unsigned int shapeFnType;
        readPseudoDataFileNames >> shapeFnType;
        double rc;
        readPseudoDataFileNames >> rc;
        unsigned int lmaxAug = d_dftParamsPtr->noShapeFnsInPAW;
        double       rmaxAug = d_RmaxAug[*it];
        for (unsigned int lQuantumNo = 0; lQuantumNo < lmaxAug; lQuantumNo++)
          {
            if (shapeFnType == 0)
              {
                // Bessel Function
                d_atomicShapeFnsMap[std::make_pair(Znum, lQuantumNo)] =
                  std::make_shared<AtomCenteredSphericalFunctionBessel>(
                    rc, rmaxAug, lQuantumNo);
              }
            else if (shapeFnType == 1)
              {
                // Gauss Function
                d_atomicShapeFnsMap[std::make_pair(Znum, lQuantumNo)] =
                  std::make_shared<AtomCenteredSphericalFunctionGaussian>(
                    rc, rmaxAug, lQuantumNo);
              }
            else if (shapeFnType == 2)
              {
                // sinc Function
                d_atomicShapeFnsMap[std::make_pair(Znum, lQuantumNo)] =
                  std::make_shared<AtomCenteredSphericalFunctionSinc>(
                    rc, rmaxAug, lQuantumNo);
              }
            else
              {
                char shapeFnFile[256];
                strcpy(shapeFnFile,
                       (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                        "/shape_functions.dat")
                         .c_str());
                d_atomicShapeFnsMap[std::make_pair(Znum, lQuantumNo)] =
                  std::make_shared<
                    AtomCenteredSphericalFunctionProjectorSpline>(shapeFnFile,
                                                                  lQuantumNo,
                                                                  0,
                                                                  lQuantumNo +
                                                                    1,
                                                                  lmaxAug + 1,
                                                                  1E-12);
              }
          }
        std::vector<double> radialMesh   = d_radialMesh[*it];
        std::vector<double> jacobianData = d_radialJacobianData[*it];
        unsigned int        rmaxAugIndex = d_RmaxAugIndex[*it];
        unsigned int        numValues    = radialMesh.size();
        std::vector<double> shapeFnGridData(lmaxAug * numValues, 0.0);
        for (unsigned int lQuantumNo = 0; lQuantumNo < lmaxAug; lQuantumNo++)
          {
            double normalizationalizationConstant = 0.0;
            std::function<double(const unsigned int &)> f =
              [&](const unsigned int &i) {
                double Value =
                  jacobianData[i] *
                  d_atomicShapeFnsMap[std::make_pair(Znum, lQuantumNo)]
                    ->getRadialValue(radialMesh[i]) *
                  pow(radialMesh[i], lQuantumNo + 2);
                // pcout<<i<<" "<<Value<<std::endl;
                return (Value);
              };
            pcout << "Computing Normalization Constant for ShapeFn:  "
                  << lQuantumNo << " of Znum: " << Znum << " ";
            normalizationalizationConstant = // normalizationConstant2;
              simpsonIntegral(0, rmaxAugIndex + 1, f);
            pcout << "Normalization Constant Value: "
                  << normalizationalizationConstant << std::endl;
            for (int iRow = 0; iRow < numValues; iRow++)
              {
                shapeFnGridData[lQuantumNo * numValues + iRow] =
                  d_atomicShapeFnsMap[std::make_pair(Znum, lQuantumNo)]
                    ->getRadialValue(radialMesh[iRow]) /
                  normalizationalizationConstant;
              }
          }
        d_atomicShapeFn[*it] = shapeFnGridData;



      } //*it
  }


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType,
           memorySpace>::createAtomCenteredSphericalFunctionsForDensities()
  {
    d_atomicCoreDensityVector.clear();
    d_atomicCoreDensityVector.resize(d_nOMPThreads);
    d_atomicValenceDensityVector.clear();
    d_atomicValenceDensityVector.resize(d_nOMPThreads);

    for (std::set<unsigned int>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        unsigned int atomicNumber = *it;
        char         valenceDataFile[256];
        strcpy(valenceDataFile,
               (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                "/pseudo_valence_density.dat")
                 .c_str());
        char coreDataFileAE[256];
        strcpy(coreDataFileAE,
               (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                "/ae_core_density.dat")
                 .c_str());
        char coreDataFilePS[256];
        strcpy(coreDataFilePS,
               (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                "/pseudo_core_density.dat")
                 .c_str());

        for (unsigned int i = 0; i < d_nOMPThreads; i++)
          {
            d_atomicValenceDensityVector[i][*it] = std::make_shared<
              AtomCenteredSphericalFunctionValenceDensitySpline>(
              valenceDataFile, 1E-10, true);
            d_atomicCoreDensityVector[i][*it] =
              std::make_shared<AtomCenteredSphericalFunctionCoreDensitySpline>(
                coreDataFilePS, 1E-12, true);
          }
        if (d_atomicCoreDensityVector[0][atomicNumber]->isDataPresent())
          d_atomTypeCoreFlagMap[atomicNumber] = true;
        else
          d_atomTypeCoreFlagMap[atomicNumber] = false;
        std::vector<double> radialMesh     = d_radialMesh[*it];
        unsigned int        numValues      = radialMesh.size();
        std::vector<double> jacobianValues = d_radialJacobianData[*it];
        std::vector<double> radialAECoreDensity(numValues, 0.00);
        std::vector<double> radialPSCoreDensity(numValues, 0.00);

        if (d_atomicCoreDensityVector[0][atomicNumber]->isDataPresent())
          {
            std::vector<std::vector<double>> AECoreDensityData(0),
              PSCoreDensityData(0);
            dftUtils::readFile(2, AECoreDensityData, coreDataFileAE);
            dftUtils::readFile(2, PSCoreDensityData, coreDataFilePS);

            for (unsigned int iRow = 0; iRow < numValues; iRow++)
              {
                radialAECoreDensity[iRow] = AECoreDensityData[iRow][1];
                radialPSCoreDensity[iRow] = PSCoreDensityData[iRow][1];
              }

            pcout << "Radial Derivative of Nc" << std::endl;
            d_radialCoreDerAE[*it] =
              radialDerivativeOfMeshData(radialMesh,
                                         jacobianValues,
                                         radialAECoreDensity);
            pcout << "Radial Derivative of TIlde Nc" << std::endl;
            d_radialCoreDerPS[*it] =
              radialDerivativeOfMeshData(radialMesh,
                                         jacobianValues,
                                         radialPSCoreDensity);
          }
        else
          {
            d_radialCoreDerAE[*it] = std::vector<double>(numValues, 0.0);
            d_radialCoreDerPS[*it] = std::vector<double>(numValues, 0.0);
          }
        d_atomCoreDensityAE[*it] = radialAECoreDensity;
        d_atomCoreDensityPS[*it] = radialPSCoreDensity;
        double charge            = double(*it) / sqrt(4.0 * M_PI);
        d_DeltaL0coeff[*it]      = -charge;
        if (d_atomTypeCoreFlagMap[atomicNumber])
          {
            std::function<double(const unsigned int &)> f =
              [&](const unsigned int &i) {
                double Value = jacobianValues[i] * radialAECoreDensity[i] *
                               pow(radialMesh[i], 2);
                // pcout<<i<<" "<<Value<<std::endl;
                return (Value);
              };
            double Q1 = simpsonIntegral(0, radialAECoreDensity.size() - 1, f);
            pcout
              << "PAW Initialization: Integral All Electron CoreDensity from Radial integration: "
              << Q1 * sqrt(4 * M_PI) << std::endl;
            std::function<double(const unsigned int &)> g =
              [&](const unsigned int &i) {
                double Value = jacobianValues[i] * radialPSCoreDensity[i] *
                               pow(radialMesh[i], 2);
                // pcout<<i<<" "<<Value<<std::endl;
                return (Value);
              };
            double Q2 = simpsonIntegral(0, radialAECoreDensity.size() - 1, g);
            pcout
              << "PAW Initialization: Integral Pseudo Smooth CoreDensity from Radial integration: "
              << Q2 * sqrt(4 * M_PI) << std::endl;
            d_DeltaL0coeff[*it] = (Q1 - Q2);
          }

      } //*it loop
  }



  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::initialise(
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
      basisOperationsHostPtr,
#if defined(DFTFE_WITH_DEVICE)
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::DEVICE>>
      basisOperationsDevicePtr,
#endif
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
      basisOperationsElectroHostPtr,
#if defined(DFTFE_WITH_DEVICE)
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::DEVICE>>
      basisOperationsElectroDevicePtr,
#endif
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
      BLASWrapperPtrHost,
#if defined(DFTFE_WITH_DEVICE)
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
      BLASWrapperPtrDevice,
#endif
    unsigned int                            densityQuadratureId,
    unsigned int                            localContributionQuadratureId,
    unsigned int                            sparsityPatternQuadratureId,
    unsigned int                            nlpspQuadratureId,
    unsigned int                            densityQuadratureIdElectro,
    std::shared_ptr<excManager>             excFunctionalPtr,
    const std::vector<std::vector<double>> &atomLocations,
    unsigned int                            numEigenValues,
    unsigned int compensationChargeQuadratureIdElectro,
    std::map<dealii::CellId, std::vector<double>> &bQuadValuesAllAtoms)
  {
    MPI_Barrier(d_mpiCommParent);
    d_BasisOperatorHostPtr        = basisOperationsHostPtr;
    d_BLASWrapperHostPtr          = BLASWrapperPtrHost;
    d_BasisOperatorElectroHostPtr = basisOperationsElectroHostPtr;
#if defined(DFTFE_WITH_DEVICE)
    d_BLASWrapperDevicePtr          = BLASWrapperPtrDevice;
    d_BasisOperatorDevicePtr        = basisOperationsDevicePtr;
    d_BasisOperatorElectroDevicePtr = basisOperationsElectroDevicePtr;
#endif

    d_bQuadValuesAllAtoms = &bQuadValuesAllAtoms;
    std::vector<unsigned int> atomicNumbers;
    for (int iAtom = 0; iAtom < atomLocations.size(); iAtom++)
      {
        atomicNumbers.push_back(atomLocations[iAtom][0]);
      }

    d_densityQuadratureId           = densityQuadratureId;
    d_localContributionQuadratureId = localContributionQuadratureId;
    d_densityQuadratureIdElectro    = densityQuadratureIdElectro;
    d_sparsityPatternQuadratureId   = sparsityPatternQuadratureId;
    d_nlpspQuadratureId             = nlpspQuadratureId;
    d_excManagerPtr                 = excFunctionalPtr;
    d_numEigenValues                = numEigenValues;
    d_compensationChargeQuadratureIdElectro =
      compensationChargeQuadratureIdElectro;
    // Read Derivative File
    for (std::set<unsigned int>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        char DerivativeFileName[256];
        strcpy(DerivativeFileName,
               (d_dftfeScratchFolderName + "/z" + std::to_string(*it) + "/" +
                "derivatives.dat")
                 .c_str());
        std::vector<std::vector<double>> derivativeData(0);
        dftUtils::readFile(2, derivativeData, DerivativeFileName);
        unsigned int        RmaxIndex = 0;
        unsigned int        maxIndex  = derivativeData.size();
        std::vector<double> RadialMesh(maxIndex, 0.0);
        std::vector<double> radialDerivative(maxIndex, 0.0);
        char                pseudoAtomDataFile[256];
        strcpy(pseudoAtomDataFile,
               (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                "/PseudoAtomDat")
                 .c_str());
        std::ifstream readPseudoDataFileNames(pseudoAtomDataFile);
        char          isCore;
        double        coreKE;
        double        RmaxAug;
        if (readPseudoDataFileNames.is_open())
          {
            readPseudoDataFileNames >> isCore;
            readPseudoDataFileNames >> RmaxAug;
            if (isCore == 'T')
              readPseudoDataFileNames >> coreKE;
            else
              coreKE = 0.0;
            d_RmaxAug[*it] = RmaxAug;
            d_coreKE[*it]  = coreKE;
          }
        double deltaR = 1000;
        for (int iRow = 0; iRow < maxIndex; iRow++)
          {
            RadialMesh[iRow]       = derivativeData[iRow][0];
            radialDerivative[iRow] = derivativeData[iRow][1];
            if (std::fabs(RadialMesh[iRow] - RmaxAug) < deltaR)
              {
                RmaxIndex = iRow;
                deltaR    = std::fabs(RadialMesh[iRow] - RmaxAug);
              }
          }
        d_radialMesh[*it]         = RadialMesh;
        d_radialJacobianData[*it] = radialDerivative;
        pcout << "PAW Initialization: Rmax Index is: " << RmaxIndex
              << std::endl;
        pcout
          << "PAW Initialization: Difference in RmaxAug with xml augmentation Radius: "
          << deltaR << std::endl;
        d_RmaxAugIndex[*it] = RmaxIndex;
        pcout
          << "PAW Initializaion: Warning! make sure the above value is not large!!"
          << std::endl;
        double Rold = d_RmaxAug[*it];
        if (deltaR > 1E-8)
          d_RmaxAug[*it] = RadialMesh[RmaxIndex];
        pcout << "PAW Initialization: Warning!! PAW RmaxAug is reset to: "
              << d_RmaxAug[*it] << " from: " << Rold << std::endl;
      }
    // Reading ZeroPotential Data
    createAtomCenteredSphericalFunctionsForZeroPotential();
    // Reading Core Density Data
    createAtomCenteredSphericalFunctionsForDensities();
    // Reading Projectors/partial and PS partial waves Data
    createAtomCenteredSphericalFunctionsForProjectors();
    // Rading Shapefunctions Data
    createAtomCenteredSphericalFunctionsForShapeFunctions();
    // COmputing Various Radial Quantities on the Radial Grid



    d_atomicProjectorFnsContainer =
      std::make_shared<AtomCenteredSphericalFunctionContainer>();

    d_atomicProjectorFnsContainer->init(atomicNumbers, d_atomicProjectorFnsMap);

    d_atomicShapeFnsContainer =
      std::make_shared<AtomCenteredSphericalFunctionContainer>();

    d_atomicShapeFnsContainer->init(atomicNumbers, d_atomicShapeFnsMap);

    if (!d_useDevice)
      {
        if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
          d_nonLocalOperator = std::make_shared<
            AtomicCenteredNonLocalOperator<ValueType, memorySpace>>(
            d_BLASWrapperHostPtr,
            d_BasisOperatorHostPtr,
            d_atomicProjectorFnsContainer,
            d_mpiCommParent);
      }
#if defined(DFTFE_WITH_DEVICE)
    else
      {
        if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
          d_nonLocalOperator = std::make_shared<
            AtomicCenteredNonLocalOperator<ValueType, memorySpace>>(
            d_BLASWrapperDevicePtr,
            d_BasisOperatorDevicePtr,
            d_atomicProjectorFnsContainer,
            d_mpiCommParent);
      }
#endif

    computeRadialMultipoleData();
    computeMultipoleInverse();
    computeNonlocalPseudoPotentialConstants(CouplingType::pawOverlapEntries);
    initialiseKineticEnergyCorrection();
    initialiseColoumbicEnergyCorrection();
    initialiseZeroPotential();
    initialiseDataonRadialMesh();
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::initialiseNonLocalContribution(
    const std::vector<std::vector<double>> &atomLocations,
    const std::vector<int> &                imageIds,
    const std::vector<std::vector<double>> &periodicCoords,
    const std::vector<double> &             kPointWeights,
    const std::vector<double> &             kPointCoordinates,
    const bool                              updateNonlocalSparsity,
    const unsigned int                      dofHanderId)
  {
    std::vector<unsigned int> atomicNumbers;
    std::vector<double>       atomCoords;
    d_kpointWeights = kPointWeights;

    for (int iAtom = 0; iAtom < atomLocations.size(); iAtom++)
      {
        atomicNumbers.push_back(atomLocations[iAtom][0]);
        for (int dim = 2; dim < 5; dim++)
          atomCoords.push_back(atomLocations[iAtom][dim]);
      }
    createAtomTypesList(atomLocations);

    d_atomicProjectorFnsContainer->initaliseCoordinates(atomCoords,
                                                        periodicCoords,
                                                        imageIds);
    d_atomicShapeFnsContainer->initaliseCoordinates(atomCoords,
                                                    periodicCoords,
                                                    imageIds);

    if (updateNonlocalSparsity)
      {
        d_HamiltonianCouplingMatrixEntriesUpdated = false;
        d_overlapCouplingMatrixEntriesUpdated     = false;
        d_inverseCouplingMatrixEntriesUpdated     = false;
        MPI_Barrier(d_mpiCommParent);
        double InitTime = MPI_Wtime();
        d_atomicProjectorFnsContainer->computeSparseStructure(
          d_BasisOperatorHostPtr, d_sparsityPatternQuadratureId, 1E-8, 0);
        d_atomicShapeFnsContainer->computeSparseStructure(
          d_BasisOperatorHostPtr, d_sparsityPatternQuadratureId, 1E-8, 0);
        d_atomicShapeFnsContainer->computeFEEvaluationMaps(
          d_BasisOperatorElectroHostPtr,
          d_compensationChargeQuadratureIdElectro,
          dofHanderId);
        MPI_Barrier(d_mpiCommParent);
        double TotalTime = MPI_Wtime() - InitTime;
        if (d_verbosity >= 2)
          pcout
            << "pawclass: Time taken for computeSparseStructureNonLocalProjectors: "
            << TotalTime << std::endl;
      }
    MPI_Barrier(d_mpiCommParent);
    double InitTimeTotal = MPI_Wtime();
    d_nonLocalOperator->intitialisePartitionerKPointsAndComputeCMatrixEntries(
      updateNonlocalSparsity,
      kPointWeights,
      kPointCoordinates,
      d_BasisOperatorHostPtr,
      d_nlpspQuadratureId);
    pcout << "-----Compensation Charge---" << std::endl;
    computeCompensationChargeCoeff();
    computeCompensationChargeL0();
    computeNonlocalPseudoPotentialConstants(
      CouplingType::inversePawOverlapEntries);

    MPI_Barrier(d_mpiCommParent);
    double TotalTime = MPI_Wtime() - InitTimeTotal;
    if (d_verbosity >= 2)
      pcout << "pawclass: Time taken for non local psp init: " << TotalTime
            << std::endl;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::initialiseNonLocalContribution(
    const std::vector<std::vector<double>> &        atomLocations,
    const std::vector<int> &                        imageIds,
    const std::vector<std::vector<double>> &        periodicCoords,
    const std::vector<double> &                     kPointWeights,
    const std::vector<double> &                     kPointCoordinates,
    const bool                                      updateNonlocalSparsity,
    const std::map<unsigned int, std::vector<int>> &sparsityPattern,
    const std::vector<std::vector<dealii::CellId>>
      &elementIdsInAtomCompactSupport,
    const std::vector<std::vector<unsigned int>>
      &                              elementIndexesInAtomCompactSupport,
    const std::vector<unsigned int> &atomIdsInCurrentProcess,
    unsigned int                     numberElements)
  {
    std::vector<unsigned int> atomicNumbers;
    std::vector<double>       atomCoords;


    for (int iAtom = 0; iAtom < atomLocations.size(); iAtom++)
      {
        atomicNumbers.push_back(atomLocations[iAtom][0]);
        for (int dim = 2; dim < 5; dim++)
          atomCoords.push_back(atomLocations[iAtom][dim]);
      }


    d_atomicProjectorFnsContainer->initaliseCoordinates(atomCoords,
                                                        periodicCoords,
                                                        imageIds);



    if (updateNonlocalSparsity)
      {
        d_HamiltonianCouplingMatrixEntriesUpdated = false;
        d_overlapCouplingMatrixEntriesUpdated     = false;
        d_inverseCouplingMatrixEntriesUpdated     = false;
        MPI_Barrier(d_mpiCommParent);
        double InitTime = MPI_Wtime();
        d_atomicProjectorFnsContainer->getDataForSparseStructure(
          sparsityPattern,
          elementIdsInAtomCompactSupport,
          elementIndexesInAtomCompactSupport,
          atomIdsInCurrentProcess,
          numberElements);

        MPI_Barrier(d_mpiCommParent);
        double TotalTime = MPI_Wtime() - InitTime;
        if (d_verbosity >= 2)
          pcout
            << "pawclass: Time taken for computeSparseStructureNonLocalProjectors: "
            << TotalTime << std::endl;
      }
    MPI_Barrier(d_mpiCommParent);
    double InitTimeTotal = MPI_Wtime();
    d_nonLocalOperator->intitialisePartitionerKPointsAndComputeCMatrixEntries(
      updateNonlocalSparsity,
      kPointWeights,
      kPointCoordinates,
      d_BasisOperatorHostPtr,
      d_nlpspQuadratureId);
    MPI_Barrier(d_mpiCommParent);
    double TotalTime = MPI_Wtime() - InitTimeTotal;
    if (d_verbosity >= 2)
      pcout << "pawclass: Time taken for non local psp init: " << TotalTime
            << std::endl;
  }



  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeNonlocalPseudoPotentialConstants(
    CouplingType couplingtype,
    unsigned int s)
  {
    if (couplingtype == CouplingType::pawOverlapEntries)
      {
        for (std::set<unsigned int>::iterator it = d_atomTypes.begin();
             it != d_atomTypes.end();
             ++it)
          {
            const unsigned int  Znum           = *it;
            std::vector<double> multipoleTable = d_multipole[Znum];
            const std::map<std::pair<unsigned int, unsigned int>,
                           std::shared_ptr<AtomCenteredSphericalFunctionBase>>
              sphericalFunction =
                d_atomicProjectorFnsContainer->getSphericalFunctions();
            unsigned int numberOfRadialProjectors =
              d_atomicProjectorFnsContainer
                ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);
            unsigned int numTotalProjectors =
              d_atomicProjectorFnsContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
            std::vector<double> fullMultipoleTable(numTotalProjectors *
                                                     numTotalProjectors,
                                                   0.0);
            int                 projectorIndex_i = 0;
            for (unsigned int alpha_i = 0; alpha_i < numberOfRadialProjectors;
                 alpha_i++)
              {
                std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn_i =
                  sphericalFunction.find(std::make_pair(Znum, alpha_i))->second;
                int lQuantumNo_i = sphFn_i->getQuantumNumberl();

                for (int mQuantumNo_i = -lQuantumNo_i;
                     mQuantumNo_i <= lQuantumNo_i;
                     mQuantumNo_i++)
                  {
                    int projectorIndex_j = 0;
                    for (unsigned int alpha_j = 0;
                         alpha_j < numberOfRadialProjectors;
                         alpha_j++)
                      {
                        std::shared_ptr<AtomCenteredSphericalFunctionBase>
                          sphFn_j = sphericalFunction
                                      .find(std::make_pair(Znum, alpha_j))
                                      ->second;
                        int lQuantumNo_j = sphFn_j->getQuantumNumberl();
                        for (int mQuantumNo_j = -lQuantumNo_j;
                             mQuantumNo_j <= lQuantumNo_j;
                             mQuantumNo_j++)
                          {
                            fullMultipoleTable[projectorIndex_i *
                                                 numTotalProjectors +
                                               projectorIndex_j] =
                              sqrt(4 * M_PI) *
                              multipoleTable[alpha_i *
                                               numberOfRadialProjectors +
                                             alpha_j] *
                              gaunt(lQuantumNo_i,
                                    lQuantumNo_j,
                                    0,
                                    mQuantumNo_i,
                                    mQuantumNo_j,
                                    0);

                            projectorIndex_j++;
                          } // mQuantumeNo_j

                      } // alpha_j


                    projectorIndex_i++;
                  } // mQuantumNo_i
              }     // alpha_i

            d_atomicNonLocalPseudoPotentialConstants
              [CouplingType::pawOverlapEntries][Znum] = fullMultipoleTable;
          } //*it
      }
    else if (couplingtype == CouplingType::inversePawOverlapEntries)
      {
        const unsigned int numberNodesPerElement =
          d_BasisOperatorHostPtr->nDofsPerCell();

        const unsigned int numberAtomsOfInterest =
          d_atomicProjectorFnsContainer->getNumAtomCentersSize();
        const std::vector<unsigned int> &atomicNumber =
          d_atomicProjectorFnsContainer->getAtomicNumbers();
        const std::vector<unsigned int> atomIdsInCurrentProcess =
          d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
        std::vector<std::vector<ValueType>> Pmatrix(numberAtomsOfInterest);
        std::vector<double>                 PmatrixVector(d_nProjSqTotal, 0.0);
        for (int iAtom = 0; iAtom < numberAtomsOfInterest; iAtom++)
          {
            unsigned int atomId = atomIdsInCurrentProcess[iAtom];
            unsigned int Znum   = atomicNumber[atomId];
            unsigned int numberOfProjectors =
              d_atomicProjectorFnsContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
            Pmatrix[iAtom].clear();
            Pmatrix[iAtom].resize(numberOfProjectors * numberOfProjectors, 0.0);
            std::vector<unsigned int> elementIndexesInAtomCompactSupport =
              d_atomicProjectorFnsContainer
                ->d_elementIndexesInAtomCompactSupport[atomId];
            int numberElementsInAtomCompactSupport =
              elementIndexesInAtomCompactSupport.size();
            for (int kPoint = 0; kPoint < d_kpointWeights.size(); kPoint++)
              {
                for (int iElem = 0; iElem < numberElementsInAtomCompactSupport;
                     iElem++)
                  {
                    unsigned int elementIndex =
                      elementIndexesInAtomCompactSupport[iElem];
                    // convert this to a ValueType* for better access.
                    // IMPORTANT...
                    std::vector<ValueType> CMatrixEntries =
                      d_nonLocalOperator->getCmatrixEntries(kPoint,
                                                            atomId,
                                                            elementIndex);
                    AssertThrow(
                      CMatrixEntries.size() ==
                        numberOfProjectors * numberNodesPerElement,
                      dealii::ExcMessage(
                        "PAW::Initialization No. of  projectors mismatch in CmatrixEntries. Check input data "));
                    // scaling of CMatrixEntries
                    d_BLASWrapperHostPtr->stridedBlockScale(
                      numberOfProjectors,
                      numberNodesPerElement,
                      1.0,
                      d_BasisOperatorHostPtr->inverseMassVectorBasisData()
                          .data() +
                        elementIndex * numberNodesPerElement,
                      CMatrixEntries.data());
                    // GEMMCall
                    char      transA = 'N';
                    ValueType alpha  = d_kpointWeights[kPoint];
                    ValueType beta   = 1.0;
#ifdef USE_COMPLEX
                    char transB = 'C';
#else
                    char transB = 'T';
#endif
                    d_BLASWrapperHostPtr->xgemm(transA,
                                                transB,
                                                numberOfProjectors,
                                                numberOfProjectors,
                                                numberNodesPerElement,
                                                &alpha,
                                                &CMatrixEntries[0],
                                                numberOfProjectors,
                                                &CMatrixEntries[0],
                                                numberOfProjectors,
                                                &beta,
                                                &Pmatrix[iAtom][0],
                                                numberOfProjectors);


                  } // iElem
              }     // kpoint

            unsigned int startIndex = d_projectorStartIndex[atomId];
            unsigned int alpha      = 0;
            for (int i = 0; i < numberOfProjectors; i++)
              {
                for (int j = 0; j <= i; j++)
                  {
                    PmatrixVector[startIndex + alpha] =
                      std::abs(Pmatrix[iAtom][i * numberOfProjectors + j]);
                    alpha++;
                  }
              }

          } // iAtom
        MPI_Allreduce(MPI_IN_PLACE,
                      &PmatrixVector[0],
                      d_nProjSqTotal,
                      MPI_DOUBLE,
                      MPI_SUM,
                      d_mpiCommParent);
        // Across kpools and across bands all reduce to be called
        for (unsigned int atomId = 0; atomId < atomicNumber.size(); atomId++)
          {
            unsigned int Znum = atomicNumber[atomId];
            unsigned int numberOfProjectors =
              d_atomicProjectorFnsContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
            unsigned int        startIndex = d_projectorStartIndex[atomId];
            std::vector<double> Pij(numberOfProjectors * numberOfProjectors,
                                    0.0);
            std::vector<double> multipoleInverse = d_multipoleInverse[Znum];
            unsigned int        index            = 0;
            pcout << "Delta Matrix: " << std::endl;
            for (int i = 0; i < numberOfProjectors; i++)
              {
                for (int j = 0; j < numberOfProjectors; j++)
                  {
                    Pij[i * numberOfProjectors + j] =
                      PmatrixVector[(startIndex + index)] +
                      multipoleInverse[i * numberOfProjectors + j];
                    pcout << Pij[i * numberOfProjectors + j] << " ";
                    index++;
                  } // j
                pcout << std::endl;
              } // i
            dftfe::linearAlgebraOperations::inverse(&Pij[0],
                                                    numberOfProjectors);
            pcout << "Inverse Delta Matrix: " << std::endl;
            for (int i = 0; i < numberOfProjectors; i++)
              {
                for (int j = 0; j < numberOfProjectors; j++)
                  {
                    pcout << Pij[i * numberOfProjectors + j] << " ";
                  } // j
                pcout << std::endl;
              }
            d_atomicNonLocalPseudoPotentialConstants
              [CouplingType::inversePawOverlapEntries][atomId] = Pij;

          } // atomId
      }
    else if (couplingtype == CouplingType::HamiltonianEntries)
      {
        // initialiseExchangeCorrelationEnergyCorrection(s);
        unsigned int       one     = 1;
        const char         transA  = 'N';
        const char         transB  = 'N';
        const char         transB2 = 'T';
        const double       alpha   = 1.0;
        const double       beta    = 1.0;
        const unsigned int inc     = 1;
        const std::map<std::pair<unsigned int, unsigned int>,
                       std::shared_ptr<AtomCenteredSphericalFunctionBase>>
          shapeFunction = d_atomicShapeFnsContainer->getSphericalFunctions();
        const std::map<std::pair<unsigned int, unsigned int>,
                       std::shared_ptr<AtomCenteredSphericalFunctionBase>>
          projectorFunction =
            d_atomicProjectorFnsContainer->getSphericalFunctions();
        for (std::set<unsigned int>::iterator it = d_atomTypes.begin();
             it != d_atomTypes.end();
             ++it)
          {
            unsigned int              atomType  = *it;
            std::vector<unsigned int> atomLists = d_atomTypesList[atomType];
            unsigned int              Znum      = *it;
            const unsigned int        numberOfProjectorFns =
              d_atomicProjectorFnsContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(*it);
            const unsigned int npjsq =
              numberOfProjectorFns * numberOfProjectorFns;
            std::vector<double> KEContribution =
              d_KineticEnergyCorrectionTerm[Znum];
            std::vector<double> C_ijcontribution  = d_deltaCij[Znum];
            std::vector<double> Cijkl             = d_deltaCijkl[Znum];
            std::vector<double> zeroPotentialAtom = d_zeroPotentialij[Znum];
            std::vector<double> multipoleValue    = d_multipole[Znum];
            std::vector<double> nonLocalHamiltonianVector(npjsq *
                                                            atomLists.size(),
                                                          0.0);
            std::vector<double> deltaColoumbicEnergyDijVector(
              npjsq * atomLists.size(), 0.0);
            std::vector<double> NonLocalComputedContributions(
              npjsq * atomLists.size(), 0.0);
            std::vector<double> LocalContribution(npjsq, 0.0);
            int                 numRadProj =
              d_atomicProjectorFnsContainer
                ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);
            const unsigned int noOfShapeFns =
              d_atomicShapeFnsContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
            const int numRadShapeFns =
              d_atomicShapeFnsContainer
                ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);
            std::vector<double> FullMultipoleTable(npjsq * noOfShapeFns, 0.0);
            int                 projectorIndex_i = 0;
            for (int alpha_i = 0; alpha_i < numRadProj; alpha_i++)
              {
                std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn_i =
                  projectorFunction.find(std::make_pair(Znum, alpha_i))->second;
                int lQuantumNumber_i = sphFn_i->getQuantumNumberl();
                for (int mQuantumNo_i = -lQuantumNumber_i;
                     mQuantumNo_i <= lQuantumNumber_i;
                     mQuantumNo_i++)
                  {
                    int projectorIndex_j = 0;
                    for (int alpha_j = 0; alpha_j < numRadProj; alpha_j++)
                      {
                        std::shared_ptr<AtomCenteredSphericalFunctionBase>
                          sphFn_j = projectorFunction
                                      .find(std::make_pair(Znum, alpha_j))
                                      ->second;
                        int lQuantumNumber_j = sphFn_j->getQuantumNumberl();
                        for (int mQuantumNo_j = -lQuantumNumber_j;
                             mQuantumNo_j <= lQuantumNumber_j;
                             mQuantumNo_j++)
                          {
                            int LshapeFnIndex = 0;
                            for (int L = 0; L < numRadShapeFns; L++)
                              {
                                std::shared_ptr<
                                  AtomCenteredSphericalFunctionBase>
                                  sphFn =
                                    shapeFunction.find(std::make_pair(Znum, L))
                                      ->second;
                                int lQuantumNumber_L =
                                  sphFn->getQuantumNumberl();
                                for (int mQuantumNo_L = -lQuantumNumber_L;
                                     mQuantumNo_L <= lQuantumNumber_L;
                                     mQuantumNo_L++)
                                  {
                                    FullMultipoleTable[projectorIndex_i *
                                                         numberOfProjectorFns *
                                                         noOfShapeFns +
                                                       projectorIndex_j *
                                                         noOfShapeFns +
                                                       LshapeFnIndex] =
                                      multipoleValue[L * numRadProj *
                                                       numRadProj +
                                                     alpha_i * numRadProj +
                                                     alpha_j] *
                                      gaunt(lQuantumNumber_i,
                                            lQuantumNumber_j,
                                            lQuantumNumber_L,
                                            mQuantumNo_i,
                                            mQuantumNo_j,
                                            mQuantumNo_L);

                                    LshapeFnIndex++;
                                  } // mQuantumNo_L
                              }     // L

                            projectorIndex_j++;
                          } // mQuantumNo_j
                      }     // alpha_j

                    projectorIndex_i++;
                  } // mQuantumNo_i

              } // alpha_i

            for (int iProj = 0; iProj < npjsq; iProj++)
              LocalContribution[iProj] = KEContribution[iProj] +
                                         C_ijcontribution[iProj] -
                                         zeroPotentialAtom[iProj];

            for (int i = 0; i < atomLists.size(); i++)
              {
                for (int iAtomList = 0; iAtomList < d_LocallyOwnedAtomId.size();
                     iAtomList++)
                  {
                    unsigned int atomId = d_LocallyOwnedAtomId[iAtomList];

                    if (atomLists[i] == atomId)
                      {
                        // Cijkl Contribution
                        std::vector<double> CijklContribution(npjsq, 0.0);
                        std::vector<double> Dij = D_ij[TypeOfField::In][atomId];
                        dgemm_(&transA,
                               &transB,
                               &inc,
                               &npjsq,
                               &npjsq,
                               &alpha,
                               &Dij[0],
                               &inc,
                               &Cijkl[0],
                               &npjsq,
                               &beta,
                               &CijklContribution[0],
                               &inc);
                        dgemm_(&transA,
                               &transB2,
                               &inc,
                               &npjsq,
                               &npjsq,
                               &alpha,
                               &Dij[0],
                               &inc,
                               &Cijkl[0],
                               &npjsq,
                               &beta,
                               &CijklContribution[0],
                               &inc);
                        std::vector<double> XCcontribution =
                          d_ExchangeCorrelationEnergyCorrectionTerm[atomId];
                        for (int iProj = 0; iProj < npjsq; iProj++)
                          {
                            NonLocalComputedContributions[i * npjsq + iProj] =
                              CijklContribution[iProj] + XCcontribution[iProj];
                            // if (flagEnergy)
                            //   deltaColoumbicEnergyDijVector[i * npjsq +
                            //                                 iProj] +=
                            //     CijklContribution[iProj];
                          }

                      } // Accessing locally owned atom in the current AtomList
                  }     // AtomList in locallyOwnedLoop
                std::vector<double> nonLocalElectrostatics =
                  d_nonLocalHamiltonianElectrostaticValue[atomLists[i]];
                std::vector<double> NonLocalElectorstaticsContributions(npjsq,
                                                                        0.0);
                pcout
                  << "DEBUG: Size of d_nonLocalHamiltonianElectrostaticValue "
                  << nonLocalElectrostatics.size() << std::endl;
                pcout << " DEBUG: Size of NonLocalElectorstaticsContributions "
                      << NonLocalElectorstaticsContributions.size()
                      << std::endl;
                pcout << " DEBUG: No of shapeFns: " << noOfShapeFns
                      << std::endl;
                dgemm_(&transA,
                       &transB,
                       &inc,
                       &npjsq,
                       &noOfShapeFns,
                       &alpha,
                       &nonLocalElectrostatics[0],
                       &inc,
                       &FullMultipoleTable[0],
                       &noOfShapeFns,
                       &beta,
                       &NonLocalElectorstaticsContributions[0],
                       &inc);
                for (int iProj = 0; iProj < npjsq; iProj++)
                  {
                    NonLocalComputedContributions[i * npjsq + iProj] +=
                      NonLocalElectorstaticsContributions[iProj];
                    // deltaColoumbicEnergyDijVector[i * npjsq + iProj] +=
                    //   NonLocalElectorstaticsContributions[iProj];
                  }
              } // i in AtomList Loop
            MPI_Allreduce(MPI_IN_PLACE,
                          &NonLocalComputedContributions[0],
                          npjsq * atomLists.size(),
                          MPI_DOUBLE,
                          MPI_SUM,
                          d_mpiCommParent);
            // if (flagEnergy)
            //   MPI_Allreduce(MPI_IN_PLACE,
            //                 &deltaColoumbicEnergyDijVector[0],
            //                 npjsq * atomLists.size(),
            //                 MPI_DOUBLE,
            //                 MPI_SUM,
            //                 d_mpiCommParent);
            for (int i = 0; i < atomLists.size(); i++)
              {
                std::vector<double> ValueAtom(npjsq, 0.0);
                std::vector<double> deltaColoumbicEnergyDij(npjsq, 0.0);
                unsigned int        iAtom = atomLists[i];
                // pcout << "Entries in nonLocal Hamiltonian for iAtom: " <<
                // iAtom
                //       << std::endl;
                for (int iProj = 0; iProj < npjsq; iProj++)
                  {
                    // nonLocalHamiltonianVector[i * npjsq + iProj] =
                    //   NonLocalComputedContributions[i * npjsq + iProj] +
                    //   LocalContribution[iProj];
                    ValueAtom[iProj] =
                      NonLocalComputedContributions[i * npjsq + iProj] +
                      LocalContribution[iProj];
                    // ValueAtom[iProj] =
                    //   nonLocalHamiltonianVector[i * npjsq + iProj];
                    // if (flagEnergy)
                    //   deltaColoumbicEnergyDij[iProj] =
                    //     deltaColoumbicEnergyDijVector[i * npjsq + iProj];
                    // pcout << iProj << " "
                    //       << nonLocalHamiltonianVector[i * npjsq + iProj] <<
                    //       " "
                    //       << ValueAtom[iProj] << " "
                    //       << NonLocalComputedContributions[i * npjsq + iProj]
                    //       << " "
                    //       << LocalContribution[iProj] << std::endl;
                  }
                d_atomicNonLocalPseudoPotentialConstants
                  [CouplingType::HamiltonianEntries][atomLists[i]] = ValueAtom;
                // d_DeltaColoumbicEnergyDij[iAtom]    =
                // deltaColoumbicEnergyDij;
              } // i in atomList

          } // *it
      }
  }



  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType,
           memorySpace>::createAtomCenteredSphericalFunctionsForProjectors()
  {
    d_atomicProjectorFnsVector.clear();
    std::vector<std::vector<int>> projectorIdDetails;
    std::vector<std::vector<int>> atomicFunctionIdDetails;
    for (std::set<unsigned int>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        char         pseudoAtomDataFile[256];
        unsigned int cumulativeSplineId = 0;
        strcpy(pseudoAtomDataFile,
               (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                "/PseudoAtomDat")
                 .c_str());

        unsigned int  Znum = *it;
        std::ifstream readPseudoDataFileNames(pseudoAtomDataFile);
        unsigned int  numberOfProjectors;
        for (int i = 0; i <= 4; i++)
          {
            std::string temp;
            std::getline(readPseudoDataFileNames, temp);
            // pcout<<temp<<std::endl;
          }
        readPseudoDataFileNames >> numberOfProjectors;
        std::vector<unsigned int> projectorPerOrbital(4, 0);
        readPseudoDataFileNames >> projectorPerOrbital[0];
        readPseudoDataFileNames >> projectorPerOrbital[1];
        readPseudoDataFileNames >> projectorPerOrbital[2];
        readPseudoDataFileNames >> projectorPerOrbital[3];
        unsigned int totalProjectors =
          projectorPerOrbital[0] + projectorPerOrbital[1] +
          projectorPerOrbital[2] + projectorPerOrbital[3];
        pcout << "Znum: " << *it
              << " has no. of radial projectors to be: " << numberOfProjectors
              << std::endl;
        pcout << " Projector l = 0 has: " << projectorPerOrbital[0]
              << " components" << std::endl;
        pcout << " Projector l = 1 has: " << projectorPerOrbital[1]
              << " components" << std::endl;
        pcout << " Projector l = 2 has: " << projectorPerOrbital[2]
              << " components" << std::endl;
        pcout << " Projector l = 3 has: " << projectorPerOrbital[3]
              << " components" << std::endl;
        if (totalProjectors == numberOfProjectors)
          pcout
            << "PAW::Initialization total Radial Projectors in pseudopotential file: "
            << totalProjectors << std::endl;
        else
          AssertThrow(
            false,
            dealii::ExcMessage(
              "PAW::Initialization No. of radial projectors mismatch. Check input data "));
        std::vector<double> radialMesh   = d_radialMesh[*it];
        std::vector<double> jacobianData = d_radialJacobianData[*it];
        unsigned int        meshSize     = radialMesh.size();
        unsigned int        alpha        = 0;
        std::vector<double> radialValuesAE(meshSize * numberOfProjectors);
        std::vector<double> radialValuesPS(meshSize * numberOfProjectors);
        std::vector<double> radialDerivativeAE(meshSize * numberOfProjectors);
        std::vector<double> radialDerivativePS(meshSize * numberOfProjectors);
        for (unsigned int lQuantumNo = 0; lQuantumNo < 4; lQuantumNo++)
          {
            if (projectorPerOrbital[lQuantumNo] == 0)
              continue;
            else
              {
                unsigned int noOfProjectors = projectorPerOrbital[lQuantumNo];
                char         projectorFile[256];
                char         AEpartialWaveFile[256];
                char         PSpartialWaveFile[256];
                strcpy(projectorFile,
                       (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                        "/proj_l" + std::to_string(lQuantumNo) + ".dat")
                         .c_str());
                strcpy(AEpartialWaveFile,
                       (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                        "/allelectron_partial_l" + std::to_string(lQuantumNo) +
                        ".dat")
                         .c_str());
                strcpy(PSpartialWaveFile,
                       (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                        "/smooth_partial_l" + std::to_string(lQuantumNo) +
                        ".dat")
                         .c_str());
                std::vector<std::vector<double>> allElectronPartialData(0);
                dftUtils::readFile(noOfProjectors + 1,
                                   allElectronPartialData,
                                   AEpartialWaveFile);
                std::vector<std::vector<double>> smoothPartialData(0);
                dftUtils::readFile(noOfProjectors + 1,
                                   smoothPartialData,
                                   PSpartialWaveFile);

                for (int j = 1; j < noOfProjectors + 1; j++)
                  {
                    unsigned int        startIndex = alpha * meshSize;
                    std::vector<double> aePhi(meshSize, 0.0);
                    std::vector<double> psPhi(meshSize, 0.0);
                    for (int iRow = 0; iRow < meshSize; iRow++)
                      {
                        aePhi[iRow] = allElectronPartialData[iRow][j];
                        psPhi[iRow] = smoothPartialData[iRow][j];
                      }
                    std::vector<double> functionDerivativesAE =
                      radialDerivativeOfMeshData(radialMesh,
                                                 jacobianData,
                                                 aePhi);
                    std::vector<double> functionDerivativesPS =
                      radialDerivativeOfMeshData(radialMesh,
                                                 jacobianData,
                                                 psPhi);
                    for (int iRow = 0; iRow < meshSize; iRow++)
                      {
                        radialValuesAE[startIndex + iRow] = aePhi[iRow];
                        radialValuesPS[startIndex + iRow] = psPhi[iRow];
                        radialDerivativeAE[startIndex + iRow] =
                          functionDerivativesAE[iRow];
                        radialDerivativePS[startIndex + iRow] =
                          functionDerivativesPS[iRow];
                      }
                    d_atomicProjectorFnsMap[std::make_pair(Znum, alpha)] =
                      std::make_shared<
                        AtomCenteredSphericalFunctionProjectorSpline>(
                        projectorFile,
                        lQuantumNo,
                        0,
                        j,
                        noOfProjectors + 1,
                        1E-12);
                    d_atomicAEPartialWaveFnsMap[std::make_pair(Znum, alpha)] =
                      std::make_shared<
                        AtomCenteredSphericalFunctionProjectorSpline>(
                        AEpartialWaveFile,
                        lQuantumNo,
                        0,
                        j,
                        noOfProjectors + 1,
                        1E-12);
                    d_atomicPSPartialWaveFnsMap[std::make_pair(Znum, alpha)] =
                      std::make_shared<
                        AtomCenteredSphericalFunctionProjectorSpline>(
                        PSpartialWaveFile,
                        lQuantumNo,
                        0,
                        j,
                        noOfProjectors + 1,
                        1E-12);

                    alpha++;
                  }
              }
          }
        d_radialWfcValAE[*it] = radialValuesAE;
        d_radialWfcValPS[*it] = radialValuesPS;
        d_radialWfcDerAE[*it] = radialDerivativeAE;
        d_radialWfcDerPS[*it] = radialDerivativePS;



      } // for loop *it
  }



  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType,
           memorySpace>::createAtomCenteredSphericalFunctionsForZeroPotential()
  {
    d_atomicZeroPotVector.clear();
    d_atomicZeroPotVector.resize(d_nOMPThreads);

    for (std::set<unsigned int>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        unsigned int atomicNumber = *it;
        char         LocalDataFile[256];
        strcpy(LocalDataFile,
               (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                "/zeroPotential.dat")
                 .c_str());
        for (unsigned int i = 0; i < d_nOMPThreads; i++)
          d_atomicZeroPotVector[i][*it] =
            std::make_shared<AtomCenteredSphericalFunctionCoreDensitySpline>(
              LocalDataFile, 1E-12, true);
        std::vector<std::vector<double>> zeroPotentialData(0);
        dftUtils::readFile(2, zeroPotentialData, LocalDataFile);
        unsigned int        numValues = zeroPotentialData.size();
        std::vector<double> zeroPotential(numValues, 0.0);
        for (int iRow = 0; iRow < numValues; iRow++)
          zeroPotential[iRow] = zeroPotentialData[iRow][1];
        d_zeroPotentialRadialValues[*it] = zeroPotential;

      } //*it loop
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::getRadialValenceDensity(unsigned int Znum,
                                                            double       rad)
  {
    unsigned int threadId = omp_get_thread_num();
    double       Value =
      d_atomicValenceDensityVector[threadId][Znum]->getRadialValue(rad) /
      (sqrt(4 * M_PI));

    return (Value);
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::getRadialValenceDensity(
    unsigned int         Znum,
    double               rad,
    std::vector<double> &Val)
  {
    unsigned int threadId = omp_get_thread_num();
    Val.clear();
    Val = d_atomicValenceDensityVector[threadId][Znum]->getDerivativeValue(rad);
    for (int i = 0; i < Val.size(); i++)
      Val[i] /= sqrt(4 * M_PI);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::getRmaxValenceDensity(unsigned int Znum)
  {
    unsigned int threadId = omp_get_thread_num();
    return (d_atomicValenceDensityVector[threadId][Znum]->getRadialCutOff());
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::getRmaxCoreDensity(unsigned int Znum)
  {
    unsigned int threadId = omp_get_thread_num();

    return (d_atomicCoreDensityVector[threadId][Znum]->getRadialCutOff());
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::getRadialCoreDensity(unsigned int Znum,
                                                         double       rad)
  {
    unsigned int threadId = omp_get_thread_num();
    double       Value =
      d_atomicCoreDensityVector[threadId][Znum]->getRadialValue(rad);
    Value /= sqrt(4 * M_PI);
    return (Value);
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::getRadialCoreDensity(
    unsigned int         Znum,
    double               rad,
    std::vector<double> &Val)
  {
    unsigned int threadId = omp_get_thread_num();
    Val.clear();
    Val = d_atomicCoreDensityVector[threadId][Znum]->getDerivativeValue(rad);
    for (int i = 0; i < Val.size(); i++)
      Val[i] /= sqrt(4 * M_PI);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::getRadialZeroPotential(unsigned int Znum,
                                                           double       rad)
  {
    unsigned int threadId = omp_get_thread_num();
    double Value = d_atomicZeroPotVector[threadId][Znum]->getRadialValue(rad);
    Value /= sqrt(4 * M_PI);
    return (Value);
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::getRmaxZeroPotential(unsigned int Znum)
  {
    return (d_atomicZeroPotVector[0][Znum]->getRadialCutOff());
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  bool
  pawClass<ValueType, memorySpace>::coreNuclearDensityPresent(unsigned int Znum)
  {
    return (d_atomTypeCoreFlagMap[Znum]);
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::setImageCoordinates(
    const std::vector<std::vector<double>> &atomLocations,
    const std::vector<int> &                imageIds,
    const std::vector<std::vector<double>> &periodicCoords,
    std::vector<unsigned int> &             imageIdsTemp,
    std::vector<double> &                   imageCoordsTemp)
  {
    imageIdsTemp.clear();
    imageCoordsTemp.clear();
    imageCoordsTemp.resize(imageIds.size() * 3, 0.0);
    std::vector<unsigned int> imageLoc(int(atomLocations.size()), 0.0);
    for (int jImage = 0; jImage < imageIds.size(); jImage++)
      {
        unsigned int atomId = (imageIds[jImage]);
        imageIdsTemp.push_back(atomId);
        int startLoc = imageLoc[atomId];
        imageCoordsTemp[3 * jImage + 0] =
          periodicCoords[atomId][3 * startLoc + 0];
        imageCoordsTemp[3 * jImage + 1] =
          periodicCoords[atomId][3 * startLoc + 1];
        imageCoordsTemp[3 * jImage + 2] =
          periodicCoords[atomId][3 * startLoc + 2];
        imageLoc[atomId] += 1;
      }
  }


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::shared_ptr<AtomicCenteredNonLocalOperator<ValueType, memorySpace>>
  pawClass<ValueType, memorySpace>::getNonLocalOperator()
  {
    return d_nonLocalOperator;
  }



  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  unsigned int
  pawClass<ValueType, memorySpace>::getTotalNumberOfAtomsInCurrentProcessor()
  {
    return d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess().size();
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  unsigned int
  pawClass<ValueType, memorySpace>::getAtomIdInCurrentProcessor(
    unsigned int iAtom)
  {
    std::vector<unsigned int> atomIdList =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
    return (atomIdList[iAtom]);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  unsigned int
  pawClass<ValueType, memorySpace>::getTotalNumberOfSphericalFunctionsForAtomId(
    unsigned int atomId)
  {
    std::vector<unsigned int> atomicNumbers =
      d_atomicProjectorFnsContainer->getAtomicNumbers();
    return (
      d_atomicProjectorFnsContainer->getTotalNumberOfSphericalFunctionsPerAtom(
        atomicNumbers[atomId]));
  }


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::initialiseColoumbicEnergyCorrection()
  {
    pcout << "Initalising Delta C Correction Term" << std::endl;
    for (std::set<unsigned int>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        unsigned int atomicNumber = *it;

        const unsigned int numberOfProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(atomicNumber);
        const unsigned int numberOfRadialProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(atomicNumber);
        const std::map<std::pair<unsigned int, unsigned int>,
                       std::shared_ptr<AtomCenteredSphericalFunctionBase>>
          sphericalFunction =
            d_atomicProjectorFnsContainer->getSphericalFunctions();

        const unsigned int numRadialShapeFunctions =
          d_atomicShapeFnsContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(atomicNumber);
        const unsigned int numShapeFunctions =
          d_atomicShapeFnsContainer->getTotalNumberOfSphericalFunctionsPerAtom(
            atomicNumber);

        unsigned int        RmaxIndex  = d_RmaxAugIndex[atomicNumber];
        std::vector<double> radialMesh = d_radialMesh[atomicNumber];
        const unsigned int  meshSize   = radialMesh.size();
        std::vector<double> rab        = d_radialJacobianData[atomicNumber];

        std::vector<double> multipoleTable = d_multipole[atomicNumber];
        std::vector<double> Delta_Cij(numberOfProjectors * numberOfProjectors,
                                      0.0);
        std::vector<double> Delta_Cijkl(pow(numberOfProjectors, 4), 0.0);
        double              DeltaC        = 0.0;
        double              DeltaCValence = 0.0;
        std::map<int, int>  mapOfRadProjLval;
        std::vector<std::vector<int>> projectorDetailsOfAtom;
        for (int iProj = 0; iProj < numberOfRadialProjectors; iProj++)
          {
            const std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn =
              sphericalFunction.find(std::make_pair(atomicNumber, iProj))
                ->second;
            const int lQuantumNo    = sphFn->getQuantumNumberl();
            mapOfRadProjLval[iProj] = lQuantumNo;
            std::vector<int> temp(3, 0);
            for (int mQuantumNumber = -lQuantumNo; mQuantumNumber <= lQuantumNo;
                 mQuantumNumber++)
              {
                temp[0] = iProj;
                temp[1] = lQuantumNo;
                temp[2] = mQuantumNumber;
                projectorDetailsOfAtom.push_back(temp);
              }
          }

        pcout << "DEBUG check number of entries are matching? "
              << numberOfProjectors << " " << projectorDetailsOfAtom.size()
              << std::endl;

        std::vector<double> psCoreDensity = d_atomCoreDensityPS[*it];
        std::vector<double> aeCoreDensity = d_atomCoreDensityAE[*it];
        std::vector<double> shapeFnRadial = d_atomicShapeFn[*it];
        std::vector<double> NcorePotential, tildeNCorePotential;
        if (d_atomTypeCoreFlagMap[*it])
          {
            oneTermPoissonPotential(&aeCoreDensity[0],
                                    0,
                                    0,
                                    RmaxIndex,
                                    2,
                                    radialMesh,
                                    rab,
                                    NcorePotential);
            oneTermPoissonPotential(&psCoreDensity[0],
                                    0,
                                    0,
                                    RmaxIndex,
                                    2,
                                    radialMesh,
                                    rab,
                                    tildeNCorePotential);
          }
        std::vector<std::vector<double>> gLPotential;
        for (int lShapeFn = 0; lShapeFn < numRadialShapeFunctions; lShapeFn++)
          {
            std::vector<double> tempPotential;
            oneTermPoissonPotential(&shapeFnRadial[lShapeFn * meshSize],
                                    lShapeFn,
                                    0,
                                    RmaxIndex,
                                    2,
                                    radialMesh,
                                    rab,
                                    tempPotential);
            gLPotential.push_back(tempPotential);
          }
        double ShapeFn0PseudoElectronDensityContribution = 0.0,
               AllElectronDensityContribution            = 0.0,
               PseudoElectronDensityContribution         = 0.0,
               ShapeFnContribution[numShapeFunctions];
        if (d_atomTypeCoreFlagMap[*it])
          {
            std::function<double(const unsigned int &)> Integral1 =
              [&](const unsigned int &i) {
                double Value =
                  rab[i] * gLPotential[0][i] * psCoreDensity[i] * radialMesh[i];

                return (Value);
              };
            ShapeFn0PseudoElectronDensityContribution =
              simpsonIntegral(0, RmaxIndex + 1, Integral1);


            std::function<double(const unsigned int &)> Integral2 =
              [&](const unsigned int &i) {
                double Value = rab[i] * tildeNCorePotential[i] *
                               psCoreDensity[i] * radialMesh[i];
                return (Value);
              };
            PseudoElectronDensityContribution =
              simpsonIntegral(0, RmaxIndex + 1, Integral2);

            std::function<double(const unsigned int &)> Integral3 =
              [&](const unsigned int &i) {
                double Value =
                  rab[i] * NcorePotential[i] * aeCoreDensity[i] * radialMesh[i];
                return (Value);
              };
            AllElectronDensityContribution =
              simpsonIntegral(0, RmaxIndex + 1, Integral3);
          }
        int lshapeFn = 0;
        for (int L = 0; L < numRadialShapeFunctions; L++)
          {
            std::function<double(const unsigned int &)> IntegralLoop =
              [&](const unsigned int &i) {
                double Value = rab[i] * gLPotential[L][i] *
                               shapeFnRadial[L * meshSize + i] * radialMesh[i];
                return (Value);
              };
            double ValTempShapeFnContribution =
              simpsonIntegral(0, RmaxIndex + 1, IntegralLoop);

            for (int m = -L; m <= L; m++)
              {
                ShapeFnContribution[lshapeFn] = ValTempShapeFnContribution;
                pcout << "ShapeFn COntribution: " << lshapeFn << " "
                      << ValTempShapeFnContribution << std::endl;
                lshapeFn++;
              }
          }
        std::map<std::pair<int, int>, std::vector<double>> phiIphiJPotentialAE,
          phiIphiJPotentialPS;
        std::vector<double> allElectronPhiIphiJCoreDensityContribution(
          numberOfRadialProjectors * numberOfRadialProjectors, 0.0);
        std::vector<double> pseudoSmoothPhiIphiJCoreDensityContribution(
          numberOfRadialProjectors * numberOfRadialProjectors, 0.0);

        std::vector<double> pseudoSmoothPhiIphiJgLContribution(
          numberOfRadialProjectors * numberOfRadialProjectors *
            numShapeFunctions,
          0.0);
        std::vector<double> integralAllElectronPhiIphiJContribution(
          numberOfRadialProjectors * numberOfRadialProjectors, 0.0);

        std::vector<double> psPhi = d_radialWfcValPS[*it];
        std::vector<double> aePhi = d_radialWfcValAE[*it];

        for (int iProj = 0; iProj < numberOfRadialProjectors; iProj++)
          {
            int l_i = mapOfRadProjLval[iProj];

            for (int jProj = 0; jProj <= iProj; jProj++)
              {
                int       l_j    = mapOfRadProjLval[jProj];
                const int index2 = jProj * numberOfRadialProjectors + iProj;
                const int index1 = iProj * numberOfRadialProjectors + jProj;
                int lmin = std::min(std::abs(l_i - l_j), std::abs(l_i + l_j));
                int lmax = std::max(std::abs(l_i - l_j), std::abs(l_i + l_j));
                for (int lShapeFn = lmin; lShapeFn <= lmax; lShapeFn++)
                  {
                    std::vector<double> tempPotentialAE, tempPotentialPS;
                    twoTermPoissonPotential(&aePhi[iProj * meshSize],
                                            &aePhi[jProj * meshSize],
                                            lShapeFn,
                                            0,
                                            RmaxIndex,
                                            2,
                                            radialMesh,
                                            rab,
                                            tempPotentialAE);
                    twoTermPoissonPotential(&psPhi[iProj * meshSize],
                                            &psPhi[jProj * meshSize],
                                            lShapeFn,
                                            0,
                                            RmaxIndex,
                                            2,
                                            radialMesh,
                                            rab,
                                            tempPotentialPS);
                    phiIphiJPotentialAE[std::make_pair(index1, lShapeFn)] =
                      tempPotentialAE;
                    phiIphiJPotentialAE[std::make_pair(index2, lShapeFn)] =
                      tempPotentialAE;
                    phiIphiJPotentialPS[std::make_pair(index1, lShapeFn)] =
                      tempPotentialPS;
                    phiIphiJPotentialPS[std::make_pair(index2, lShapeFn)] =
                      tempPotentialPS;
                  }
                double              tempAE, tempPS;
                std::vector<double> tempPotentialPS =
                  phiIphiJPotentialPS[std::make_pair(index1, 0)];
                std::vector<double> tempPotentialAE =
                  phiIphiJPotentialAE[std::make_pair(index1, 0)];
                if (d_atomTypeCoreFlagMap[*it])
                  {
                    std::function<double(const unsigned int &)> IntegralLoop1 =
                      [&](const unsigned int &i) {
                        double Value = rab[i] * aeCoreDensity[i] *
                                       tempPotentialAE[i] * radialMesh[i];
                        return (Value);
                      };
                    tempAE = tempPotentialAE.size() == 0 ?
                               0.0 :
                               simpsonIntegral(0, RmaxIndex + 1, IntegralLoop1);

                    std::function<double(const unsigned int &)> IntegralLoop2 =
                      [&](const unsigned int &i) {
                        double Value = rab[i] * psCoreDensity[i] *
                                       tempPotentialPS[i] * radialMesh[i];
                        return (Value);
                      };
                    tempPS = tempPotentialPS.size() == 0 ?
                               0.0 :
                               simpsonIntegral(0, RmaxIndex + 1, IntegralLoop2);


                    allElectronPhiIphiJCoreDensityContribution[index1] = tempAE;
                    allElectronPhiIphiJCoreDensityContribution[index2] = tempAE;
                    pseudoSmoothPhiIphiJCoreDensityContribution[index1] =
                      tempPS;
                    pseudoSmoothPhiIphiJCoreDensityContribution[index2] =
                      tempPS;
                  } // if core present

                integralAllElectronPhiIphiJContribution[index1] =
                  integralOfProjectorsInAugmentationSphere(
                    &aePhi[iProj * meshSize],
                    &aePhi[jProj * meshSize],
                    radialMesh,
                    rab,
                    0,
                    RmaxIndex + 1);
                integralAllElectronPhiIphiJContribution[index2] =
                  integralAllElectronPhiIphiJContribution[index1];
                int shapeFnIndex = 0;
                for (int L = 0; L < numRadialShapeFunctions; L++)
                  {
                    std::function<double(const unsigned int &)> IntegralLoop =
                      [&](const unsigned int &i) {
                        double Value = rab[i] * gLPotential[L][i] *
                                       psPhi[iProj * meshSize + i] *
                                       psPhi[jProj * meshSize + i] *
                                       radialMesh[i];
                        return (Value);
                      };
                    double ValTempShapeFnContribution =
                      simpsonIntegral(0, RmaxIndex + 1, IntegralLoop);
                    for (int m = -L; m <= L; m++)
                      {
                        pseudoSmoothPhiIphiJgLContribution
                          [iProj * numberOfRadialProjectors *
                             numShapeFunctions +
                           jProj * numShapeFunctions + shapeFnIndex] =
                            ValTempShapeFnContribution;
                        pseudoSmoothPhiIphiJgLContribution
                          [jProj * numberOfRadialProjectors *
                             numShapeFunctions +
                           iProj * numShapeFunctions + shapeFnIndex] =
                            ValTempShapeFnContribution;

                        pcout << shapeFnIndex << " " << iProj << " " << jProj
                              << " "
                              << "Debug2_0: PhitIPhiJgL: "
                              << ValTempShapeFnContribution << std::endl;
                        shapeFnIndex++;
                      }
                  }


              } // jProj
          }     // iProj
        // Computing Delta C0 Term
        double dL0       = d_DeltaL0coeff[*it];
        double valueTemp = 0.0;

        valueTemp = 0.5 * (AllElectronDensityContribution);
        pcout << " Core-Core contribution: " << valueTemp << std::endl;
        DeltaC += valueTemp;


        valueTemp = -0.5 * (PseudoElectronDensityContribution);
        pcout << " - psedo-pseduo contribution: " << valueTemp << std::endl;
        DeltaC += valueTemp;
        DeltaCValence += valueTemp;

        valueTemp = -0.5 * (dL0 * dL0 * ShapeFnContribution[0]);
        pcout << " -g_L(x)g_L(x) contribution: " << valueTemp << std::endl;
        DeltaC += valueTemp;
        DeltaCValence += valueTemp;

        valueTemp =
          -(d_DeltaL0coeff[*it]) * (ShapeFn0PseudoElectronDensityContribution);
        pcout << " -g_L(x)-pseudo contribution: " << valueTemp << std::endl;
        DeltaC += valueTemp;
        DeltaCValence += valueTemp;

        valueTemp = -sqrt(4 * M_PI) * (*it) *
                    integralOfDensity(
                      &aeCoreDensity[0], radialMesh, rab, 0, RmaxIndex + 1);

        pcout << " integral core/r: " << valueTemp << std::endl;
        DeltaC += valueTemp;

        pcout << "Start of Filling in entries to Delta C_ij matrices"
              << std::endl;

        for (int i = 0; i < numberOfProjectors; i++)
          {
            int l_i           = projectorDetailsOfAtom[i][1];
            int m_i           = projectorDetailsOfAtom[i][2];
            int radProjIndexI = projectorDetailsOfAtom[i][0];

            for (int j = 0; j < numberOfProjectors; j++)
              {
                // pcout << "DEBUG2: " << i << " " << j << " ";
                int    l_j           = projectorDetailsOfAtom[j][1];
                int    m_j           = projectorDetailsOfAtom[j][2];
                int    radProjIndexJ = projectorDetailsOfAtom[j][0];
                double GauntValueij  = gaunt(l_i, l_j, 0, m_i, m_j, 0);
                if (d_atomTypeCoreFlagMap[*it])
                  {
                    Delta_Cij[i * numberOfProjectors + j] +=
                      GauntValueij *
                      (allElectronPhiIphiJCoreDensityContribution
                         [radProjIndexI * numberOfRadialProjectors +
                          radProjIndexJ] -
                       pseudoSmoothPhiIphiJCoreDensityContribution
                         [radProjIndexI * numberOfRadialProjectors +
                          radProjIndexJ]);
                    //               pcout << GauntValueij *
                    //      (allElectronPhiIphiJCoreDensityContribution
                    //         [radProjIndexI * numberOfRadialProjectors +
                    //         radProjIndexJ] -
                    //       pseudoSmoothPhiIphiJCoreDensityContribution
                    //         [radProjIndexI * numberOfRadialProjectors +
                    //         radProjIndexJ])
                    // << " ";
                  }
                if (l_i == l_j && m_i == m_j)
                  {
                    Delta_Cij[i * numberOfProjectors + j] +=
                      -(double(atomicNumber)) *
                      integralAllElectronPhiIphiJContribution
                        [radProjIndexI * numberOfRadialProjectors +
                         radProjIndexJ];
                    //               pcout << -(double(atomicNumber))*
                    //      integralAllElectronPhiIphiJContribution[radProjIndexI
                    //      * numberOfRadialProjectors
                    //      +
                    //                         radProjIndexJ]
                    // << " ";
                  }
                double multipoleValue =
                  multipoleTable[radProjIndexI * numberOfRadialProjectors +
                                 radProjIndexJ];
                Delta_Cij[i * numberOfProjectors + j] -=
                  multipoleValue * GauntValueij *
                  (dL0 * ShapeFnContribution[0]);
                // pcout << -multipoleValue *
                //            (d_DeltaL0coeff[*it] * ShapeFnContribution[0]) *
                //            GauntValueij
                //       << " ";
                if (d_atomTypeCoreFlagMap[*it])
                  {
                    Delta_Cij[i * numberOfProjectors + j] -=
                      multipoleValue *
                      ShapeFn0PseudoElectronDensityContribution * GauntValueij;
                  }
                Delta_Cij[i * numberOfProjectors + j] -=
                  GauntValueij * dL0 *
                  pseudoSmoothPhiIphiJgLContribution
                    [radProjIndexI * numShapeFunctions *
                       numberOfRadialProjectors +
                     radProjIndexJ * numShapeFunctions + 0];
              } // j
          }     // i
        pcout << "Start of Filling in entries to Delta C_ijkl matrices"
              << std::endl;
        for (int iProj = 0; iProj < numberOfProjectors; iProj++)
          {
            int l_i           = projectorDetailsOfAtom[iProj][1];
            int m_i           = projectorDetailsOfAtom[iProj][2];
            int radProjIndexI = projectorDetailsOfAtom[iProj][0];

            for (int jProj = 0; jProj < numberOfProjectors; jProj++)
              {
                int       l_j           = projectorDetailsOfAtom[jProj][1];
                int       m_j           = projectorDetailsOfAtom[jProj][2];
                int       radProjIndexJ = projectorDetailsOfAtom[jProj][0];
                const int index_ij =
                  numberOfRadialProjectors * radProjIndexI + radProjIndexJ;

                for (int kProj = 0; kProj < numberOfProjectors; kProj++)
                  {
                    int l_k           = projectorDetailsOfAtom[kProj][1];
                    int m_k           = projectorDetailsOfAtom[kProj][2];
                    int radProjIndexK = projectorDetailsOfAtom[kProj][0];
                    for (int lProj = 0; lProj < numberOfProjectors; lProj++)
                      {
                        int l_l           = projectorDetailsOfAtom[lProj][1];
                        int m_l           = projectorDetailsOfAtom[lProj][2];
                        int radProjIndexL = projectorDetailsOfAtom[lProj][0];
                        const int index   = pow(numberOfProjectors, 3) * iProj +
                                          pow(numberOfProjectors, 2) * jProj +
                                          pow(numberOfProjectors, 1) * kProj +
                                          lProj;
                        const int index_ijkl =
                          pow(numberOfRadialProjectors, 3) * radProjIndexI +
                          pow(numberOfRadialProjectors, 2) * radProjIndexJ +
                          pow(numberOfRadialProjectors, 1) * radProjIndexK +
                          radProjIndexL;

                        double radValijkl = 0.0;
                        int    lmin =
                          std::min(std::abs(l_i - l_j), std::abs(l_k - l_l));
                        int lmax = std::max((l_i + l_j), (l_k + l_l));
                        pcout << "Lmin and Lmax: " << lmin << " " << lmax
                              << std::endl;

                        for (int lprojShapeFn = lmin; lprojShapeFn <= lmax;
                             lprojShapeFn++)
                          {
                            bool flag = false;
                            for (int mprojShapeFn = -lprojShapeFn;
                                 mprojShapeFn <= lprojShapeFn;
                                 mprojShapeFn++)
                              {
                                double CG1, CG2;
                                CG1 = gaunt(l_i,
                                            l_j,
                                            lprojShapeFn,
                                            m_i,
                                            m_j,
                                            mprojShapeFn);
                                CG2 = gaunt(l_k,
                                            l_l,
                                            lprojShapeFn,
                                            m_k,
                                            m_l,
                                            mprojShapeFn);
                                if (std::fabs(CG1 * CG2) > 1E-10)
                                  flag = true;
                              } // mproj
                            if (flag)
                              {
                                if (phiIphiJPotentialAE
                                      .find(
                                        std::make_pair(index_ij, lprojShapeFn))
                                      ->second.size() > 0)
                                  {
                                    std::vector<double> potentialPhiIPhiJ =
                                      phiIphiJPotentialAE
                                        .find(std::make_pair(index_ij,
                                                             lprojShapeFn))
                                        ->second;
                                    std::vector<double>
                                      potentialTildePhiITildePhiJ =
                                        phiIphiJPotentialPS
                                          .find(std::make_pair(index_ij,
                                                               lprojShapeFn))
                                          ->second;


                                    std::function<double(const unsigned int &)>
                                      IntegralContribution =
                                        [&](const unsigned int &i) {
                                          double Value1 =
                                            rab[i] * potentialPhiIPhiJ[i] *
                                            aePhi[radProjIndexK * meshSize +
                                                  i] *
                                            aePhi[radProjIndexL * meshSize +
                                                  i] *
                                            radialMesh[i];
                                          double Value2 =
                                            rab[i] *
                                            potentialTildePhiITildePhiJ[i] *
                                            psPhi[radProjIndexK * meshSize +
                                                  i] *
                                            psPhi[radProjIndexL * meshSize +
                                                  i] *
                                            radialMesh[i];
                                          return (Value1 - Value2);
                                        };


                                    double TotalValue =
                                      simpsonIntegral(0,
                                                      RmaxIndex + 1,
                                                      IntegralContribution);
                                    double TotalContribution = 0.0;

                                    for (int mprojShapeFn = -lprojShapeFn;
                                         mprojShapeFn <= lprojShapeFn;
                                         mprojShapeFn++)
                                      {
                                        double CG1, CG2;
                                        CG1 = gaunt(l_i,
                                                    l_j,
                                                    lprojShapeFn,
                                                    m_i,
                                                    m_j,
                                                    mprojShapeFn);
                                        CG2 = gaunt(l_k,
                                                    l_l,
                                                    lprojShapeFn,
                                                    m_k,
                                                    m_l,
                                                    mprojShapeFn);
                                        if (std::fabs(CG1 * CG2) > 1E-10)
                                          TotalContribution +=
                                            (TotalValue)*CG1 * CG2;

                                      } // mproj
                                    Delta_Cijkl[index] +=
                                      0.5 * TotalContribution;
                                    pcout
                                      << "DEBUG: Value Check: " << (TotalValue)
                                      << " " << iProj << " " << jProj << " "
                                      << kProj << " " << lProj << " "
                                      << lprojShapeFn << std::endl;
                                  }

                                else
                                  {
                                    pcout
                                      << "Mising Entries for lproj: " << lProj
                                      << " " << index_ij << std::endl;
                                  }
                              }
                          }
                        double val           = 0;
                        int    lShapeFnIndex = 0;
                        for (int L = 0; L < numRadialShapeFunctions; L++)
                          {
                            int lQuantumNo = L;
                            for (int mQuantumNo = -lQuantumNo;
                                 mQuantumNo <= lQuantumNo;
                                 mQuantumNo++)
                              {
                                double multipoleValue1 =
                                  multipoleTable[lQuantumNo *
                                                   numberOfRadialProjectors *
                                                   numberOfRadialProjectors +
                                                 radProjIndexI *
                                                   numberOfRadialProjectors +
                                                 radProjIndexJ];
                                double multipoleValue2 =
                                  multipoleTable[lQuantumNo *
                                                   numberOfRadialProjectors *
                                                   numberOfRadialProjectors +
                                                 radProjIndexK *
                                                   numberOfRadialProjectors +
                                                 radProjIndexL];
                                double GauntValueijL = gaunt(
                                  l_i, l_j, lQuantumNo, m_i, m_j, mQuantumNo);
                                double GauntValueklL = gaunt(
                                  l_k, l_l, lQuantumNo, m_k, m_l, mQuantumNo);
                                val += multipoleValue2 * GauntValueklL *
                                       pseudoSmoothPhiIphiJgLContribution
                                         [radProjIndexI * numShapeFunctions *
                                            numberOfRadialProjectors +
                                          radProjIndexJ * numShapeFunctions +
                                          lShapeFnIndex] *
                                       GauntValueijL;

                                val += 0.5 * multipoleValue1 * GauntValueijL *
                                       multipoleValue2 * GauntValueklL *
                                       ShapeFnContribution[lShapeFnIndex];

                                lShapeFnIndex++;
                              } // mQuantumNo
                          }     // L
                        Delta_Cijkl[index] -= val;

                      } // lProj
                  }     // kProj


              } // j
          }     // i

        // Copying the data to class
        d_deltaCij[*it]      = Delta_Cij;
        d_deltaCijkl[*it]    = Delta_Cijkl;
        d_deltaC[*it]        = DeltaC;
        d_deltaValenceC[*it] = DeltaCValence;

        // printing the entries
        pcout << "** Delta C0 Term: " << DeltaC << std::endl;
        pcout << "** Delta C0 Valence Term: " << DeltaCValence << std::endl;
        pcout << "Delta Cij Term: " << std::endl;
        for (int i = 0; i < numberOfProjectors; i++)
          {
            for (int j = 0; j < numberOfProjectors; j++)
              pcout << Delta_Cij[i * numberOfProjectors + j] << " ";
            pcout << std::endl;
          }
        pcout << "Delta sum_klCijkl: " << std::endl;
        for (int i = 0; i < numberOfProjectors; i++)
          {
            for (int j = 0; j < numberOfProjectors; j++)
              {
                double Value = 0.0;
                for (int k = 0; k < numberOfProjectors; k++)
                  {
                    for (int l = 0; l < numberOfProjectors; l++)
                      {
                        Value += Delta_Cijkl[i * pow(numberOfProjectors, 3) +
                                             j * pow(numberOfProjectors, 2) +
                                             k * numberOfProjectors + l];
                      }
                  }
                pcout << Value << " ";
              }
            pcout << std::endl;
          }

      } //*it
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::initialiseZeroPotential()
  {
    for (std::set<unsigned int>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        unsigned int       atomicNumber = *it;
        const unsigned int numberOfProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(atomicNumber);
        const unsigned int numberOfRadialProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(atomicNumber);
        const std::map<std::pair<unsigned int, unsigned int>,
                       std::shared_ptr<AtomCenteredSphericalFunctionBase>>
          sphericalFunction =
            d_atomicProjectorFnsContainer->getSphericalFunctions();

        unsigned int        RmaxIndex  = d_RmaxAugIndex[atomicNumber];
        std::vector<double> radialMesh = d_radialMesh[atomicNumber];
        std::vector<double> rab        = d_radialJacobianData[atomicNumber];

        std::vector<double> tempZeroPotentialIJ(numberOfProjectors *
                                                  numberOfProjectors,
                                                0.0);
        std::vector<double> radialIntegralData(numberOfRadialProjectors *
                                                 numberOfRadialProjectors,
                                               0.0);
        std::vector<double> radialPSWaveFunctionsData =
          d_radialWfcValPS[atomicNumber];
        std::vector<double> zeroPotentialData =
          d_zeroPotentialRadialValues[atomicNumber];
        for (int i = 0; i < numberOfRadialProjectors; i++)
          {
            for (int j = 0; j <= i; j++)
              {
                radialIntegralData[i * numberOfRadialProjectors + j] =
                  threeTermIntegrationOverAugmentationSphere(
                    &radialPSWaveFunctionsData[i * radialMesh.size()],
                    &radialPSWaveFunctionsData[j * radialMesh.size()],
                    &zeroPotentialData[0],
                    radialMesh,
                    rab,
                    0,
                    RmaxIndex + 1);
                radialIntegralData[j * numberOfRadialProjectors + i] =
                  radialIntegralData[i * numberOfRadialProjectors + j];
              } // j
          }     // i

        int projIndexI = 0;
        pcout << "Delta Zero potential " << std::endl;
        for (int iProj = 0; iProj < numberOfRadialProjectors; iProj++)
          {
            std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn_i =
              sphericalFunction.find(std::make_pair(atomicNumber, iProj))
                ->second;
            const int lQuantumNo_i = sphFn_i->getQuantumNumberl();
            for (int mQuantumNumber_i = -lQuantumNo_i;
                 mQuantumNumber_i <= lQuantumNo_i;
                 mQuantumNumber_i++)
              {
                int projIndexJ = 0;
                for (int jProj = 0; jProj < numberOfRadialProjectors; jProj++)
                  {
                    std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn_j =
                      sphericalFunction
                        .find(std::make_pair(atomicNumber, jProj))
                        ->second;
                    const int lQuantumNo_j = sphFn_j->getQuantumNumberl();
                    for (int mQuantumNumber_j = -lQuantumNo_j;
                         mQuantumNumber_j <= lQuantumNo_j;
                         mQuantumNumber_j++)
                      {
                        tempZeroPotentialIJ[projIndexI * numberOfProjectors +
                                            projIndexJ] =
                          gaunt(lQuantumNo_i,
                                lQuantumNo_j,
                                0,
                                mQuantumNumber_i,
                                mQuantumNumber_j,
                                0) *
                          radialIntegralData[iProj * numberOfRadialProjectors +
                                             jProj];
                        pcout << tempZeroPotentialIJ[projIndexI *
                                                       numberOfProjectors +
                                                     projIndexJ]
                              << " ";
                        projIndexJ++;
                      } // mQuantumNumber_j

                  } // jProj
                pcout << std::endl;
                projIndexI++;
              } // mQuantumNumber_i



          } // iProj

        d_zeroPotentialij[*it] = tempZeroPotentialIJ;
      } //*it
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::
    initialiseExchangeCorrelationEnergyCorrection(unsigned int s)
  {
    const bool isGGA =
      d_excManagerPtr->getDensityBasedFamilyType() == densityFamilyType::GGA;

    std::vector<double>              quad_weights;
    std::vector<std::vector<double>> quad_points;
    getSphericalQuadratureRule(quad_weights, quad_points);
    double DijCreation, LDAContribution, Part0Contribution, PartAContribution,
      PartBContribution, PartCContribution, VxcCompute;
    double       TimerStart, TimerEnd;
    int          numberofSphericalValues = quad_weights.size();
    unsigned int atomId                  = 0;
    const std::map<std::pair<unsigned int, unsigned int>,
                   std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      sphericalFunction =
        d_atomicProjectorFnsContainer->getSphericalFunctions();
    pcout << "Initializing XC contribution: " << std::endl;
    std::vector<unsigned int> atomicNumbers =
      d_atomicProjectorFnsContainer->getAtomicNumbers();
    if (d_LocallyOwnedAtomId.size() > 0)
      {
        for (int iAtomList = 0; iAtomList < d_LocallyOwnedAtomId.size();
             iAtomList++)
          {
            atomId                         = d_LocallyOwnedAtomId[iAtomList];
            std::vector<double> Dij        = D_ij[TypeOfField::In][atomId];
            unsigned int        Znum       = atomicNumbers[atomId];
            std::vector<double> RadialMesh = d_radialMesh[Znum];
            unsigned int        RmaxIndex  = d_RmaxAugIndex[Znum];

            std::vector<double> rab            = d_radialJacobianData[Znum];
            unsigned int        RadialMeshSize = RadialMesh.size();
            const unsigned int  numberofValues =
              std::min(RmaxIndex + 5, RadialMeshSize);
            // pcout << "Number of Values: " << numberofValues << std::endl;


            const unsigned int numberOfProjectors =
              d_atomicProjectorFnsContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
            const unsigned int numberOfRadialProjectors =
              d_atomicProjectorFnsContainer
                ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);
            std::vector<double> Delta_Excij(numberOfProjectors *
                                              numberOfProjectors,
                                            0.0);
            std::vector<double> Delta_ExcijDensity(numberOfProjectors *
                                                     numberOfProjectors,
                                                   0.0);
            std::vector<double> Delta_ExcijSigma(numberOfProjectors *
                                                   numberOfProjectors,
                                                 0.0);
            const unsigned int  numberOfProjectorsSq =
              numberOfProjectors * numberOfProjectors;
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
                    std::vector<double> exchangePotentialValAE(numberofValues);
                    std::vector<double> corrPotentialValAE(numberofValues);
                    std::vector<double> exchangePotentialValPS(numberofValues);
                    std::vector<double> corrPotentialValPS(numberofValues);
                    std::map<rhoDataAttributes, const std::vector<double> *>
                      rhoDataAE, rhoDataPS;

                    std::map<VeffOutputDataAttributes, std::vector<double> *>
                      outputDerExchangeEnergyAE, outputDerExchangeEnergyPS;
                    std::map<VeffOutputDataAttributes, std::vector<double> *>
                      outputDerCorrEnergyAE, outputDerCorrEnergyPS;

                    rhoDataAE[rhoDataAttributes::values] =
                      &atomDensityAllelectron;
                    rhoDataPS[rhoDataAttributes::values] = &atomDensitySmooth;

                    outputDerExchangeEnergyAE
                      [VeffOutputDataAttributes::derEnergyWithDensity] =
                        &exchangePotentialValAE;

                    outputDerCorrEnergyAE
                      [VeffOutputDataAttributes::derEnergyWithDensity] =
                        &corrPotentialValAE;
                    outputDerExchangeEnergyPS
                      [VeffOutputDataAttributes::derEnergyWithDensity] =
                        &exchangePotentialValPS;

                    outputDerCorrEnergyPS
                      [VeffOutputDataAttributes::derEnergyWithDensity] =
                        &corrPotentialValPS;
                    d_excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
                      numberofValues,
                      rhoDataAE,
                      outputDerExchangeEnergyAE,
                      outputDerCorrEnergyAE);
                    d_excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
                      numberofValues,
                      rhoDataPS,
                      outputDerExchangeEnergyPS,
                      outputDerCorrEnergyPS);

                    // pcout<<"Line 2733"<<std::endl;

                    for (int i = 0; i < numberOfProjectors; i++)
                      {
                        // Proj J
                        for (int j = 0; j <= i; j++)
                          {
                            // Radial Integration
                            std::function<double(const unsigned int &)>
                              Integral = [&](const unsigned int &rpoint) {
                                unsigned int index =
                                  rpoint * numberOfProjectorsSq +
                                  i * numberOfProjectors + j;
                                double Val1 = productOfAEpartialWfc[index] *
                                              (exchangePotentialValAE[rpoint] +
                                               corrPotentialValAE[rpoint]);
                                double Val2 = productOfPSpartialWfc[index] *
                                              (exchangePotentialValPS[rpoint] +
                                               corrPotentialValPS[rpoint]);
                                double Value = rab[rpoint] * (Val1 - Val2) *
                                               pow(RadialMesh[rpoint], 2);
                                // pcout<<i<<" "<<Value<<std::endl;
                                return (std::fabs(RadialMesh[rpoint]) > 1E-8 ?
                                          Value :
                                          0.0);
                              };

                            double RadialIntegral =
                              simpsonIntegral(0, RmaxIndex + 1, Integral);
                            Delta_Excij[i * numberOfProjectors + j] +=
                              RadialIntegral * quadwt * 4.0 * M_PI *
                              SphericalHarmonics[i * numberOfProjectors + j];
                          } // Proj J
                      }     // Proj I



                    exchangePotentialValAE.resize(0);
                    exchangePotentialValPS.resize(0);
                    corrPotentialValAE.resize(0);
                    corrPotentialValPS.resize(0);
                    atomDensityAllelectron.resize(0);
                    atomDensitySmooth.resize(0);

                    exchangePotentialValAE.clear();
                    exchangePotentialValPS.clear();
                    corrPotentialValAE.clear();
                    corrPotentialValPS.clear();
                    atomDensityAllelectron.clear();
                    atomDensitySmooth.clear();



                  } // qpoint
              }     // LDA case

            else
              {
                double Yi, Yj;
                pcout << "Starting GGA Delta XC: " << std::endl;
                // double timerLDAContribution, timerGGAContribution;
                // timerLDAContribution                       = 0.0;
                // timerGGAContribution                       = 0.0;
                // double               timerGGA0Contribution = 0.0;
                // double               timerGGAAContribution = 0.0;
                // double               timerGGABContribution = 0.0;
                // double               timerGGACContribution = 0.0;
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
                    // timerGGAAContribution += (MPI_Wtime() -
                    // TimerGGAAStart);


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
                          tempAETrialB[iRad] * pow(RadialMesh[iRad], 2);
                        const double scalePS =
                          tempPSTrialB[iRad] * pow(RadialMesh[iRad], 2);
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
                    // timerGGABContribution += (MPI_Wtime() -
                    // TimerGGABStart);
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
                          tempAETrialC[iRad] * pow(RadialMesh[iRad], 2);
                        const double scalePS =
                          tempPSTrialC[iRad] * pow(RadialMesh[iRad], 2);
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
                    // MPI_Barrier(d_mpiCommParent);
                    // timerGGAContribution += MPI_Wtime() - TimerGGAStart;
                    // timerGGACContribution += MPI_Wtime() - TimerGGACStart;
                    std::vector<double> exchangePotentialValAE(numberofValues);
                    std::vector<double> corrPotentialValAE(numberofValues);
                    std::vector<double> exchangePotentialValPS(numberofValues);
                    std::vector<double> corrPotentialValPS(numberofValues);

                    std::vector<double> exchangePotentialValAEfromSigma(
                      numberofValues);
                    std::vector<double> corrPotentialValAEfromSigma(
                      numberofValues);
                    std::vector<double> exchangePotentialValPSfromSigma(
                      numberofValues);
                    std::vector<double> corrPotentialValPSfromSigma(
                      numberofValues);

                    std::map<rhoDataAttributes, const std::vector<double> *>
                      rhoDataAE, rhoDataPS;

                    std::map<VeffOutputDataAttributes, std::vector<double> *>
                      outputDerExchangeEnergyAE, outputDerExchangeEnergyPS;
                    std::map<VeffOutputDataAttributes, std::vector<double> *>
                      outputDerCorrEnergyAE, outputDerCorrEnergyPS;



                    std::map<VeffOutputDataAttributes, std::vector<double> *>
                      outputDerExchangeEnergyAEfromSigma,
                      outputDerExchangeEnergyPSfromSigma;
                    std::map<VeffOutputDataAttributes, std::vector<double> *>
                      outputDerCorrEnergyAEfromSigma,
                      outputDerCorrEnergyPSfromSigma;

                    rhoDataAE[rhoDataAttributes::values] =
                      &atomDensityAllelectron;
                    rhoDataPS[rhoDataAttributes::values] = &atomDensitySmooth;
                    rhoDataAE[rhoDataAttributes::sigmaGradValue] =
                      &sigmaAllElectron;
                    rhoDataPS[rhoDataAttributes::sigmaGradValue] = &sigmaSmooth;


                    outputDerExchangeEnergyAE
                      [VeffOutputDataAttributes::derEnergyWithDensity] =
                        &exchangePotentialValAE;
                    outputDerCorrEnergyAE
                      [VeffOutputDataAttributes::derEnergyWithDensity] =
                        &corrPotentialValAE;
                    outputDerExchangeEnergyPS
                      [VeffOutputDataAttributes::derEnergyWithDensity] =
                        &exchangePotentialValPS;
                    outputDerCorrEnergyPS
                      [VeffOutputDataAttributes::derEnergyWithDensity] =
                        &corrPotentialValPS;

                    outputDerExchangeEnergyAE[VeffOutputDataAttributes::
                                                derEnergyWithSigmaGradDensity] =
                      &exchangePotentialValAEfromSigma;
                    outputDerCorrEnergyAE[VeffOutputDataAttributes::
                                            derEnergyWithSigmaGradDensity] =
                      &corrPotentialValAEfromSigma;
                    outputDerExchangeEnergyPS[VeffOutputDataAttributes::
                                                derEnergyWithSigmaGradDensity] =
                      &exchangePotentialValPSfromSigma;
                    outputDerCorrEnergyPS[VeffOutputDataAttributes::
                                            derEnergyWithSigmaGradDensity] =
                      &corrPotentialValPSfromSigma;


                    d_excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
                      numberofValues,
                      rhoDataAE,
                      outputDerExchangeEnergyAE,
                      outputDerCorrEnergyAE);
                    d_excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
                      numberofValues,
                      rhoDataPS,
                      outputDerExchangeEnergyPS,
                      outputDerCorrEnergyPS);


                    for (int i = 0; i < numberOfProjectors; i++)
                      {
                        // Proj J
                        for (int j = 0; j <= i; j++)
                          {
                            // Radial Integration
                            std::function<double(const unsigned int &)>
                              IntegralLDA = [&](const unsigned int &rpoint) {
                                unsigned int index =
                                  rpoint * numberOfProjectorsSq +
                                  i * numberOfProjectors + j;
                                double Val1 =
                                  productOfAEpartialWfc[index] *
                                  (exchangePotentialValAE[rpoint] +
                                   corrPotentialValAE[rpoint]) *
                                  SphericalHarmonics[i * numberOfProjectors +
                                                     j];
                                double Val2 =
                                  productOfPSpartialWfc[index] *
                                  (exchangePotentialValPS[rpoint] +
                                   corrPotentialValPS[rpoint]) *
                                  SphericalHarmonics[i * numberOfProjectors +
                                                     j];
                                double Value = rab[rpoint] * (Val1 - Val2) *
                                               pow(RadialMesh[rpoint], 2);
                                // pcout<<i<<" "<<Value<<std::endl;
                                return (std::fabs(RadialMesh[rpoint]) > 1E-8 ?
                                          Value :
                                          0.0);
                              };

                            double RadialIntegralLDA =
                              simpsonIntegral(0, RmaxIndex + 1, IntegralLDA);
                            Delta_ExcijDensity[i * numberOfProjectors + j] +=
                              RadialIntegralLDA * quadwt * 4.0 * M_PI;
                          } // Proj J
                      }     // Proj I
                    for (int i = 0; i < numberOfProjectors; i++)
                      {
                        // Proj J
                        for (int j = 0; j < numberOfProjectors; j++)
                          {
                            // Radial Integration
                            std::function<double(const unsigned int &)>
                              IntegralGGA = [&](const unsigned int &rpoint) {
                                unsigned int index =
                                  rpoint * numberOfProjectorsSq +
                                  i * numberOfProjectors + j;
                                double Val1 = 0.0;
                                double Val2 = 0.0;
                                Val1 +=
                                  (exchangePotentialValAEfromSigma[rpoint] +
                                   corrPotentialValAEfromSigma[rpoint]) *
                                  (productDerCoreDensityWfcDerWfcAE[index] *
                                     SphericalHarmonics[i * numberOfProjectors +
                                                        j] +
                                   2 * tempAEcontributionA[index] *
                                     SphericalHarmonics[i * numberOfProjectors +
                                                        j] +
                                   2 * tempAEcontributionB[index] *
                                     GradThetaSphericalHarmonics
                                       [i * numberOfProjectors + j] +
                                   2 * tempAEcontributionC[index] *
                                     GradPhiSphericalHarmonics
                                       [i * numberOfProjectors + j]);
                                Val2 +=
                                  (exchangePotentialValPSfromSigma[rpoint] +
                                   corrPotentialValPSfromSigma[rpoint]) *
                                  (productDerCoreDensityWfcDerWfcPS[index] *
                                     SphericalHarmonics[i * numberOfProjectors +
                                                        j] +
                                   2 * tempPScontributionA[index] *
                                     SphericalHarmonics[i * numberOfProjectors +
                                                        j] +
                                   2 * tempPScontributionB[index] *
                                     GradThetaSphericalHarmonics
                                       [i * numberOfProjectors + j] +
                                   2 * tempPScontributionC[index] *
                                     GradPhiSphericalHarmonics
                                       [i * numberOfProjectors + j]);
                                double Value = RadialMesh[rpoint] *
                                               (Val1 - Val2) *
                                               pow(rab[rpoint], 2);
                                return (std::fabs(RadialMesh[rpoint]) > 1E-8 ?
                                          Value :
                                          0.0);
                              };

                            double RadialIntegralGGA =
                              simpsonIntegral(0, RmaxIndex + 1, IntegralGGA);
                            Delta_ExcijSigma[i * numberOfProjectors + j] +=
                              RadialIntegralGGA * quadwt * 4.0 * M_PI;
                          } // Proj J
                      }     // Proj I


                    exchangePotentialValAE.resize(0);
                    exchangePotentialValPS.resize(0);
                    corrPotentialValAE.resize(0);
                    corrPotentialValPS.resize(0);
                    atomDensityAllelectron.resize(0);
                    atomDensitySmooth.resize(0);
                    exchangePotentialValAEfromSigma.resize(0);
                    exchangePotentialValPSfromSigma.resize(0);
                    corrPotentialValAEfromSigma.resize(0);
                    corrPotentialValPSfromSigma.resize(0);
                    sigmaAllElectron.resize(0);
                    sigmaSmooth.resize(0);


                    exchangePotentialValAE.clear();
                    exchangePotentialValPS.clear();
                    corrPotentialValAE.clear();
                    corrPotentialValPS.clear();
                    atomDensityAllelectron.clear();
                    atomDensitySmooth.clear();
                    exchangePotentialValAEfromSigma.clear();
                    exchangePotentialValPSfromSigma.clear();
                    corrPotentialValAEfromSigma.clear();
                    corrPotentialValPSfromSigma.clear();
                    sigmaAllElectron.clear();
                    sigmaSmooth.clear();



                  } // qpoint
                    //     // pcout << "Timer LLDA COntribution: " <<
                    //     timerLDAContribution
                    //     //       << std::endl;
                    //     // pcout << "Timer GGA Contribution: " <<
                    //     timerGGAContribution
                    //     //       << std::endl;
                //     // pcout << "Timer GGA part 0: " << timerGGA0Contribution
                //     //       << std::endl;
                //     // pcout << "Timer GGA part A: " << timerGGAAContribution
                //     //       << std::endl;
                //     // pcout << "Timer GGA part B: " << timerGGABContribution
                //     //       << std::endl;
                //     // pcout << "Timer GGA part C: " << timerGGACContribution
                //     //       << std::endl;

              } // GGA case

            pcout << "Delta XC term " << std::endl;
            for (int i = 0; i < numberOfProjectors; i++)
              {
                for (int j = 0; j <= i; j++)
                  {
                    if (!isGGA)
                      Delta_Excij[j * numberOfProjectors + i] =
                        Delta_Excij[i * numberOfProjectors + j];
                    else
                      {
                        double temp =
                          Delta_ExcijDensity[i * numberOfProjectors + j] +
                          2 * Delta_ExcijSigma[i * numberOfProjectors + j] +
                          2 * Delta_ExcijSigma[j * numberOfProjectors + i];
                        Delta_Excij[j * numberOfProjectors + i] = temp;
                        Delta_Excij[i * numberOfProjectors + j] = temp;
                        // pcout
                        //   << Delta_ExcijDensity[i * numberOfProjectors +
                        //   j]
                        //   << " "
                        //   << 2 * Delta_ExcijSigma[i * numberOfProjectors
                        //   + j]
                        //   << " "
                        //   << 2 * Delta_ExcijSigma[j * numberOfProjectors
                        //   + i]
                        //   << std::endl;
                      } // else
                  }     // jProj
              }         // iProj

            d_ExchangeCorrelationEnergyCorrectionTerm[atomId] = Delta_Excij;


          } // iAtom
      }     // locallyOwned atomSet

    // MPI_Barrier(d_mpiCommParent);
    // // Moved to nonLocalHamiltonian
    // for (std::set<unsigned int>::iterator it = d_atomTypes.begin();
    //      it != d_atomTypes.end();
    //      ++it)
    //   {
    //     unsigned int                  atomType  = *it;
    //     std::vector<unsigned int>     atomLists =
    //     d_atomTypesList[atomType]; std::vector<std::vector<int>>
    //     projectorDetailsOfAtom =
    //       d_atomicNumberToWaveFunctionIdDetails[atomType];
    //     unsigned int numberOfProjectorsSq =
    //       projectorDetailsOfAtom.size() * projectorDetailsOfAtom.size();
    //     std::vector<double> DeltaXCAtom(numberOfProjectorsSq *
    //     atomLists.size(),
    //                                     0.0);
    //     for (int i = 0; i < atomLists.size(); i++)
    //       {
    //         for (int iAtomList = 0; iAtomList <
    //         d_LocallyOwnedAtomId.size();
    //              iAtomList++)
    //           {
    //             iAtom = d_LocallyOwnedAtomId[iAtomList];
    //             if (atomLists[i] == iAtom)
    //               {
    //                 std::copy(
    //                   &(d_ExchangeCorrelationEnergyCorrectionTerm[iAtom][0]),
    //                   &(d_ExchangeCorrelationEnergyCorrectionTerm[iAtom][0])
    //                   +
    //                     numberOfProjectorsSq,
    //                   &(DeltaXCAtom[i * numberOfProjectorsSq]));
    //               }
    //           }
    //       }
    //     // Moved to nonLocalHamiltonian
    //     // MPI_Allreduce(MPI_IN_PLACE,
    //     //               &DeltaXCAtom[0],
    //     //               numberOfProjectorsSq * atomLists.size(),
    //     //               MPI_DOUBLE,
    //     //               MPI_SUM,
    //     //               d_mpiCommParent);
    //     for (int i = 0; i < atomLists.size(); i++)
    //       {
    //         std::vector<double> ValueAtom(numberOfProjectorsSq, 0.0);
    //         std::copy(&(DeltaXCAtom[i * numberOfProjectorsSq]),
    //                   &(DeltaXCAtom[i * numberOfProjectorsSq]) +
    //                     numberOfProjectorsSq,
    //                   &(ValueAtom[0]));
    //         d_ExchangeCorrelationEnergyCorrectionTerm[atomLists[i]] =
    //         ValueAtom;
    //       }
    //   } //*it
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::initialiseKineticEnergyCorrection()
  {
    pcout << "PAWClass Init: Reading KE_ij correction terms from XML file..."
          << std::endl;
    const std::map<std::pair<unsigned int, unsigned int>,
                   std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      sphericalFunction =
        d_atomicProjectorFnsContainer->getSphericalFunctions();
    for (std::set<unsigned int>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        unsigned int atomicNumber = *it;
        char         keFileName[256];
        strcpy(keFileName,
               (d_dftfeScratchFolderName + "/z" + std::to_string(atomicNumber) +
                "/" + "KineticEnergyij.dat")
                 .c_str());
        std::cout << "DEBUG: Line 3233" << std::endl;
        std::vector<double> KineticEnergyij;
        dftUtils::readFile(KineticEnergyij, keFileName);

        const unsigned int numberOfProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(atomicNumber);
        const unsigned int numberOfRadialProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(atomicNumber);
        std::vector<double> Tij(numberOfProjectors * numberOfProjectors, 0.0);
        AssertThrow(
          KineticEnergyij.size() ==
            numberOfRadialProjectors * numberOfRadialProjectors,
          dealii::ExcMessage(
            "PAW::Initialization Kinetic Nergy correction term mismatch in number of entries"));
        unsigned int projIndex_i = 0;
        for (unsigned int alpha_i = 0; alpha_i < numberOfRadialProjectors;
             alpha_i++)
          {
            std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn_i =
              sphericalFunction.find(std::make_pair(atomicNumber, alpha_i))
                ->second;
            unsigned int lQuantumNo_i = sphFn_i->getQuantumNumberl();
            for (int mQuantumNo_i = -lQuantumNo_i; mQuantumNo_i <= lQuantumNo_i;
                 mQuantumNo_i++)
              {
                unsigned int projIndex_j = 0;
                for (unsigned int alpha_j = 0;
                     alpha_j < numberOfRadialProjectors;
                     alpha_j++)
                  {
                    std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn_j =
                      sphericalFunction
                        .find(std::make_pair(atomicNumber, alpha_j))
                        ->second;
                    unsigned int lQuantumNo_j = sphFn_j->getQuantumNumberl();
                    for (int mQuantumNo_j = -lQuantumNo_j;
                         mQuantumNo_j <= lQuantumNo_j;
                         mQuantumNo_j++)
                      {
                        if (lQuantumNo_i == lQuantumNo_j &&
                            mQuantumNo_i == mQuantumNo_j)
                          Tij[projIndex_i * numberOfProjectors + projIndex_j] =
                            KineticEnergyij[alpha_i * numberOfRadialProjectors +
                                            alpha_j];
                        pcout
                          << "DEBUG: " << alpha_i << " " << alpha_j << " "
                          << projIndex_j << " " << projIndex_i << " "
                          << Tij[projIndex_i * numberOfProjectors + projIndex_j]
                          << std::endl;
                        projIndex_j++;
                      } // mQuantumNo_j
                  }     // alpha_j


                projIndex_i++;
              } // mQuantumNo_i
          }     // alpha_i
        d_KineticEnergyCorrectionTerm[*it] = Tij;
      } //*it
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeRadialMultipoleData()
  {
    pcout << "PAWClass Init: computing Multipole Table" << std::endl;
    const std::map<std::pair<unsigned int, unsigned int>,
                   std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      sphericalFunction =
        d_atomicProjectorFnsContainer->getSphericalFunctions();
    for (std::set<unsigned int>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        unsigned int       atomicNumber = *it;
        const unsigned int numberOfRadialProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(atomicNumber);
        const unsigned int  lmaxAug = d_dftParamsPtr->noShapeFnsInPAW;
        std::vector<double> multipoleTable(lmaxAug * numberOfRadialProjectors *
                                             numberOfRadialProjectors,
                                           0.0);
        std::vector<double> aePhi        = d_radialWfcValAE[*it];
        std::vector<double> psPhi        = d_radialWfcValPS[*it];
        std::vector<double> radialMesh   = d_radialMesh[*it];
        std::vector<double> jacobianData = d_radialJacobianData[*it];
        unsigned int        meshSize     = radialMesh.size();
        unsigned int        rmaxAugIndex = d_RmaxAugIndex[*it];
        for (unsigned int L = 0; L < lmaxAug; L++)
          {
            for (unsigned int alpha_i = 0; alpha_i < numberOfRadialProjectors;
                 alpha_i++)
              {
                std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn_i =
                  sphericalFunction.find(std::make_pair(atomicNumber, alpha_i))
                    ->second;
                int lQuantumNo_i = sphFn_i->getQuantumNumberl();
                for (unsigned int alpha_j = 0;
                     alpha_j < numberOfRadialProjectors;
                     alpha_j++)
                  {
                    std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn_j =
                      sphericalFunction
                        .find(std::make_pair(atomicNumber, alpha_j))
                        ->second;
                    int    lQuantumNo_j = sphFn_j->getQuantumNumberl();
                    double Value        = 0.0;
                    if (L >= std::abs(lQuantumNo_i - lQuantumNo_j) &&
                        L <= (lQuantumNo_i + lQuantumNo_j))
                      {
                        Value = multipoleIntegrationGrid(aePhi.data() +
                                                           (alpha_i)*meshSize,
                                                         aePhi.data() +
                                                           (alpha_j)*meshSize,
                                                         radialMesh,
                                                         jacobianData,
                                                         L,
                                                         0,
                                                         rmaxAugIndex + 1) -
                                multipoleIntegrationGrid(psPhi.data() +
                                                           (alpha_i)*meshSize,
                                                         psPhi.data() +
                                                           (alpha_j)*meshSize,
                                                         radialMesh,
                                                         jacobianData,
                                                         L,
                                                         0,
                                                         rmaxAugIndex + 1);
                      }
                    multipoleTable[L * numberOfRadialProjectors *
                                     numberOfRadialProjectors +
                                   alpha_i * numberOfRadialProjectors +
                                   alpha_j] = Value;
                    pcout << alpha_i << " " << alpha_j << " " << Value
                          << std::endl;
                  } // alpha_j
              }     // alpha_i
          }         // L



        d_multipole[*it] = multipoleTable;
      } //*it
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeInverseOfMultipoleData()
  {
    pcout << "PAWClass: Computing inverse multipole data from XML file..."
          << std::endl;
    for (std::set<unsigned int>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        unsigned int        atomicNumber   = *it;
        std::vector<double> multipoleTable = d_multipole[atomicNumber];
        const unsigned int  numberOfProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(atomicNumber);
        const unsigned int numberOfRadialProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(atomicNumber);
        const std::map<std::pair<unsigned int, unsigned int>,
                       std::shared_ptr<AtomCenteredSphericalFunctionBase>>
          sphericalFunction =
            d_atomicProjectorFnsContainer->getSphericalFunctions();
        std::vector<double> Multipole(numberOfProjectors * numberOfProjectors,
                                      0.0);
        int                 projIndexI = 0;
        for (int iProj = 0; iProj < numberOfRadialProjectors; iProj++)
          {
            std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn_i =
              sphericalFunction.find(std::make_pair(atomicNumber, iProj))
                ->second;
            int lQuantumNo_i = sphFn_i->getQuantumNumberl();
            for (int mQuantumNumber_i = -lQuantumNo_i;
                 mQuantumNumber_i <= lQuantumNo_i;
                 mQuantumNumber_i++)
              {
                int projIndexJ = 0;
                for (int jProj = 0; jProj < numberOfRadialProjectors; jProj++)
                  {
                    std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn_j =
                      sphericalFunction
                        .find(std::make_pair(atomicNumber, jProj))
                        ->second;
                    int lQuantumNo_j = sphFn_j->getQuantumNumberl();
                    for (int mQuantumNumber_j = -lQuantumNo_j;
                         mQuantumNumber_j <= lQuantumNo_j;
                         mQuantumNumber_j++)
                      {
                        Multipole[projIndexI * numberOfProjectors +
                                  projIndexJ] =
                          gaunt(lQuantumNo_i,
                                lQuantumNo_j,
                                0,
                                mQuantumNumber_i,
                                mQuantumNumber_j,
                                0) *
                          multipoleTable[iProj * numberOfRadialProjectors +
                                         jProj] *
                          sqrt(4 * M_PI);
                        projIndexJ++;
                      } // mQuantumNumber_j

                  } // jProj
                projIndexI++;
              } // mQuantumNumber_i



          } // iProj
        const char          uplo = 'L';
        const int           N    = numberOfProjectors;
        std::vector<double> A    = Multipole;
        // pcout << "Multipole Table: " << std::endl;
        // for (int i = 0; i < numberOfProjectors; i++)
        //   {
        //     for (int j = 0; j < numberOfProjectors; j++)
        //       pcout << A[i * numberOfProjectors + j] << " ";
        //     pcout << std::endl;
        //   }

        dftfe::linearAlgebraOperations::inverse(&A[0], N);
        d_multipoleInverse[atomicNumber] = A;

        // pcout << "Multipole Table Inverse: " << std::endl;
        // for (int i = 0; i < numberOfProjectors; i++)
        //   {
        //     for (int j = 0; j < numberOfProjectors; j++)
        //       pcout << A[i * numberOfProjectors + j] << " ";
        //     pcout << std::endl;
        //   }
      } //*it
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::createAtomTypesList(
    const std::vector<std::vector<double>> &atomLocations)
  {
    pcout << "Creating Atom Type List " << std::endl;
    for (std::set<unsigned int>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        unsigned                  atomType = *it;
        std::vector<unsigned int> atomLocation;
        for (int iAtom = 0; iAtom < atomLocations.size(); iAtom++)
          {
            if (atomLocations[iAtom][0] == atomType)
              atomLocation.push_back(iAtom);
          }
        d_atomTypesList[atomType] = atomLocation;
      }
    if (d_n_mpi_processes > atomLocations.size())
      {
        if (d_this_mpi_process >= atomLocations.size())
          {
          }
        else
          d_LocallyOwnedAtomId.push_back(d_this_mpi_process);
      }
    else
      {
        int no_atoms       = atomLocations.size() / d_n_mpi_processes;
        int remainderAtoms = atomLocations.size() % d_n_mpi_processes;
        for (int i = 0; i < no_atoms; i++)
          {
            d_LocallyOwnedAtomId.push_back(d_this_mpi_process * no_atoms + i);
          }
        if (d_this_mpi_process < remainderAtoms)
          d_LocallyOwnedAtomId.push_back(d_n_mpi_processes * no_atoms +
                                         d_this_mpi_process);
      }
    const std::vector<unsigned int> &atomicNumber =
      d_atomicShapeFnsContainer->getAtomicNumbers();

    for (unsigned int i = 0; i < d_LocallyOwnedAtomId.size(); i++)
      {
        unsigned int atomId = d_LocallyOwnedAtomId[i];
        unsigned int Znum   = atomicNumber[atomId];
        d_nProjPerTask += d_atomicProjectorFnsContainer
                            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
      }
    d_nProjSqTotal = 0;
    for (unsigned int i = 0; i < atomicNumber.size(); i++)
      {
        unsigned int Znum = atomicNumber[i];
        unsigned int numberOfProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);

        d_projectorStartIndex.push_back(d_nProjSqTotal);
        d_nProjSqTotal += (numberOfProjectors * (numberOfProjectors + 1)) / 2;
      }
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::initialiseDataonRadialMesh()
  {
    for (std::set<unsigned int>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        unsigned int        Znum           = *it;
        std::vector<double> radialMesh     = d_radialMesh[*it];
        std::vector<double> jacobianData   = d_radialJacobianData[*it];
        const unsigned int  radialMeshSize = radialMesh.size();
        const unsigned int  rmaxAugIndex   = d_RmaxAugIndex[*it];
        const unsigned int  numberOfProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        const unsigned int numberOfRadialProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);
        std::vector<double> productOfAEpartialWfc(
          radialMeshSize * numberOfProjectors * numberOfProjectors, 0.0);
        std::vector<double> productOfPSpartialWfc(
          radialMeshSize * numberOfProjectors * numberOfProjectors, 0.0);

        // Core densit changes

        for (int rPoint = 0; rPoint < radialMeshSize; rPoint++)
          {
            double r                = radialMesh[rPoint];
            int    projectorIndex_i = 0;
            for (int alpha_i = 0; alpha_i < numberOfRadialProjectors; alpha_i++)
              {
                std::shared_ptr<AtomCenteredSphericalFunctionBase> AEsphFn_i =
                  d_atomicAEPartialWaveFnsMap
                    .find(std::make_pair(Znum, alpha_i))
                    ->second;
                std::shared_ptr<AtomCenteredSphericalFunctionBase> PSsphFn_i =
                  d_atomicPSPartialWaveFnsMap
                    .find(std::make_pair(Znum, alpha_i))
                    ->second;
                int    lQuantumNo_i  = AEsphFn_i->getQuantumNumberl();
                double radialValAE_i = AEsphFn_i->getRadialValue(r);
                double radialValPS_i = PSsphFn_i->getRadialValue(r);
                for (int mQuantumNo_i = -lQuantumNo_i;
                     mQuantumNo_i <= lQuantumNo_i;
                     mQuantumNo_i++)
                  {
                    int projectorIndex_j = 0;
                    for (int alpha_j = 0; alpha_j < numberOfRadialProjectors;
                         alpha_j++)
                      {
                        std::shared_ptr<AtomCenteredSphericalFunctionBase>
                          AEsphFn_j = d_atomicAEPartialWaveFnsMap
                                        .find(std::make_pair(Znum, alpha_j))
                                        ->second;
                        std::shared_ptr<AtomCenteredSphericalFunctionBase>
                          PSsphFn_j = d_atomicPSPartialWaveFnsMap
                                        .find(std::make_pair(Znum, alpha_j))
                                        ->second;
                        double radialValAE_j = AEsphFn_j->getRadialValue(r);
                        double radialValPS_j = PSsphFn_j->getRadialValue(r);
                        int    lQuantumNo_j  = AEsphFn_j->getQuantumNumberl();
                        for (int mQuantumNo_j = -lQuantumNo_j;
                             mQuantumNo_j <= lQuantumNo_j;
                             mQuantumNo_j++)
                          {
                            unsigned int index =
                              rPoint * numberOfProjectors * numberOfProjectors +
                              projectorIndex_i * numberOfProjectors +
                              projectorIndex_j;
                            productOfAEpartialWfc[index] =
                              radialValAE_j * radialValAE_i;
                            productOfPSpartialWfc[index] =
                              radialValPS_i * radialValPS_j;
                            projectorIndex_j++;
                          } // mQuantumNo_j
                      }     // alpha_j

                    projectorIndex_i++;
                  } // mQuantumNo_i



              } // alpha_i

          } // rPoint
        d_productOfAEpartialWfc[*it] = productOfAEpartialWfc;
        d_productOfPSpartialWfc[*it] = productOfPSpartialWfc;
        for (int rPoint = 0; rPoint < radialMeshSize; rPoint++)
          {
            d_atomCoreDensityAE[*it][rPoint] /= sqrt(4 * M_PI);
            d_atomCoreDensityPS[*it][rPoint] /= sqrt(4 * M_PI);
          }
        const bool isGGA = d_excManagerPtr->getDensityBasedFamilyType() ==
                           densityFamilyType::GGA;
        if (isGGA)
          {
            std::vector<double> derAECoreSq(radialMeshSize, 0.0);
            std::vector<double> derPSCoreSq(radialMeshSize, 0.0);
            std::vector<double> derAECoreWfc(
              radialMeshSize * numberOfProjectors * numberOfProjectors, 0.0);
            std::vector<double> derPSCoreWfc(
              radialMeshSize * numberOfProjectors * numberOfProjectors, 0.0);
            unsigned int        npj_4 = pow(numberOfProjectors, 4);
            unsigned int        npj_3 = pow(numberOfProjectors, 3);
            unsigned int        npj_2 = pow(numberOfProjectors, 2);
            std::vector<double> productValDerPS(npj_2 * radialMeshSize, 0.0);
            std::vector<double> productValDerAE(npj_2 * radialMeshSize, 0.0);

            std::vector<double> productValsPS(npj_2 * radialMeshSize, 0.0);
            std::vector<double> productValsAE(npj_2 * radialMeshSize, 0.0);

            std::vector<double> productValsijklAE(radialMeshSize * npj_4, 0.0);
            std::vector<double> productValsijklPS(radialMeshSize * npj_4, 0.0);

            std::vector<double> productDerValsijklAE(radialMeshSize * npj_4,
                                                     0.0);
            std::vector<double> productDerValsijklPS(radialMeshSize * npj_4,
                                                     0.0);

            std::vector<double> derCoreRhoAE = d_radialCoreDerAE[*it];
            std::vector<double> derCoreRhoPS = d_radialCoreDerPS[*it];

            std::vector<double> derWfcAE = d_radialWfcDerAE[*it];
            std::vector<double> derWfcPS = d_radialWfcDerPS[*it];
            std::vector<double> WfcAE    = d_radialWfcValAE[*it];
            std::vector<double> WfcPS    = d_radialWfcValPS[*it];

            // map of projectroIndex tot radialProjectorId
            std::vector<unsigned int> projectorIndexRadialIndexMap(
              numberOfProjectors);
            unsigned int projectorIndex = 0;
            for (unsigned int alpha = 0; alpha < numberOfRadialProjectors;
                 alpha++)
              {
                std::shared_ptr<AtomCenteredSphericalFunctionBase> AEsphFn =
                  d_atomicAEPartialWaveFnsMap.find(std::make_pair(Znum, alpha))
                    ->second;
                int lQuantumNo = AEsphFn->getQuantumNumberl();
                for (int mQuantumNo = -lQuantumNo; mQuantumNo <= lQuantumNo;
                     mQuantumNo++)
                  {
                    projectorIndexRadialIndexMap[projectorIndex] = alpha;
                    projectorIndex++;
                  }
              }

            for (int rpoint = 0; rpoint < radialMeshSize; rpoint++)
              {
                // CoreDensity Changes Pending
                if (d_atomTypeCoreFlagMap[*it])
                  {
                    derAECoreSq[rpoint] = 1 / (4 * M_PI) *
                                          derCoreRhoAE[rpoint] *
                                          derCoreRhoAE[rpoint];
                    derPSCoreSq[rpoint] = 1 / (4 * M_PI) *
                                          derCoreRhoPS[rpoint] *
                                          derCoreRhoPS[rpoint];
                  }
                for (int projectorIndex_i = 0;
                     projectorIndex_i < numberOfProjectors;
                     projectorIndex_i++)
                  {
                    unsigned int alpha_i =
                      projectorIndexRadialIndexMap[projectorIndex_i];
                    for (int projectorIndex_j = 0;
                         projectorIndex_j < numberOfProjectors;
                         projectorIndex_j++)
                      {
                        unsigned int alpha_j =
                          projectorIndexRadialIndexMap[projectorIndex_j];
                        unsigned int index =
                          rpoint * numberOfProjectors * numberOfProjectors +
                          projectorIndex_i * numberOfProjectors +
                          projectorIndex_j;
                        if (d_atomTypeCoreFlagMap[*it])
                          {
                            derAECoreWfc[index] =
                              1 / sqrt(4 * M_PI) * derCoreRhoAE[rpoint] *
                              WfcAE[alpha_i * radialMeshSize + rpoint] *
                              derWfcAE[alpha_j * radialMeshSize + rpoint];
                            derPSCoreWfc[index] =
                              1 / sqrt(4 * M_PI) * derCoreRhoPS[rpoint] *
                              WfcPS[alpha_i * radialMeshSize + rpoint] *
                              derWfcPS[alpha_j * radialMeshSize + rpoint];
                          }
                        double ValAEij =
                          WfcAE[alpha_i * radialMeshSize + rpoint] *
                          WfcAE[alpha_j * radialMeshSize + rpoint];
                        double ValPSij =
                          WfcPS[alpha_i * radialMeshSize + rpoint] *
                          WfcPS[alpha_j * radialMeshSize + rpoint];

                        double DerAEij =
                          WfcAE[alpha_i * radialMeshSize + rpoint] *
                          derWfcAE[alpha_j * radialMeshSize + rpoint];
                        double DerPSij =
                          WfcPS[alpha_i * radialMeshSize + rpoint] *
                          derWfcPS[alpha_j * radialMeshSize + rpoint];
                        productValDerAE[index] = DerAEij;
                        productValDerPS[index] = DerPSij;
                        for (int projectorIndex_k = 0;
                             projectorIndex_k < numberOfProjectors;
                             projectorIndex_k++)
                          {
                            unsigned int alpha_k =
                              projectorIndexRadialIndexMap[projectorIndex_k];
                            for (int projectorIndex_l = 0;
                                 projectorIndex_l < numberOfProjectors;
                                 projectorIndex_l++)
                              {
                                unsigned int alpha_l =
                                  projectorIndexRadialIndexMap
                                    [projectorIndex_l];
                                unsigned int indexijkl =
                                  rpoint * npj_4 + projectorIndex_i * npj_3 +
                                  projectorIndex_j * npj_2 +
                                  projectorIndex_k * numberOfProjectors +
                                  projectorIndex_l;
                                productValsijklAE[indexijkl] =
                                  ValAEij *
                                  WfcAE[alpha_k * radialMeshSize + rpoint] *
                                  WfcAE[alpha_l * radialMeshSize + rpoint];
                                productValsijklPS[indexijkl] =
                                  ValPSij *
                                  WfcPS[alpha_k * radialMeshSize + rpoint] *
                                  WfcPS[alpha_l * radialMeshSize + rpoint];
                                productDerValsijklAE[indexijkl] =
                                  DerAEij *
                                  WfcAE[alpha_k * radialMeshSize + rpoint] *
                                  derWfcAE[alpha_l * radialMeshSize + rpoint];
                                productDerValsijklPS[indexijkl] =
                                  DerPSij *
                                  WfcPS[alpha_k * radialMeshSize + rpoint] *
                                  derWfcPS[alpha_l * radialMeshSize + rpoint];
                              } // projectorIndex_l

                          } // projectorIndex_k
                      }     // projectorIndex_j
                  }         // projectorIndex_i



              } // rPoint
            d_gradCoreSqAE[*it]                     = derAECoreSq;
            d_gradCoreSqPS[*it]                     = derPSCoreSq;
            d_productOfAEpartialWfcDer[*it]         = productValDerAE;
            d_productOfPSpartialWfcDer[*it]         = productValDerPS;
            d_productOfAEpartialWfcValue[*it]       = productValsAE;
            d_productOfPSpartialWfcValue[*it]       = productValsPS;
            d_productDerCoreDensityWfcDerWfcAE[*it] = derAECoreWfc;
            d_productDerCoreDensityWfcDerWfcPS[*it] = derPSCoreWfc;
            d_tensorWfcAE[*it]                      = productValsijklAE;
            d_tensorWfcPS[*it]                      = productValsijklPS;
            d_tensorWfcDerAE[*it]                   = productDerValsijklAE;
            d_tensorWfcDerPS[*it]                   = productDerValsijklPS;

          } // isGGA

      } //*it
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeDijFromPSIinitialGuess(
    const dftfe::utils::MemoryStorage<ValueType,
                                      dftfe::utils::MemorySpace::HOST> *X,
    const unsigned int         numberOfElectrons,
    const unsigned int         totalNumWaveFunctions,
    const unsigned int         quadratureIndex,
    const std::vector<double> &kPointWeights,
    const MPI_Comm &           interpoolcomm,
    const MPI_Comm &           interBandGroupComm)
  {
    if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
      {
        MPI_Barrier(d_mpiCommParent);
        const unsigned int numKPoints   = kPointWeights.size();
        const unsigned int numLocalDofs = d_BasisOperatorHostPtr->nOwnedDofs();
        const unsigned int totalLocallyOwnedCells =
          d_BasisOperatorHostPtr->nCells();
        const unsigned int numNodesPerElement =
          d_BasisOperatorHostPtr->nDofsPerCell();
        // band group parallelization data structures
        const unsigned int numberBandGroups =
          dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
        const unsigned int bandGroupTaskId =
          dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
        std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
        dftUtils::createBandParallelizationIndices(
          interBandGroupComm,
          totalNumWaveFunctions,
          bandGroupLowHighPlusOneIndices);

        const unsigned int BVec = std::min(d_dftParamsPtr->chebyWfcBlockSize,
                                           bandGroupLowHighPlusOneIndices[1]);

        dftfe::utils::MemoryStorage<ValueType, memorySpace> tempCellNodalData;

        const double spinPolarizedFactor =
          (d_dftParamsPtr->spinPolarized == 1) ? 1.0 : 2.0;
        const unsigned int numSpinComponents =
          (d_dftParamsPtr->spinPolarized == 1) ? 2 : 1;

        const ValueType zero                    = 0;
        const ValueType scalarCoeffAlphaRho     = 1.0;
        const ValueType scalarCoeffBetaRho      = 1.0;
        const ValueType scalarCoeffAlphaGradRho = 1.0;
        const ValueType scalarCoeffBetaGradRho  = 1.0;

        const unsigned int cellsBlockSize =
          memorySpace == dftfe::utils::MemorySpace::DEVICE ? 50 : 1;
        const unsigned int numCellBlocks =
          totalLocallyOwnedCells / cellsBlockSize;
        const unsigned int remCellBlockSize =
          totalLocallyOwnedCells - numCellBlocks * cellsBlockSize;
        d_BasisOperatorHostPtr->reinit(BVec, cellsBlockSize, quadratureIndex);
        const unsigned int numQuadPoints =
          d_BasisOperatorHostPtr->nQuadsPerCell();
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
              partialOccupVecHost(BVec, 0.0);
        auto &partialOccupVec = partialOccupVecHost;

        dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
          *flattenedArrayBlock;

        dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
                     projectorKetTimesVector;
        unsigned int previousSize = 0;
        pcout << "DEBUG: Line 3863" << std::endl;
        for (unsigned int kPoint = 0; kPoint < kPointWeights.size(); ++kPoint)
          {
            unsigned int numberOfRemainingElectrons = numberOfElectrons;
            for (unsigned int spinIndex = 0; spinIndex < numSpinComponents;
                 ++spinIndex)
              {
                d_nonLocalOperator->initialiseOperatorActionOnX(kPoint);

                for (unsigned int jvec = 0; jvec < totalNumWaveFunctions;
                     jvec += BVec)
                  {
                    pcout << "DEBUG: Line 3873" << std::endl;
                    const unsigned int currentBlockSize =
                      std::min(BVec, totalNumWaveFunctions - jvec);
                    flattenedArrayBlock =
                      &(d_BasisOperatorHostPtr->getMultiVector(currentBlockSize,
                                                               0));
                    d_nonLocalOperator->initialiseFlattenedDataStructure(
                      currentBlockSize, projectorKetTimesVector);
                    if ((jvec + currentBlockSize) <=
                          bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId +
                                                         1] &&
                        (jvec + currentBlockSize) >
                          bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                      {
                        // compute occupancy Vector
                        for (unsigned int iEigenVec = 0;
                             iEigenVec < currentBlockSize;
                             ++iEigenVec)
                          {
                            double OccupancyFactor = 0.0;
                            if (numberOfRemainingElectrons == 1)
                              {
                                OccupancyFactor            = 0.5;
                                numberOfRemainingElectrons = 0;
                              }
                            else if (numberOfRemainingElectrons > 1)
                              {
                                OccupancyFactor = 1.0;
                                numberOfRemainingElectrons -=
                                  spinPolarizedFactor;
                              }



                            *(partialOccupVecHost.begin() + iEigenVec) =
                              OccupancyFactor * kPointWeights[kPoint] *
                              spinPolarizedFactor;
                          }
                        for (unsigned int iNode = 0; iNode < numLocalDofs;
                             ++iNode)
                          std::memcpy(flattenedArrayBlock->data() +
                                        iNode * currentBlockSize,
                                      X->data() +
                                        numLocalDofs * totalNumWaveFunctions *
                                          (numSpinComponents * kPoint +
                                           spinIndex) +
                                        iNode * totalNumWaveFunctions + jvec,
                                      currentBlockSize * sizeof(ValueType));
                        flattenedArrayBlock->updateGhostValues();
                        d_BasisOperatorHostPtr->distribute(
                          *(flattenedArrayBlock));

                        for (int iblock = 0; iblock < (numCellBlocks + 1);
                             iblock++)
                          {
                            const unsigned int currentCellsBlockSize =
                              (iblock == numCellBlocks) ? remCellBlockSize :
                                                          cellsBlockSize;
                            if (currentCellsBlockSize > 0)
                              {
                                const unsigned int startingCellId =
                                  iblock * cellsBlockSize;
                                if (currentCellsBlockSize * currentBlockSize !=
                                    previousSize)
                                  {
                                    tempCellNodalData.resize(
                                      currentCellsBlockSize * currentBlockSize *
                                      numLocalDofs);
                                    previousSize =
                                      currentCellsBlockSize * currentBlockSize;
                                  }
                                d_BasisOperatorHostPtr
                                  ->extractToCellNodalDataKernel(
                                    *(flattenedArrayBlock),
                                    tempCellNodalData.data(),
                                    std::pair<unsigned int, unsigned int>(
                                      startingCellId,
                                      startingCellId + currentCellsBlockSize));
                                d_nonLocalOperator->applyCconjtransOnX(
                                  tempCellNodalData.data(),
                                  std::pair<unsigned int, unsigned int>(
                                    startingCellId,
                                    startingCellId + currentCellsBlockSize));
                                // Call apply CconjTranspose


                              } // non-trivial cell block check
                          }     // cells block loop

                        d_nonLocalOperator->applyAllReduceOnCconjtransX(
                          projectorKetTimesVector);
                        d_nonLocalOperator
                          ->copyBackFromDistributedVectorToLocalDataStructure(
                            projectorKetTimesVector, partialOccupVec);
                        pcout << "DEBUG: Line 3967" << std::endl;
                        computeDij(
                          false, jvec, currentBlockSize, spinIndex, kPoint);
                        // Call computeDij
                      }
                  }
              }
          }



        communicateDijAcrossAllProcessors(TypeOfField::In,
                                          interpoolcomm,
                                          interBandGroupComm);
      }
  }
} // namespace dftfe
