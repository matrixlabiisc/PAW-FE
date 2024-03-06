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
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
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
                "/density.inp")
                 .c_str());
        char coreDataFile[256];
        strcpy(coreDataFile,
               (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                "/coreDensity.inp")
                 .c_str());

        for (unsigned int i = 0; i < d_nOMPThreads; i++)
          {
            d_atomicValenceDensityVector[i][*it] = std::make_shared<
              AtomCenteredSphericalFunctionValenceDensitySpline>(
              valenceDataFile, 1E-10, false);
            d_atomicCoreDensityVector[i][*it] =
              std::make_shared<AtomCenteredSphericalFunctionCoreDensitySpline>(
                coreDataFile, 1E-12, true);
          }
        if (d_atomicCoreDensityVector[0][atomicNumber]->isDataPresent())
          d_atomTypeCoreFlagMap[atomicNumber] = true;
        else
          d_atomTypeCoreFlagMap[atomicNumber] = false;
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
    createAtomCenteredSphericalFunctionsForDensities();
    createAtomCenteredSphericalFunctionsForProjectors();
    createAtomCenteredSphericalFunctionsForZeroPotential();

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

    computeNonlocalPseudoPotentialConstants();
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::initialiseNonLocalContribution(
    const std::vector<std::vector<double>> &atomLocations,
    const std::vector<int> &                imageIds,
    const std::vector<std::vector<double>> &periodicCoords,
    const std::vector<double> &             kPointWeights,
    const std::vector<double> &             kPointCoordinates,
    const bool                              updateNonlocalSparsity)
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
  pawClass<ValueType, memorySpace>::computeNonlocalPseudoPotentialConstants()
  {
    for (std::set<unsigned int>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        const unsigned int Znum = *it;
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
        char denominatorDataFileName[256];
        strcpy(denominatorDataFileName,
               (d_dftfeScratchFolderName + "/z" + std::to_string(Znum) + "/" +
                "denom.dat")
                 .c_str());
        std::vector<std::vector<double>> denominator(0);
        dftUtils::readFile(numberOfRadialProjectors,
                           denominator,
                           denominatorDataFileName);
        std::vector<double> pseudoPotentialConstants(numTotalProjectors, 0.0);
        unsigned int        ProjId = 0;
        for (unsigned int iProj = 0; iProj < numberOfRadialProjectors; iProj++)
          {
            std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn =
              sphericalFunction.find(std::make_pair(Znum, iProj))->second;
            unsigned int lQuantumNumber = sphFn->getQuantumNumberl();
            for (int l = 0; l < 2 * lQuantumNumber + 1; l++)
              {
                pseudoPotentialConstants[ProjId] = denominator[iProj][iProj];
                ProjId++;
              }
          }
        d_atomicNonLocalPseudoPotentialConstants[Znum] =
          pseudoPotentialConstants;

      } //*it
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
        readPseudoDataFileNames >> numberOfProjectors;
        readPseudoDataFileNames.ignore();
        projectorIdDetails.resize(numberOfProjectors);
        std::string   readLine;
        std::set<int> radFunctionIds;
        atomicFunctionIdDetails.resize(numberOfProjectors);
        for (unsigned int i = 0; i < numberOfProjectors; ++i)
          {
            std::vector<int> &radAndAngularFunctionId =
              atomicFunctionIdDetails[i];
            radAndAngularFunctionId.resize(3, 0);
            std::getline(readPseudoDataFileNames, readLine);

            std::istringstream lineString(readLine);
            unsigned int       count = 0;
            int                Id;
            double             mollifierRadius;
            std::string        dummyString;
            while (lineString >> dummyString)
              {
                if (count < 3)
                  {
                    Id = atoi(dummyString.c_str());

                    if (count == 1)
                      radFunctionIds.insert(Id);
                    radAndAngularFunctionId[count] = Id;
                  }

                if (count > 3)
                  {
                    std::cerr << "Invalid argument in the SingleAtomData file"
                              << std::endl;
                    exit(-1);
                  }

                count++;
              }
          }
        std::string  tempProjRadialFunctionFileName;
        unsigned int numProj;
        unsigned int alpha = 0;
        for (std::set<int>::iterator i = radFunctionIds.begin();
             i != radFunctionIds.end();
             ++i)
          {
            char         projRadialFunctionFileName[512];
            unsigned int lQuantumNo = *i;
            readPseudoDataFileNames >> tempProjRadialFunctionFileName;
            readPseudoDataFileNames >> numProj;
            strcpy(projRadialFunctionFileName,
                   (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                    "/" + tempProjRadialFunctionFileName)
                     .c_str());

            //
            // 2D vector to store the radial coordinate and its
            // corresponding function value
            std::vector<std::vector<double>> radialFunctionData(0);

            //
            // read the radial function file
            //
            dftUtils::readFile(numProj + 1,
                               radialFunctionData,
                               projRadialFunctionFileName);

            for (int j = 1; j < numProj + 1; j++)
              {
                d_atomicProjectorFnsMap[std::make_pair(Znum, alpha)] =
                  std::make_shared<
                    AtomCenteredSphericalFunctionProjectorSpline>(
                    projRadialFunctionFileName,
                    lQuantumNo,
                    0,
                    j,
                    numProj + 1,
                    1E-12);
                alpha++;
              }
          } // i loop

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
            std::make_shared<AtomCenteredSphericalFunctionLocalPotentialSpline>(
              LocalDataFile,
              d_atomTypeAtributes[*it],
              d_reproducible_output ? 1.0e-8 : 1.0e-7,
              d_reproducible_output ? 8.0001 : 10.0001);

      } //*it loop
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::getRadialValenceDensity(unsigned int Znum,
                                                            double       rad)
  {
    unsigned int threadId = omp_get_thread_num();
    double       Value =
      d_atomicValenceDensityVector[threadId][Znum]->getRadialValue(rad);

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
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::getRadialZeroPotential(unsigned int Znum,
                                                           double       rad)
  {
    unsigned int threadId = omp_get_thread_num();
    double Value = d_atomicZeroPotVector[threadId][Znum]->getRadialValue(rad);
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
  pawClass<ValueType, memorySpace>::initialiseDataonRadialMesh()
  {}

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::initialiseColoumbicEnergyCorrection()
  {
    pcout << "Initlising Delta C Correction Term" << std::endl;
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
            oneTermPoissonPotential(aeCoreDensity[0],
                                    0,
                                    0,
                                    RmaxIndex,
                                    radialMesh,
                                    rab,
                                    NcorePotential);
            oneTermPoissonPotential(psCoreDensity[0],
                                    0,
                                    0,
                                    RmaxIndex,
                                    radialMesh,
                                    rab,
                                    tildeNCorePotential);
          }
        std::vector<std::vector<double>> gLPotential;
        for (int lShapeFn = 0; lShapeFn < numRadialShapeFunctions; lShapeFn++)
          {
            std::vector<double> tempPotential;
            oneTermPoissonPotential(shapeFnRadial[lShapeFn * meshSize],
                                    0,
                                    0,
                                    RmaxIndex,
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
                    twoTermPoissonPotential(aePhi[iProj * meshSize],
                                            aePhi[jProj * meshSize],
                                            lShapeFn,
                                            0,
                                            RmaxIndex,
                                            rab,
                                            radialMesh,
                                            tempPotentialAE);
                    twoTermPoissonPotential(psPhi[iProj * meshSize],
                                            psPhi[jProj * meshSize],
                                            lShapeFn,
                                            0,
                                            RmaxIndex,
                                            rab,
                                            radialMesh,
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
                    aePhi[iProj * meshSize],
                    aePhi[jProj * meshSize],
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

        valueTemp =
          -sqrt(4 * M_PI) * (*it) *
          integralOfDensity(aeCoreDensity, radialMesh, rab, 0, RmaxIndex + 1);

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
                  }
                if (l_i == l_j && m_i == m_j)
                  {
                    Delta_Cij[i * numberOfProjectors + j] +=
                      -(*it) * integralAllElectronPhiIphiJContribution
                                 [radProjIndexI * numberOfRadialProjectors +
                                  radProjIndexJ];
                  }
                double multipoleValue =
                  multipoleTable[radProjIndexI * numberOfRadialProjectors +
                                 radProjIndexJ];
                Delta_Cij[i * numberOfProjectors + j] -=
                  multipoleValue * GauntValueij *
                  (dL0 * ShapeFnContribution[0]);
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


                        for (int lproj = lmin; lproj <= lmax; lproj++)
                          {
                            bool flag = false;
                            for (int mproj = -lproj; mproj <= lproj; mproj++)
                              {
                                double CG1, CG2;
                                CG1 = gaunt(l_i, l_j, lproj, m_i, m_j, mproj);
                                CG2 = gaunt(l_k, l_l, lproj, m_k, m_l, mproj);
                                if (std::fabs(CG1 * CG2) > 1E-10)
                                  flag = true;
                              } // mproj
                            if (flag)
                              {
                                if (phiIphiJPotentialAE
                                      .find(std::make_pair(index_ij, lproj))
                                      ->second.size() > 0)
                                  {
                                    std::vector<double> potentialPhiIPhiJ =
                                      phiIphiJPotentialAE
                                        .find(std::make_pair(index_ij, lproj))
                                        ->second;
                                    std::vector<double>
                                      potentialTildePhiITildePhiJ =
                                        phiIphiJPotentialPS
                                          .find(std::make_pair(index_ij, lproj))
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

                                    for (int mproj = -lproj; mproj <= lproj;
                                         mproj++)
                                      {
                                        double CG1, CG2;
                                        CG1 = gaunt(
                                          l_i, l_j, lproj, m_i, m_j, mproj);
                                        CG2 = gaunt(
                                          l_k, l_l, lproj, m_k, m_l, mproj);
                                        if (std::fabs(CG1 * CG2) > 1E-10)
                                          TotalContribution +=
                                            (TotalValue)*CG1 * CG2;

                                      } // mproj
                                    Delta_Cijkl[index] +=
                                      0.5 * TotalContribution;
                                    // pcout
                                    //   << "DEBUG: Value Check: " <<
                                    //   (TotalValue)
                                    //   << " " << i << " " << j << " " << k <<
                                    //   " "
                                    //   << l << " " << lproj << std::endl;
                                  }

                                else
                                  {
                                    pcout
                                      << "Mising Entries for lproj: " << lproj
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
                    radialPSWaveFunctionsData[i * radialMesh.size()],
                    radialPSWaveFunctionsData[j * radialMesh.size()],
                    zeroPotentialData[0],
                    radialMesh,
                    rab,
                    0,
                    RmaxIndex + 1);
                radialIntegralData[j * numberOfRadialProjectors + i] =
                  radialIntegralData[i * numberOfRadialProjectors + j];
              } // j
          }     // i

        int projIndexI = 0;
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
                        projIndexJ++;
                      } // mQuantumNumber_j

                  } // jProj
                projIndexI++;
              } // mQuantumNumber_i



          } // iProj

        d_zeroPotentialij[*it] = tempZeroPotentialIJ;
      } //*it
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType,
           memorySpace>::initialiseExchangeCorrelationEnergyCorrection()
  {
    std::map<unsigned int, std::vector<std::vector<int>> projectorDetailsOfAtomFull;
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



        std::vector<std::vector<int>> temp_projectorDetailsOfAtom;
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
                temp_projectorDetailsOfAtom.push_back(temp);
              }
          }

        pcout << "DEBUG check number of entries are matching? "
              << numberOfProjectors << " " << temp_projectorDetailsOfAtom.size()
              << std::endl;
        projectorDetailsOfAtomFull[*it] = temp_projectorDetailsOfAtom;      
      } //*it
      
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::initialiseKineticEnergyCorrection()
  {
    pcout << "PAWClass: Reading KE_ij correction terms from XML file..."
          << std::endl;
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

        std::vector<double> KineticEnergyij;
        dftUtils::readFile(KineticEnergyij, keFileName);
        d_KineticEnergyCorrectionTerm[atomicNumber] = KineticEnergyij;
      } //*it
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeRadialMultipoleData()
  {}

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


} // namespace dftfe
