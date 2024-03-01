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
// @author Phani Motamarri, Sambit Das
//

#include <deviceKernelsGeneric.h> //KFIX ME
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceTypeConfig.h>
#include <DeviceKernelLauncherConstants.h>
#include <DeviceBlasWrapper.h> //KFIX ME
#include <kohnShamDFTOperatorDevice.h>
#include <linearAlgebraOperations.h>
#include <linearAlgebraOperationsInternal.h>
#include <linearAlgebraOperationsDevice.h>
#include <vectorUtilities.h>
#include <dft.h>
#include <dftParameters.h>
#include <dftUtils.h>

namespace dftfe
{
  namespace
  {
    __global__ void
    copyFloatArrToDoubleArrLocallyOwned(const unsigned int  contiguousBlockSize,
                                        const unsigned int  numContiguousBlocks,
                                        const float *       floatArr,
                                        const unsigned int *locallyOwnedFlagArr,
                                        double *            doubleArr)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          unsigned int blockIndex = index / contiguousBlockSize;
          if (locallyOwnedFlagArr[blockIndex] == 1)
            doubleArr[index] = floatArr[index];
        }
    }

    __global__ void
    copyFloatArrToDoubleArrLocallyOwned(
      const unsigned int                      contiguousBlockSize,
      const unsigned int                      numContiguousBlocks,
      const dftfe::utils::deviceFloatComplex *floatArr,
      const unsigned int *                    locallyOwnedFlagArr,
      dftfe::utils::deviceDoubleComplex *     doubleArr)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          unsigned int blockIndex = index / contiguousBlockSize;
          if (locallyOwnedFlagArr[blockIndex] == 1)
            dftfe::utils::copyValue(doubleArr + index, floatArr[index]);
        }
    }



  } // namespace

  //
  // constructor
  //
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    kohnShamDFTOperatorDeviceClass(
      dftClass<FEOrder, FEOrderElectro, memorySpace> *_dftPtr,
      const MPI_Comm &                                mpi_comm_parent,
      const MPI_Comm &                                mpi_comm_domain)
    : dftPtr(_dftPtr)
    , d_kPointIndex(0)
    , d_numberNodesPerElement(_dftPtr->matrix_free_data.get_dofs_per_cell())
    , d_numberMacroCells(_dftPtr->matrix_free_data.n_cell_batches())
    , d_numLocallyOwnedCells(dftPtr->matrix_free_data.n_physical_cells())
    , d_numQuadPoints(
        dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId)
          .size())
    , d_isStiffnessMatrixExternalPotCorrComputed(false)
    , d_isMallocCalled(false)
    , d_mpiCommParent(mpi_comm_parent)
    , mpi_communicator(mpi_comm_domain)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm_domain))
    , this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
    , computing_timer(mpi_comm_domain,
                      pcout,
                      dealii::TimerOutput::never,
                      dealii::TimerOutput::wall_times)
    , operatorDFTDeviceClass(mpi_comm_domain,
                             _dftPtr->getMatrixFreeData(),
                             _dftPtr->constraintsNoneDataInfo,
                             _dftPtr->d_constraintsNoneDataInfoDevice)
  {}

  //
  // destructor
  //
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    ~kohnShamDFTOperatorDeviceClass()
  {
    if (d_isMallocCalled == true)
      d_ONCVnonLocalOperator->freeDeviceVectors();
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    createDeviceBlasHandle()
  {
    dftfe::utils::deviceBlasWrapper::create(&d_deviceBlasHandle);
#ifdef DFTFE_WTIH_DEVICE_CUDA
    if (dftPtr->d_dftParamsPtr->useTF32Device)
      dftfe::utils::deviceBlasWrapper::setMathMode(
        d_deviceBlasHandle, DEVICEBLAS_TF32_TENSOR_OP_MATH);
#endif
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    destroyDeviceBlasHandle()
  {
    dftfe::utils::deviceBlasWrapper::destroy(d_deviceBlasHandle);
  }


  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  const double *
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    getSqrtMassVec()
  {
    return d_sqrtMassVectorDevice.begin();
  }


  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  const double *
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    getInvSqrtMassVec()
  {
    return d_invSqrtMassVectorDevice.begin();
  }


  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  distributedCPUVec<dataTypes::number> &
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    getProjectorKetTimesVectorSingle()
  {
    // FIX ME with correct call from ONCV
    // return dftPtr->d_projectorKetTimesVectorPar[0];
  }


  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE> &
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    getShapeFunctionValues()
  {
    d_basisOperationsPtrDevice->reinit(0, 0, dftPtr->d_densityQuadratureId);
    return d_basisOperationsPtrDevice->shapeFunctionBasisData(true);
  }


  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE> &
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    getShapeFunctionValuesTransposed(const bool use2pPlusOneGLQuad)
  {
    d_basisOperationsPtrDevice->reinit(0,
                                       0,
                                       use2pPlusOneGLQuad ?
                                         dftPtr->d_gllQuadratureId :
                                         dftPtr->d_densityQuadratureId);
    return d_basisOperationsPtrDevice->shapeFunctionBasisData(false);
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE> &
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    getShapeFunctionValuesNLPTransposed()
  {
    d_basisOperationsPtrDevice->reinit(0, 0, dftPtr->d_nlpspQuadratureId);
    return d_basisOperationsPtrDevice->shapeFunctionBasisData(false);
  }


  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE> &
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    getShapeFunctionGradientValuesNLPTransposed()
  {
    return d_shapeFunctionGradientValueNLPTransposedDevice;
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE> &
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    getInverseJacobiansNLP()
  {
    return d_inverseJacobiansNLPDevice;
  }


  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  dftfe::utils::MemoryStorage<dealii::types::global_dof_index,
                              dftfe::utils::MemorySpace::DEVICE> &
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    getFlattenedArrayCellLocalProcIndexIdMap()
  {
    return d_flattenedArrayCellLocalProcIndexIdMapDevice;
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  dftfe::utils::MemoryStorage<dataTypes::number,
                              dftfe::utils::MemorySpace::DEVICE> &
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    getCellWaveFunctionMatrix()
  {
    return d_cellWaveFunctionMatrix;
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  distributedCPUVec<dataTypes::number> &
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    getParallelVecSingleComponent()
  {
    return d_parallelVecSingleComponent;
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  distributedDeviceVec<dataTypes::number> &
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    getParallelChebyBlockVectorDevice()
  {
    const unsigned int BVec =
      std::min(dftPtr->d_dftParamsPtr->chebyWfcBlockSize,
               dftPtr->d_numEigenValues);
    return d_basisOperationsPtrDevice->getMultiVector(BVec);
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  distributedDeviceVec<dataTypes::number> &
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    getParallelChebyBlockVector2Device()
  {
    const unsigned int BVec =
      std::min(dftPtr->d_dftParamsPtr->chebyWfcBlockSize,
               dftPtr->d_numEigenValues);
    return d_basisOperationsPtrDevice->getMultiVector(BVec, 1);
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  distributedDeviceVec<dataTypes::number> &
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    getParallelProjectorKetTimesBlockVectorDevice()
  {
    return d_parallelSphericalFnKetTimesBlockVectorDevice;
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  dftfe::utils::MemoryStorage<unsigned int, dftfe::utils::MemorySpace::DEVICE> &
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    getLocallyOwnedProcBoundaryNodesVectorDevice()
  {
    return d_locallyOwnedProcBoundaryNodesVectorDevice;
  }


  //
  // initialize kohnShamDFTOperatorDeviceClass object
  //
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::init()
  {
    computing_timer.enter_subsection("kohnShamDFTOperatorDeviceClass setup");

    d_basisOperationsPtrDevice = dftPtr->d_basisOperationsPtrDevice;
    d_basisOperationsPtrHost   = dftPtr->d_basisOperationsPtrHost;
    if (dftPtr->d_dftParamsPtr->isPseudopotential == true &&
        dftPtr->d_dftParamsPtr->pawPseudoPotential == false)
      {
        d_oncvClassPtr         = dftPtr->d_oncvClassPtr;
        d_ONCVnonLocalOperator = d_oncvClassPtr->getNonLocalOperator();
      }
    dftPtr->matrix_free_data.initialize_dof_vector(
      d_invSqrtMassVector, dftPtr->d_densityDofHandlerIndex);
    d_sqrtMassVector.reinit(d_invSqrtMassVector);



    //
    // compute mass vector
    //
    computeMassVector(dftPtr->dofHandler,
                      dftPtr->constraintsNone,
                      d_sqrtMassVector,
                      d_invSqrtMassVector);

    computing_timer.leave_subsection("kohnShamDFTOperatorDeviceClass setup");
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    resetExtPotHamFlag()
  {
    d_isStiffnessMatrixExternalPotCorrComputed = false;
  }


  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::reinit(
    const unsigned int numberWaveFunctions,
    bool               flag)
  {
    d_kpointCoordsVecDevice.resize(dftPtr->d_kPointCoordinates.size());
    d_kpointCoordsVecDevice.copyFrom(dftPtr->d_kPointCoordinates);

    std::vector<double> kpointSquareTimesHalfTemp(
      dftPtr->d_kPointWeights.size());
    for (unsigned int i = 0; i < dftPtr->d_kPointWeights.size(); ++i)
      {
        kpointSquareTimesHalfTemp[i] =
          0.5 * (dftPtr->d_kPointCoordinates[3 * i + 0] *
                   dftPtr->d_kPointCoordinates[3 * i + 0] +
                 dftPtr->d_kPointCoordinates[3 * i + 1] *
                   dftPtr->d_kPointCoordinates[3 * i + 1] +
                 dftPtr->d_kPointCoordinates[3 * i + 2] *
                   dftPtr->d_kPointCoordinates[3 * i + 2]);
      }
    d_kSquareTimesHalfVecDevice.resize(kpointSquareTimesHalfTemp.size());
    d_kSquareTimesHalfVecDevice.copyFrom(kpointSquareTimesHalfTemp);

    distributedCPUMultiVec<dataTypes::number> flattenedArray;
    if (flag)
      dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
        dftPtr->matrix_free_data.get_vector_partitioner(),
        numberWaveFunctions,
        flattenedArray);

    vectorTools::createDealiiVector<dataTypes::number>(
      dftPtr->matrix_free_data.get_vector_partitioner(
        dftPtr->d_densityDofHandlerIndex),
      1,
      d_parallelVecSingleComponent);

    std::size_t free_t, total_t;

    dftfe::utils::deviceMemGetInfo(&free_t, &total_t);
    if (dftPtr->d_dftParamsPtr->verbosity >= 2)
      pcout << "starting free mem on device: " << free_t
            << ", total mem on device: " << total_t << std::endl;

    const unsigned int BVec =
      std::min(dftPtr->d_dftParamsPtr->chebyWfcBlockSize, numberWaveFunctions);



    const unsigned int n_ghosts =
      dftPtr->matrix_free_data
        .get_vector_partitioner(dftPtr->d_densityDofHandlerIndex)
        ->n_ghost_indices();
    const unsigned int localSize =
      dftPtr->matrix_free_data
        .get_vector_partitioner(dftPtr->d_densityDofHandlerIndex)
        ->local_size();
    if (std::is_same<dataTypes::number, std::complex<double>>::value)
      {
        d_tempRealVec.resize(((localSize + n_ghosts) * BVec), 0.0);
        d_tempImagVec.resize(((localSize + n_ghosts) * BVec), 0.0);
      }

    dftfe::utils::MemoryStorage<unsigned int, dftfe::utils::MemorySpace::HOST>
      locallyOwnedProcBoundaryNodesVector(localSize, 0);

    const std::vector<std::pair<unsigned int, unsigned int>>
      &locallyOwnedProcBoundaryNodes =
        dftPtr->matrix_free_data
          .get_vector_partitioner(dftPtr->d_densityDofHandlerIndex)
          ->import_indices();

    for (unsigned int iset = 0; iset < locallyOwnedProcBoundaryNodes.size();
         ++iset)
      {
        const std::pair<unsigned int, unsigned int> &localIndices =
          locallyOwnedProcBoundaryNodes[iset];
        for (unsigned int inode = localIndices.first;
             inode < localIndices.second;
             ++inode)
          {
            locallyOwnedProcBoundaryNodesVector[inode] = 1;
          }
      }

    d_locallyOwnedProcBoundaryNodesVectorDevice.resize(localSize);


    d_locallyOwnedProcBoundaryNodesVectorDevice.copyFrom(
      locallyOwnedProcBoundaryNodesVector);

    vectorTools::computeCellLocalIndexSetMap(
      flattenedArray.getMPIPatternP2P(),
      dftPtr->matrix_free_data,
      dftPtr->d_densityDofHandlerIndex,
      numberWaveFunctions,
      d_flattenedArrayMacroCellLocalProcIndexIdMapFlattened,
      d_normalCellIdToMacroCellIdMap,
      d_macroCellIdToNormalCellIdMap,
      d_flattenedArrayCellLocalProcIndexIdMap);

    d_flattenedArrayCellLocalProcIndexIdMapDevice.resize(
      d_flattenedArrayCellLocalProcIndexIdMap.size());
    d_flattenedArrayCellLocalProcIndexIdMapDevice.copyFrom(
      d_flattenedArrayCellLocalProcIndexIdMap);


    const unsigned int totalLocallyOwnedCells =
      dftPtr->matrix_free_data.n_physical_cells();

    d_cellHamiltonianMatrixFlattenedDevice.resize(
      d_numLocallyOwnedCells * d_numberNodesPerElement *
        d_numberNodesPerElement * dftPtr->d_kPointWeights.size() *
        (1 + dftPtr->d_dftParamsPtr->spinPolarized),
      dataTypes::number(0.0));

    if (dftPtr->d_dftParamsPtr->isPseudopotential)
      d_cellHamiltonianMatrixExternalPotCorrFlattenedDevice.resize(
        d_numLocallyOwnedCells * d_numberNodesPerElement *
          d_numberNodesPerElement,
        0.0);
    else
      d_cellHamiltonianMatrixExternalPotCorrFlattenedDevice.resize(10, 0.0);

    d_cellWaveFunctionMatrix.resize(totalLocallyOwnedCells *
                                      d_numberNodesPerElement *
                                      numberWaveFunctions,
                                    0.0);

    d_cellHamMatrixTimesWaveMatrix.resize(totalLocallyOwnedCells *
                                            d_numberNodesPerElement *
                                            numberWaveFunctions,
                                          0.0);

    if (dftPtr->d_dftParamsPtr->isPseudopotential &&
        !dftPtr->d_dftParamsPtr->pawPseudoPotential)
      {
        if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
          {
            d_ONCVnonLocalOperator->initialiseFlattenedDataStructure(
              BVec, d_parallelSphericalFnKetTimesBlockVectorDevice);
            d_ONCVnonLocalOperator->initialiseCellWaveFunctionPointers(
              d_cellWaveFunctionMatrix);
            d_totalNonlocalElemsPseudo =
              d_ONCVnonLocalOperator
                ->getTotalNonLocalElementsInCurrentProcessor();
            d_isMallocCalled = true;
          }
      }

    dftfe::utils::deviceMemGetInfo(&free_t, &total_t);
    if (dftPtr->d_dftParamsPtr->verbosity >= 2)
      pcout << "free mem on device after reinit allocations: " << free_t
            << ", total mem on device: " << total_t << std::endl;
  }

  //
  // compute mass Vector
  //
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    computeMassVector(const dealii::DoFHandler<3> &            dofHandler,
                      const dealii::AffineConstraints<double> &constraintMatrix,
                      distributedCPUVec<double> &              sqrtMassVec,
                      distributedCPUVec<double> &              invSqrtMassVec)
  {
    computing_timer.enter_subsection(
      "kohnShamDFTOperatorDeviceClass Mass assembly");
    invSqrtMassVec = 0.0;
    sqrtMassVec    = 0.0;

    dealii::QGaussLobatto<3> quadrature(FEOrder + 1);
    dealii::FEValues<3>      fe_values(dofHandler.get_fe(),
                                  quadrature,
                                  dealii::update_values |
                                    dealii::update_JxW_values);
    const unsigned int     dofs_per_cell = (dofHandler.get_fe()).dofs_per_cell;
    const unsigned int     num_quad_points = quadrature.size();
    dealii::Vector<double> massVectorLocal(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(
      dofs_per_cell);


    //
    // parallel loop over all elements
    //
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = dofHandler.begin_active(),
      endc = dofHandler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          // compute values for the current element
          fe_values.reinit(cell);
          massVectorLocal = 0.0;
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
              massVectorLocal(i) += fe_values.shape_value(i, q_point) *
                                    fe_values.shape_value(i, q_point) *
                                    fe_values.JxW(q_point);

          cell->get_dof_indices(local_dof_indices);
          constraintMatrix.distribute_local_to_global(massVectorLocal,
                                                      local_dof_indices,
                                                      invSqrtMassVec);
        }

    invSqrtMassVec.compress(dealii::VectorOperation::add);


    for (dealii::types::global_dof_index i = 0; i < invSqrtMassVec.size(); ++i)
      if (invSqrtMassVec.in_local_range(i) &&
          !constraintMatrix.is_constrained(i))
        {
          if (std::abs(invSqrtMassVec(i)) > 1.0e-15)
            {
              sqrtMassVec(i)    = std::sqrt(invSqrtMassVec(i));
              invSqrtMassVec(i) = 1.0 / std::sqrt(invSqrtMassVec(i));
            }
          AssertThrow(
            !std::isnan(invSqrtMassVec(i)),
            dealii::ExcMessage(
              "Value of inverse square root of mass matrix on the unconstrained node is undefined"));
        }

    invSqrtMassVec.compress(dealii::VectorOperation::insert);
    sqrtMassVec.compress(dealii::VectorOperation::insert);

    invSqrtMassVec.update_ghost_values();
    sqrtMassVec.update_ghost_values();

    const unsigned int numberLocalDofs = invSqrtMassVec.local_size();
    const unsigned int numberGhostDofs =
      invSqrtMassVec.get_partitioner()->n_ghost_indices();
    d_invSqrtMassVectorDevice.clear();
    d_sqrtMassVectorDevice.clear();
    d_invSqrtMassVectorDevice.resize(numberLocalDofs + numberGhostDofs);
    d_sqrtMassVectorDevice.resize(numberLocalDofs + numberGhostDofs);

    dftfe::utils::deviceMemcpyH2D(d_invSqrtMassVectorDevice.begin(),
                                  invSqrtMassVec.begin(),
                                  (numberLocalDofs + numberGhostDofs) *
                                    sizeof(double));

    dftfe::utils::deviceMemcpyH2D(d_sqrtMassVectorDevice.begin(),
                                  sqrtMassVec.begin(),
                                  (numberLocalDofs + numberGhostDofs) *
                                    sizeof(double));

    computing_timer.leave_subsection(
      "kohnShamDFTOperatorDeviceClass Mass assembly");
  }


  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    reinitkPointSpinIndex(const unsigned int kPointIndex,
                          const unsigned int spinIndex)
  {
    d_kPointIndex = kPointIndex;
    d_spinIndex   = spinIndex;



    if (dftPtr->d_dftParamsPtr->isPseudopotential &&
        !dftPtr->d_dftParamsPtr->pawPseudoPotential)
      {
        if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
          d_ONCVnonLocalOperator->initialiseOperatorActionOnX(d_kPointIndex);
      }
  }


  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    computeVEff(
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &rhoValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &phiValues,
      const std::map<dealii::CellId, std::vector<double>>
        &externalPotCorrValues,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
      const unsigned int externalPotCorrQuadratureId)
  {
    d_basisOperationsPtrHost->reinit(0, 0, dftPtr->d_densityQuadratureId);
    const unsigned int totalLocallyOwnedCells =
      dftPtr->matrix_free_data.n_physical_cells();

    const dealii::Quadrature<3> &quadrature_formula =
      dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId);
    const unsigned int numberQuadraturePoints = quadrature_formula.size();

    d_vEff.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);
    d_vEffJxW.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);
    typename dealii::DoFHandler<3>::active_cell_iterator cellPtr =
      dftPtr->matrix_free_data.get_dof_handler().begin_active();
    typename dealii::DoFHandler<3>::active_cell_iterator endcPtr =
      dftPtr->matrix_free_data.get_dof_handler().end();
    unsigned int iElemCount = 0;

    std::vector<double> exchangePotentialVal(numberQuadraturePoints);
    std::vector<double> corrPotentialVal(numberQuadraturePoints);
    std::vector<double> densityValue(numberQuadraturePoints);

    for (; cellPtr != endcPtr; ++cellPtr)
      if (cellPtr->is_locally_owned())
        {
          std::memcpy(densityValue.data(),
                      rhoValues[0].data() + iElemCount * numberQuadraturePoints,
                      numberQuadraturePoints * sizeof(double));

          if (dftPtr->d_dftParamsPtr->nonLinearCoreCorrection)
            {
              std::transform(densityValue.data(),
                             densityValue.data() + numberQuadraturePoints,
                             rhoCoreValues.find(cellPtr->id())->second.data(),
                             densityValue.data(),
                             std::plus<>{});
            }

          const double *tempPhi =
            phiValues.data() + iElemCount * numberQuadraturePoints;

          std::map<rhoDataAttributes, const std::vector<double> *> rhoData;

          std::map<VeffOutputDataAttributes, std::vector<double> *>
            outputDerExchangeEnergy;
          std::map<VeffOutputDataAttributes, std::vector<double> *>
            outputDerCorrEnergy;

          rhoData[rhoDataAttributes::values] = &densityValue;

          outputDerExchangeEnergy
            [VeffOutputDataAttributes::derEnergyWithDensity] =
              &exchangePotentialVal;

          outputDerCorrEnergy[VeffOutputDataAttributes::derEnergyWithDensity] =
            &corrPotentialVal;

          dftPtr->d_excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
            numberQuadraturePoints,
            rhoData,
            outputDerExchangeEnergy,
            outputDerCorrEnergy);

          for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
            {
              d_vEff[iElemCount * numberQuadraturePoints + q] =
                tempPhi[q] + exchangePotentialVal[q] + corrPotentialVal[q];

              d_vEffJxW[iElemCount * numberQuadraturePoints + q] =
                d_vEff[iElemCount * numberQuadraturePoints + q] *
                d_basisOperationsPtrHost
                  ->JxWBasisData()[iElemCount * numberQuadraturePoints + q];
            }

          iElemCount++;
        }

    d_vEffJxWDevice.resize(d_vEffJxW.size());
    d_vEffJxWDevice.copyFrom(d_vEffJxW);
    if ((dftPtr->d_dftParamsPtr->isPseudopotential ||
         dftPtr->d_dftParamsPtr->smearedNuclearCharges) &&
        !d_isStiffnessMatrixExternalPotCorrComputed)
      computeVEffExternalPotCorr(externalPotCorrValues,
                                 externalPotCorrQuadratureId);
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    computeVEff(
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &rhoValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &gradRhoValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &phiValues,
      const std::map<dealii::CellId, std::vector<double>>
        &externalPotCorrValues,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
      const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
      const unsigned int externalPotCorrQuadratureId)
  {
    d_basisOperationsPtrHost->reinit(0, 0, dftPtr->d_densityQuadratureId);
    const unsigned int totalLocallyOwnedCells =
      dftPtr->matrix_free_data.n_physical_cells();

    const dealii::Quadrature<3> &quadrature_formula =
      dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId);
    const unsigned int numberQuadraturePoints = quadrature_formula.size();


    d_vEff.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);
    d_vEffJxW.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);
    d_derExcWithSigmaTimesGradRhoJxW.resize(totalLocallyOwnedCells *
                                              numberQuadraturePoints * 3,
                                            0.0);

    typename dealii::DoFHandler<3>::active_cell_iterator cellPtr =
      dftPtr->matrix_free_data.get_dof_handler().begin_active();
    typename dealii::DoFHandler<3>::active_cell_iterator endcPtr =
      dftPtr->matrix_free_data.get_dof_handler().end();
    unsigned int iElemCount = 0;

    std::vector<double> sigmaValue(numberQuadraturePoints);
    std::vector<double> derExchEnergyWithSigmaVal(numberQuadraturePoints);
    std::vector<double> derCorrEnergyWithSigmaVal(numberQuadraturePoints);
    std::vector<double> derExchEnergyWithDensityVal(numberQuadraturePoints);
    std::vector<double> derCorrEnergyWithDensityVal(numberQuadraturePoints);
    std::vector<double> densityValue(numberQuadraturePoints);
    std::vector<double> gradDensityValue(3 * numberQuadraturePoints);

    for (; cellPtr != endcPtr; ++cellPtr)
      if (cellPtr->is_locally_owned())
        {
          std::memcpy(densityValue.data(),
                      rhoValues[0].data() + iElemCount * numberQuadraturePoints,
                      numberQuadraturePoints * sizeof(double));
          std::memcpy(gradDensityValue.data(),
                      gradRhoValues[0].data() +
                        iElemCount * numberQuadraturePoints * 3,
                      3 * numberQuadraturePoints * sizeof(double));
          if (dftPtr->d_dftParamsPtr->nonLinearCoreCorrection)
            {
              std::transform(densityValue.data(),
                             densityValue.data() + numberQuadraturePoints,
                             rhoCoreValues.find(cellPtr->id())->second.data(),
                             densityValue.data(),
                             std::plus<>{});
              std::transform(
                gradDensityValue.data(),
                gradDensityValue.data() + 3 * numberQuadraturePoints,
                gradRhoCoreValues.find(cellPtr->id())->second.data(),
                gradDensityValue.data(),
                std::plus<>{});
            }

          const double *tempPhi =
            phiValues.data() + iElemCount * numberQuadraturePoints;

          for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
            {
              const double gradRhoX = gradDensityValue[3 * q + 0];
              const double gradRhoY = gradDensityValue[3 * q + 1];
              const double gradRhoZ = gradDensityValue[3 * q + 2];
              sigmaValue[q] =
                gradRhoX * gradRhoX + gradRhoY * gradRhoY + gradRhoZ * gradRhoZ;
            }

          std::map<rhoDataAttributes, const std::vector<double> *> rhoData;

          std::map<VeffOutputDataAttributes, std::vector<double> *>
            outputDerExchangeEnergy;
          std::map<VeffOutputDataAttributes, std::vector<double> *>
            outputDerCorrEnergy;


          rhoData[rhoDataAttributes::values]         = &densityValue;
          rhoData[rhoDataAttributes::sigmaGradValue] = &sigmaValue;

          outputDerExchangeEnergy
            [VeffOutputDataAttributes::derEnergyWithDensity] =
              &derExchEnergyWithDensityVal;
          outputDerExchangeEnergy
            [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
              &derExchEnergyWithSigmaVal;

          outputDerCorrEnergy[VeffOutputDataAttributes::derEnergyWithDensity] =
            &derCorrEnergyWithDensityVal;
          outputDerCorrEnergy
            [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
              &derCorrEnergyWithSigmaVal;

          dftPtr->d_excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
            numberQuadraturePoints,
            rhoData,
            outputDerExchangeEnergy,
            outputDerCorrEnergy);


          for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
            {
              const double jxw =
                d_basisOperationsPtrHost
                  ->JxWBasisData()[iElemCount * numberQuadraturePoints + q];
              const double gradRhoX = gradDensityValue[3 * q + 0];
              const double gradRhoY = gradDensityValue[3 * q + 1];
              const double gradRhoZ = gradDensityValue[3 * q + 2];
              const double term =
                derExchEnergyWithSigmaVal[q] + derCorrEnergyWithSigmaVal[q];
              d_derExcWithSigmaTimesGradRhoJxW[iElemCount *
                                                 numberQuadraturePoints * 3 +
                                               3 * q] = term * gradRhoX * jxw;
              d_derExcWithSigmaTimesGradRhoJxW[iElemCount *
                                                 numberQuadraturePoints * 3 +
                                               3 * q + 1] =
                term * gradRhoY * jxw;
              d_derExcWithSigmaTimesGradRhoJxW[iElemCount *
                                                 numberQuadraturePoints * 3 +
                                               3 * q + 2] =
                term * gradRhoZ * jxw;
            }

          for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
            {
              d_vEff[iElemCount * numberQuadraturePoints + q] =
                tempPhi[q] + derExchEnergyWithDensityVal[q] +
                derCorrEnergyWithDensityVal[q];

              d_vEffJxW[iElemCount * numberQuadraturePoints + q] =
                d_vEff[iElemCount * numberQuadraturePoints + q] *
                d_basisOperationsPtrHost
                  ->JxWBasisData()[iElemCount * numberQuadraturePoints + q];
            }

          iElemCount++;
        }

    d_vEffJxWDevice.resize(d_vEffJxW.size());
    d_vEffJxWDevice.copyFrom(d_vEffJxW);

    d_derExcWithSigmaTimesGradRhoJxWDevice.resize(
      d_derExcWithSigmaTimesGradRhoJxW.size());
    d_derExcWithSigmaTimesGradRhoJxWDevice.copyFrom(
      d_derExcWithSigmaTimesGradRhoJxW);

    if ((dftPtr->d_dftParamsPtr->isPseudopotential ||
         dftPtr->d_dftParamsPtr->smearedNuclearCharges) &&
        !d_isStiffnessMatrixExternalPotCorrComputed)
      computeVEffExternalPotCorr(externalPotCorrValues,
                                 externalPotCorrQuadratureId);
  }


  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    computeVEffSpinPolarized(
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &rhoValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &                phiValues,
      const unsigned int spinIndex,
      const std::map<dealii::CellId, std::vector<double>>
        &externalPotCorrValues,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
      const unsigned int externalPotCorrQuadratureId)
  {
    d_basisOperationsPtrHost->reinit(0, 0, dftPtr->d_densityQuadratureId);
    const unsigned int totalLocallyOwnedCells =
      dftPtr->matrix_free_data.n_physical_cells();

    const dealii::Quadrature<3> &quadrature_formula =
      dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId);
    const unsigned int numberQuadraturePoints = quadrature_formula.size();

    d_vEff.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);
    d_vEffJxW.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);
    typename dealii::DoFHandler<3>::active_cell_iterator cellPtr =
      dftPtr->matrix_free_data.get_dof_handler().begin_active();
    typename dealii::DoFHandler<3>::active_cell_iterator endcPtr =
      dftPtr->matrix_free_data.get_dof_handler().end();
    unsigned int iElemCount = 0;

    std::vector<double> exchangePotentialVal(2 * numberQuadraturePoints);
    std::vector<double> corrPotentialVal(2 * numberQuadraturePoints);
    std::vector<double> densityValue(2 * numberQuadraturePoints);
    for (; cellPtr != endcPtr; ++cellPtr)
      if (cellPtr->is_locally_owned())
        {
          const double *cellRhoValues =
            rhoValues[0].data() + iElemCount * numberQuadraturePoints;
          const double *cellMagValues =
            rhoValues[1].data() + iElemCount * numberQuadraturePoints;
          for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
            {
              const double rhoByTwo   = cellRhoValues[q] / 2.0;
              const double magByTwo   = cellMagValues[q] / 2.0;
              densityValue[2 * q]     = rhoByTwo + magByTwo;
              densityValue[2 * q + 1] = rhoByTwo - magByTwo;
            }
          const double *tempPhi =
            phiValues.data() + iElemCount * numberQuadraturePoints;

          if (dftPtr->d_dftParamsPtr->nonLinearCoreCorrection)
            {
              const std::vector<double> &temp2 =
                rhoCoreValues.find(cellPtr->id())->second;
              for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                {
                  densityValue[2 * q] += temp2[q] / 2.0;
                  densityValue[2 * q + 1] += temp2[q] / 2.0;
                }
            }

          std::map<rhoDataAttributes, const std::vector<double> *> rhoData;

          std::map<VeffOutputDataAttributes, std::vector<double> *>
            outputDerExchangeEnergy;
          std::map<VeffOutputDataAttributes, std::vector<double> *>
            outputDerCorrEnergy;

          rhoData[rhoDataAttributes::values] = &densityValue;

          outputDerExchangeEnergy
            [VeffOutputDataAttributes::derEnergyWithDensity] =
              &exchangePotentialVal;

          outputDerCorrEnergy[VeffOutputDataAttributes::derEnergyWithDensity] =
            &corrPotentialVal;

          dftPtr->d_excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
            numberQuadraturePoints,
            rhoData,
            outputDerExchangeEnergy,
            outputDerCorrEnergy);


          for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
            {
              d_vEff[iElemCount * numberQuadraturePoints + q] =
                tempPhi[q] + exchangePotentialVal[2 * q + spinIndex] +
                corrPotentialVal[2 * q + spinIndex];

              d_vEffJxW[iElemCount * numberQuadraturePoints + q] =
                d_vEff[iElemCount * numberQuadraturePoints + q] *
                d_basisOperationsPtrHost
                  ->JxWBasisData()[iElemCount * numberQuadraturePoints + q];
            }

          iElemCount++;
        }

    d_vEffJxWDevice.resize(d_vEffJxW.size());
    d_vEffJxWDevice.copyFrom(d_vEffJxW);


    if ((dftPtr->d_dftParamsPtr->isPseudopotential ||
         dftPtr->d_dftParamsPtr->smearedNuclearCharges) &&
        !d_isStiffnessMatrixExternalPotCorrComputed)
      computeVEffExternalPotCorr(externalPotCorrValues,
                                 externalPotCorrQuadratureId);
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    computeVEffSpinPolarized(
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &rhoValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &gradRhoValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &                phiValues,
      const unsigned int spinIndex,
      const std::map<dealii::CellId, std::vector<double>>
        &externalPotCorrValues,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
      const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
      const unsigned int externalPotCorrQuadratureId)
  {
    d_basisOperationsPtrHost->reinit(0, 0, dftPtr->d_densityQuadratureId);
    const unsigned int totalLocallyOwnedCells =
      dftPtr->matrix_free_data.n_physical_cells();

    const dealii::Quadrature<3> &quadrature_formula =
      dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId);
    const unsigned int numberQuadraturePoints = quadrature_formula.size();

    d_vEff.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);
    d_vEffJxW.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);
    d_derExcWithSigmaTimesGradRhoJxW.resize(totalLocallyOwnedCells *
                                              numberQuadraturePoints * 3,
                                            0.0);

    typename dealii::DoFHandler<3>::active_cell_iterator cellPtr =
      dftPtr->matrix_free_data.get_dof_handler().begin_active();
    typename dealii::DoFHandler<3>::active_cell_iterator endcPtr =
      dftPtr->matrix_free_data.get_dof_handler().end();
    unsigned int iElemCount = 0;

    std::vector<double> sigmaValue(3 * numberQuadraturePoints);
    std::vector<double> derExchEnergyWithSigmaVal(3 * numberQuadraturePoints);
    std::vector<double> derCorrEnergyWithSigmaVal(3 * numberQuadraturePoints);
    std::vector<double> derExchEnergyWithDensityVal(2 * numberQuadraturePoints);
    std::vector<double> derCorrEnergyWithDensityVal(2 * numberQuadraturePoints);
    std::vector<double> densityValue(2 * numberQuadraturePoints);
    std::vector<double> gradDensityValue(6 * numberQuadraturePoints);

    for (; cellPtr != endcPtr; ++cellPtr)
      if (cellPtr->is_locally_owned())
        {
          const double *cellRhoValues =
            rhoValues[0].data() + iElemCount * numberQuadraturePoints;
          const double *cellMagValues =
            rhoValues[1].data() + iElemCount * numberQuadraturePoints;
          for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
            {
              const double rhoByTwo   = cellRhoValues[q] / 2.0;
              const double magByTwo   = cellMagValues[q] / 2.0;
              densityValue[2 * q]     = rhoByTwo + magByTwo;
              densityValue[2 * q + 1] = rhoByTwo - magByTwo;
            }
          const double *cellGradRhoValues =
            gradRhoValues[0].data() + 3 * iElemCount * numberQuadraturePoints;
          const double *cellGradMagValues =
            gradRhoValues[1].data() + 3 * iElemCount * numberQuadraturePoints;
          for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
            for (unsigned int iDim = 0; iDim < 3; ++iDim)
              {
                const double gradRhoByTwo =
                  cellGradRhoValues[3 * q + iDim] / 2.0;
                const double gradMagByTwo =
                  cellGradMagValues[3 * q + iDim] / 2.0;
                gradDensityValue[6 * q + iDim] = gradRhoByTwo + gradMagByTwo;
                gradDensityValue[6 * q + 3 + iDim] =
                  gradRhoByTwo - gradMagByTwo;
              }
          const double *tempPhi =
            phiValues.data() + iElemCount * numberQuadraturePoints;


          if (dftPtr->d_dftParamsPtr->nonLinearCoreCorrection)
            {
              const std::vector<double> &temp2 =
                rhoCoreValues.find(cellPtr->id())->second;
              const std::vector<double> &temp3 =
                gradRhoCoreValues.find(cellPtr->id())->second;
              for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                {
                  densityValue[2 * q] += temp2[q] / 2.0;
                  densityValue[2 * q + 1] += temp2[q] / 2.0;
                  gradDensityValue[6 * q + 0] += temp3[3 * q + 0] / 2.0;
                  gradDensityValue[6 * q + 1] += temp3[3 * q + 1] / 2.0;
                  gradDensityValue[6 * q + 2] += temp3[3 * q + 2] / 2.0;
                  gradDensityValue[6 * q + 3] += temp3[3 * q + 0] / 2.0;
                  gradDensityValue[6 * q + 4] += temp3[3 * q + 1] / 2.0;
                  gradDensityValue[6 * q + 5] += temp3[3 * q + 2] / 2.0;
                }
            }

          for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
            {
              double gradRhoX1 = gradDensityValue[6 * q + 0];
              double gradRhoY1 = gradDensityValue[6 * q + 1];
              double gradRhoZ1 = gradDensityValue[6 * q + 2];
              double gradRhoX2 = gradDensityValue[6 * q + 3];
              double gradRhoY2 = gradDensityValue[6 * q + 4];
              double gradRhoZ2 = gradDensityValue[6 * q + 5];
              //
              sigmaValue[3 * q + 0] = gradRhoX1 * gradRhoX1 +
                                      gradRhoY1 * gradRhoY1 +
                                      gradRhoZ1 * gradRhoZ1;
              sigmaValue[3 * q + 1] = gradRhoX1 * gradRhoX2 +
                                      gradRhoY1 * gradRhoY2 +
                                      gradRhoZ1 * gradRhoZ2;
              sigmaValue[3 * q + 2] = gradRhoX2 * gradRhoX2 +
                                      gradRhoY2 * gradRhoY2 +
                                      gradRhoZ2 * gradRhoZ2;
            }

          std::map<rhoDataAttributes, const std::vector<double> *> rhoData;

          std::map<VeffOutputDataAttributes, std::vector<double> *>
            outputDerExchangeEnergy;
          std::map<VeffOutputDataAttributes, std::vector<double> *>
            outputDerCorrEnergy;


          rhoData[rhoDataAttributes::values]         = &densityValue;
          rhoData[rhoDataAttributes::sigmaGradValue] = &sigmaValue;

          outputDerExchangeEnergy
            [VeffOutputDataAttributes::derEnergyWithDensity] =
              &derExchEnergyWithDensityVal;
          outputDerExchangeEnergy
            [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
              &derExchEnergyWithSigmaVal;

          outputDerCorrEnergy[VeffOutputDataAttributes::derEnergyWithDensity] =
            &derCorrEnergyWithDensityVal;
          outputDerCorrEnergy
            [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
              &derCorrEnergyWithSigmaVal;

          dftPtr->d_excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
            numberQuadraturePoints,
            rhoData,
            outputDerExchangeEnergy,
            outputDerCorrEnergy);

          for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
            {
              const double jxw =
                d_basisOperationsPtrHost
                  ->JxWBasisData()[iElemCount * numberQuadraturePoints + q];
              const double gradRhoX =
                gradDensityValue[6 * q + 0 + 3 * spinIndex];
              const double gradRhoY =
                gradDensityValue[6 * q + 1 + 3 * spinIndex];
              const double gradRhoZ =
                gradDensityValue[6 * q + 2 + 3 * spinIndex];
              const double gradRhoOtherX =
                gradDensityValue[6 * q + 0 + 3 * (1 - spinIndex)];
              const double gradRhoOtherY =
                gradDensityValue[6 * q + 1 + 3 * (1 - spinIndex)];
              const double gradRhoOtherZ =
                gradDensityValue[6 * q + 2 + 3 * (1 - spinIndex)];
              const double term =
                derExchEnergyWithSigmaVal[3 * q + 2 * spinIndex] +
                derCorrEnergyWithSigmaVal[3 * q + 2 * spinIndex];
              const double termOff = derExchEnergyWithSigmaVal[3 * q + 1] +
                                     derCorrEnergyWithSigmaVal[3 * q + 1];
              d_derExcWithSigmaTimesGradRhoJxW[iElemCount *
                                                 numberQuadraturePoints * 3 +
                                               3 * q] =
                (term * gradRhoX + 0.5 * termOff * gradRhoOtherX) * jxw;
              d_derExcWithSigmaTimesGradRhoJxW[iElemCount *
                                                 numberQuadraturePoints * 3 +
                                               3 * q + 1] =
                (term * gradRhoY + 0.5 * termOff * gradRhoOtherY) * jxw;
              d_derExcWithSigmaTimesGradRhoJxW[iElemCount *
                                                 numberQuadraturePoints * 3 +
                                               3 * q + 2] =
                (term * gradRhoZ + 0.5 * termOff * gradRhoOtherZ) * jxw;
            }

          for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
            {
              d_vEff[iElemCount * numberQuadraturePoints + q] =
                tempPhi[q] + derExchEnergyWithDensityVal[2 * q + spinIndex] +
                derCorrEnergyWithDensityVal[2 * q + spinIndex];

              d_vEffJxW[iElemCount * numberQuadraturePoints + q] =
                d_vEff[iElemCount * numberQuadraturePoints + q] *
                d_basisOperationsPtrHost
                  ->JxWBasisData()[iElemCount * numberQuadraturePoints + q];
            }

          iElemCount++;
        }

    d_vEffJxWDevice.resize(d_vEffJxW.size());
    d_vEffJxWDevice.copyFrom(d_vEffJxW);

    d_derExcWithSigmaTimesGradRhoJxWDevice.resize(
      d_derExcWithSigmaTimesGradRhoJxW.size());
    d_derExcWithSigmaTimesGradRhoJxWDevice.copyFrom(
      d_derExcWithSigmaTimesGradRhoJxW);

    if ((dftPtr->d_dftParamsPtr->isPseudopotential ||
         dftPtr->d_dftParamsPtr->smearedNuclearCharges) &&
        !d_isStiffnessMatrixExternalPotCorrComputed)
      computeVEffExternalPotCorr(externalPotCorrValues,
                                 externalPotCorrQuadratureId);
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    computeVEffExternalPotCorr(
      const std::map<dealii::CellId, std::vector<double>>
        &                externalPotCorrValues,
      const unsigned int externalPotCorrQuadratureId)
  {
    d_externalPotCorrQuadratureId = externalPotCorrQuadratureId;
    const unsigned int numberPhysicalCells =
      dftPtr->matrix_free_data.n_physical_cells();
    const int numberQuadraturePoints =
      dftPtr->matrix_free_data.get_quadrature(externalPotCorrQuadratureId)
        .size();
    dealii::FEValues<3> feValues(
      dftPtr->matrix_free_data.get_dof_handler().get_fe(),
      dftPtr->matrix_free_data.get_quadrature(externalPotCorrQuadratureId),
      dealii::update_JxW_values);
    d_vEffExternalPotCorrJxW.resize(numberPhysicalCells *
                                    numberQuadraturePoints);


    typename dealii::DoFHandler<3>::active_cell_iterator cellPtr =
      dftPtr->matrix_free_data.get_dof_handler().begin_active();
    typename dealii::DoFHandler<3>::active_cell_iterator endcPtr =
      dftPtr->matrix_free_data.get_dof_handler().end();

    unsigned int iElem = 0;
    for (; cellPtr != endcPtr; ++cellPtr)
      if (cellPtr->is_locally_owned())
        {
          feValues.reinit(cellPtr);
          const std::vector<double> &temp =
            externalPotCorrValues.find(cellPtr->id())->second;
          for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
            d_vEffExternalPotCorrJxW[iElem * numberQuadraturePoints + q] =
              temp[q] * feValues.JxW(q);

          iElem++;
        }

    d_vEffExternalPotCorrJxWDevice.resize(d_vEffExternalPotCorrJxW.size());
    d_vEffExternalPotCorrJxWDevice.copyFrom(d_vEffExternalPotCorrJxW);
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    computeVEffPrime(
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &rhoValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &rhoPrimeValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &                                                  phiPrimeValues,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues)
  {
    d_basisOperationsPtrHost->reinit(0, 0, dftPtr->d_densityQuadratureId);
    const unsigned int totalLocallyOwnedCells =
      dftPtr->matrix_free_data.n_physical_cells();
    const dealii::Quadrature<3> &quadrature_formula =
      dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId);

    const unsigned int numberQuadraturePoints = quadrature_formula.size();

    d_vEffJxW.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);

    std::vector<double> densityValue(numberQuadraturePoints);
    std::vector<double> densityPrimeValue(numberQuadraturePoints);
    std::vector<double> phiPrimeValue(numberQuadraturePoints);
    std::vector<double> der2ExchEnergyWithDensityVal(numberQuadraturePoints);
    std::vector<double> der2CorrEnergyWithDensityVal(numberQuadraturePoints);

    typename dealii::DoFHandler<3>::active_cell_iterator
      cellPtr = dftPtr->matrix_free_data
                  .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
                  .begin_active(),
      endcellPtr = dftPtr->matrix_free_data
                     .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
                     .end();

    //
    // loop over cell block
    //
    unsigned int iElemCount = 0;
    for (; cellPtr != endcellPtr; ++cellPtr)
      {
        if (cellPtr->is_locally_owned())
          {
            const auto &tempDensityTotalValues = rhoValues[0];
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              densityValue[q] =
                tempDensityTotalValues[iElemCount * numberQuadraturePoints + q];

            const auto &tempDensityTotalPrimeValues = rhoPrimeValues[0];
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              densityPrimeValue[q] =
                tempDensityTotalPrimeValues[iElemCount *
                                              numberQuadraturePoints +
                                            q];


            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              phiPrimeValue[q] =
                phiPrimeValues[iElemCount * numberQuadraturePoints + q];

            if (dftPtr->d_dftParamsPtr->nonLinearCoreCorrection)
              {
                const std::vector<double> &temp2 =
                  rhoCoreValues.find(cellPtr->id())->second;
                for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                  {
                    densityValue[q] += temp2[q];
                  }
              }

            std::map<rhoDataAttributes, const std::vector<double> *> rhoData;

            std::map<fxcOutputDataAttributes, std::vector<double> *>
              outputDer2ExchangeEnergy;
            std::map<fxcOutputDataAttributes, std::vector<double> *>
              outputDer2CorrEnergy;


            rhoData[rhoDataAttributes::values] = &densityValue;

            outputDer2ExchangeEnergy
              [fxcOutputDataAttributes::der2EnergyWithDensity] =
                &der2ExchEnergyWithDensityVal;

            outputDer2CorrEnergy
              [fxcOutputDataAttributes::der2EnergyWithDensity] =
                &der2CorrEnergyWithDensityVal;


            dftPtr->d_excManagerPtr->getExcDensityObj()->computeDensityBasedFxc(
              numberQuadraturePoints,
              rhoData,
              outputDer2ExchangeEnergy,
              outputDer2CorrEnergy);



            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                d_vEffJxW[iElemCount * numberQuadraturePoints + q] =
                  (phiPrimeValue[q] + (der2ExchEnergyWithDensityVal[q] +
                                       der2CorrEnergyWithDensityVal[q]) *
                                        densityPrimeValue[q]) *
                  d_basisOperationsPtrHost
                    ->JxWBasisData()[iElemCount * numberQuadraturePoints + q];
              }

            iElemCount++;
          } // if cellPtr->is_locally_owned() loop

      } // cell loop
    d_vEffJxWDevice.resize(d_vEffJxW.size());
    d_vEffJxWDevice.copyFrom(d_vEffJxW);
  }


  // Fourth order stencil finite difference stencil used
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    computeVEffPrimeSpinPolarized(
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &rhoValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &rhoPrimeValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &                                                  phiPrimeValues,
      const unsigned int                                   spinIndex,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues)
  {
    d_basisOperationsPtrHost->reinit(0, 0, dftPtr->d_densityQuadratureId);
    const unsigned int totalLocallyOwnedCells =
      dftPtr->matrix_free_data.n_physical_cells();
    const dealii::Quadrature<3> &quadrature_formula =
      dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId);
    const unsigned int numberQuadraturePoints = quadrature_formula.size();

    d_vEffJxW.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);

    std::vector<double> densityValue(2 * numberQuadraturePoints);
    std::vector<double> densityPrimeValue(2 * numberQuadraturePoints);
    std::vector<double> phiPrimeValue(numberQuadraturePoints);
    std::vector<double> derExchEnergyWithDensityVal(2 * numberQuadraturePoints);
    std::vector<double> derCorrEnergyWithDensityVal(2 * numberQuadraturePoints);

    typename dealii::DoFHandler<3>::active_cell_iterator
      cellPtr = dftPtr->matrix_free_data
                  .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
                  .begin_active(),
      endcellPtr = dftPtr->matrix_free_data
                     .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
                     .end();
    const double lambda = 1e-2;

    //
    // loop over cell block
    //
    unsigned int iElemCount = 0;
    for (; cellPtr != endcellPtr; ++cellPtr)
      {
        if (cellPtr->is_locally_owned())
          {
            const auto &tempDensityTotalValues = rhoValues[0];
            const auto &tempDensityMagValues   = rhoValues[1];
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[2 * q + 0] =
                  0.5 *
                  (tempDensityTotalValues[iElemCount * numberQuadraturePoints +
                                          q] +
                   tempDensityMagValues[iElemCount * numberQuadraturePoints +
                                        q]);
                densityValue[2 * q + 1] =
                  0.5 *
                  (tempDensityTotalValues[iElemCount * numberQuadraturePoints +
                                          q] -
                   tempDensityMagValues[iElemCount * numberQuadraturePoints +
                                        q]);
              }

            if (dftPtr->d_dftParamsPtr->nonLinearCoreCorrection)
              {
                const std::vector<double> &temp2 =
                  rhoCoreValues.find(cellPtr->id())->second;

                for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                  {
                    densityValue[2 * q] += temp2[q] / 2.0;
                    densityValue[2 * q + 1] += temp2[q] / 2.0;
                  }
              }


            const auto &tempDensityTotalPrimeValues = rhoPrimeValues[0];
            const auto &tempDensityMagPrimeValues   = rhoPrimeValues[1];
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityPrimeValue[2 * q + 0] =
                  0.5 * (tempDensityTotalPrimeValues[iElemCount *
                                                       numberQuadraturePoints +
                                                     q] +
                         tempDensityMagPrimeValues[iElemCount *
                                                     numberQuadraturePoints +
                                                   q]);
                densityPrimeValue[2 * q + 1] =
                  0.5 * (tempDensityTotalPrimeValues[iElemCount *
                                                       numberQuadraturePoints +
                                                     q] -
                         tempDensityMagPrimeValues[iElemCount *
                                                     numberQuadraturePoints +
                                                   q]);
              }


            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[2 * q] += 2.0 * lambda * densityPrimeValue[2 * q];
                densityValue[2 * q + 1] +=
                  2.0 * lambda * densityPrimeValue[2 * q + 1];
              }

            std::map<rhoDataAttributes, const std::vector<double> *> rhoData;

            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerExchangeEnergy;
            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerCorrEnergy;

            rhoData[rhoDataAttributes::values] = &densityValue;

            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derExchEnergyWithDensityVal;

            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derCorrEnergyWithDensityVal;

            dftPtr->d_excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
              numberQuadraturePoints,
              rhoData,
              outputDerExchangeEnergy,
              outputDerCorrEnergy);


            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                d_vEffJxW[iElemCount * numberQuadraturePoints + q] =
                  -(derExchEnergyWithDensityVal[2 * q + spinIndex] +
                    derCorrEnergyWithDensityVal[2 * q + spinIndex]) *
                  d_basisOperationsPtrHost
                    ->JxWBasisData()[iElemCount * numberQuadraturePoints + q];
              }

            iElemCount++;
          } // if cellPtr->is_locally_owned() loop

      } // cell loop

    cellPtr =
      dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex)
        .begin_active();
    iElemCount = 0;
    for (; cellPtr != endcellPtr; ++cellPtr)
      {
        if (cellPtr->is_locally_owned())
          {
            const auto &tempDensityTotalValues = rhoValues[0];
            const auto &tempDensityMagValues   = rhoValues[1];
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[2 * q + 0] =
                  0.5 *
                  (tempDensityTotalValues[iElemCount * numberQuadraturePoints +
                                          q] +
                   tempDensityMagValues[iElemCount * numberQuadraturePoints +
                                        q]);
                densityValue[2 * q + 1] =
                  0.5 *
                  (tempDensityTotalValues[iElemCount * numberQuadraturePoints +
                                          q] -
                   tempDensityMagValues[iElemCount * numberQuadraturePoints +
                                        q]);
              }


            if (dftPtr->d_dftParamsPtr->nonLinearCoreCorrection)
              {
                const std::vector<double> &temp2 =
                  rhoCoreValues.find(cellPtr->id())->second;


                for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                  {
                    densityValue[2 * q] += temp2[q] / 2.0;
                    densityValue[2 * q + 1] += temp2[q] / 2.0;
                  }
              }


            const auto &tempDensityTotalPrimeValues = rhoPrimeValues[0];
            const auto &tempDensityMagPrimeValues   = rhoPrimeValues[1];
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityPrimeValue[2 * q + 0] =
                  0.5 * (tempDensityTotalPrimeValues[iElemCount *
                                                       numberQuadraturePoints +
                                                     q] +
                         tempDensityMagPrimeValues[iElemCount *
                                                     numberQuadraturePoints +
                                                   q]);
                densityPrimeValue[2 * q + 1] =
                  0.5 * (tempDensityTotalPrimeValues[iElemCount *
                                                       numberQuadraturePoints +
                                                     q] -
                         tempDensityMagPrimeValues[iElemCount *
                                                     numberQuadraturePoints +
                                                   q]);
              }

            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[2 * q] += lambda * densityPrimeValue[2 * q];
                densityValue[2 * q + 1] +=
                  lambda * densityPrimeValue[2 * q + 1];
              }

            std::map<rhoDataAttributes, const std::vector<double> *> rhoData;

            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerExchangeEnergy;
            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerCorrEnergy;

            rhoData[rhoDataAttributes::values] = &densityValue;

            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derExchEnergyWithDensityVal;

            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derCorrEnergyWithDensityVal;

            dftPtr->d_excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
              numberQuadraturePoints,
              rhoData,
              outputDerExchangeEnergy,
              outputDerCorrEnergy);



            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                d_vEffJxW[iElemCount * numberQuadraturePoints + q] +=
                  8.0 *
                  (derExchEnergyWithDensityVal[2 * q + spinIndex] +
                   derCorrEnergyWithDensityVal[2 * q + spinIndex]) *
                  d_basisOperationsPtrHost
                    ->JxWBasisData()[iElemCount * numberQuadraturePoints + q];
              }


            iElemCount++;
          } // if cellPtr->is_locally_owned() loop

      } // cell loop


    cellPtr =
      dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex)
        .begin_active();
    iElemCount = 0;
    for (; cellPtr != endcellPtr; ++cellPtr)
      {
        if (cellPtr->is_locally_owned())
          {
            const auto &tempDensityTotalValues = rhoValues[0];
            const auto &tempDensityMagValues   = rhoValues[1];
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[2 * q + 0] =
                  0.5 *
                  (tempDensityTotalValues[iElemCount * numberQuadraturePoints +
                                          q] +
                   tempDensityMagValues[iElemCount * numberQuadraturePoints +
                                        q]);
                densityValue[2 * q + 1] =
                  0.5 *
                  (tempDensityTotalValues[iElemCount * numberQuadraturePoints +
                                          q] -
                   tempDensityMagValues[iElemCount * numberQuadraturePoints +
                                        q]);
              }


            if (dftPtr->d_dftParamsPtr->nonLinearCoreCorrection)
              {
                const std::vector<double> &temp2 =
                  rhoCoreValues.find(cellPtr->id())->second;

                for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                  {
                    densityValue[2 * q] += temp2[q] / 2.0;
                    densityValue[2 * q + 1] += temp2[q] / 2.0;
                  }
              }


            const auto &tempDensityTotalPrimeValues = rhoPrimeValues[0];
            const auto &tempDensityMagPrimeValues   = rhoPrimeValues[1];
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityPrimeValue[2 * q + 0] =
                  0.5 * (tempDensityTotalPrimeValues[iElemCount *
                                                       numberQuadraturePoints +
                                                     q] +
                         tempDensityMagPrimeValues[iElemCount *
                                                     numberQuadraturePoints +
                                                   q]);
                densityPrimeValue[2 * q + 1] =
                  0.5 * (tempDensityTotalPrimeValues[iElemCount *
                                                       numberQuadraturePoints +
                                                     q] -
                         tempDensityMagPrimeValues[iElemCount *
                                                     numberQuadraturePoints +
                                                   q]);
              }

            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[2 * q] -= 2.0 * lambda * densityPrimeValue[2 * q];
                densityValue[2 * q + 1] -=
                  2.0 * lambda * densityPrimeValue[2 * q + 1];
              }

            std::map<rhoDataAttributes, const std::vector<double> *> rhoData;

            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerExchangeEnergy;
            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerCorrEnergy;

            rhoData[rhoDataAttributes::values] = &densityValue;

            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derExchEnergyWithDensityVal;

            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derCorrEnergyWithDensityVal;

            dftPtr->d_excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
              numberQuadraturePoints,
              rhoData,
              outputDerExchangeEnergy,
              outputDerCorrEnergy);


            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                d_vEffJxW[iElemCount * numberQuadraturePoints + q] +=
                  (derExchEnergyWithDensityVal[2 * q + spinIndex] +
                   derCorrEnergyWithDensityVal[2 * q + spinIndex]) *
                  d_basisOperationsPtrHost
                    ->JxWBasisData()[iElemCount * numberQuadraturePoints + q];
              }


            iElemCount++;
          } // if cellPtr->is_locally_owned() loop

      } // cell loop


    cellPtr =
      dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex)
        .begin_active();
    iElemCount = 0;
    for (; cellPtr != endcellPtr; ++cellPtr)
      {
        if (cellPtr->is_locally_owned())
          {
            const auto &tempDensityTotalValues = rhoValues[0];
            const auto &tempDensityMagValues   = rhoValues[1];
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[2 * q + 0] =
                  0.5 *
                  (tempDensityTotalValues[iElemCount * numberQuadraturePoints +
                                          q] +
                   tempDensityMagValues[iElemCount * numberQuadraturePoints +
                                        q]);
                densityValue[2 * q + 1] =
                  0.5 *
                  (tempDensityTotalValues[iElemCount * numberQuadraturePoints +
                                          q] -
                   tempDensityMagValues[iElemCount * numberQuadraturePoints +
                                        q]);
              }

            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              phiPrimeValue[q] =
                phiPrimeValues[iElemCount * numberQuadraturePoints + q];

            if (dftPtr->d_dftParamsPtr->nonLinearCoreCorrection)
              {
                const std::vector<double> &temp2 =
                  rhoCoreValues.find(cellPtr->id())->second;

                for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                  {
                    densityValue[2 * q] += temp2[q] / 2.0;
                    densityValue[2 * q + 1] += temp2[q] / 2.0;
                  }
              }


            const auto &tempDensityTotalPrimeValues = rhoPrimeValues[0];
            const auto &tempDensityMagPrimeValues   = rhoPrimeValues[1];
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityPrimeValue[2 * q + 0] =
                  0.5 * (tempDensityTotalPrimeValues[iElemCount *
                                                       numberQuadraturePoints +
                                                     q] +
                         tempDensityMagPrimeValues[iElemCount *
                                                     numberQuadraturePoints +
                                                   q]);
                densityPrimeValue[2 * q + 1] =
                  0.5 * (tempDensityTotalPrimeValues[iElemCount *
                                                       numberQuadraturePoints +
                                                     q] -
                         tempDensityMagPrimeValues[iElemCount *
                                                     numberQuadraturePoints +
                                                   q]);
              }


            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[2 * q] -= lambda * densityPrimeValue[2 * q];
                densityValue[2 * q + 1] -=
                  lambda * densityPrimeValue[2 * q + 1];
              }

            std::map<rhoDataAttributes, const std::vector<double> *> rhoData;

            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerExchangeEnergy;
            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerCorrEnergy;

            rhoData[rhoDataAttributes::values] = &densityValue;

            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derExchEnergyWithDensityVal;

            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derCorrEnergyWithDensityVal;

            dftPtr->d_excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
              numberQuadraturePoints,
              rhoData,
              outputDerExchangeEnergy,
              outputDerCorrEnergy);



            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                d_vEffJxW[iElemCount * numberQuadraturePoints + q] -=
                  8.0 *
                  (derExchEnergyWithDensityVal[2 * q + spinIndex] +
                   derCorrEnergyWithDensityVal[2 * q + spinIndex]) *
                  d_basisOperationsPtrHost
                    ->JxWBasisData()[iElemCount * numberQuadraturePoints + q];

                d_vEffJxW[iElemCount * numberQuadraturePoints + q] *=
                  1.0 / 12.0 / lambda;
                d_vEffJxW[iElemCount * numberQuadraturePoints + q] +=
                  phiPrimeValue[q] *
                  d_basisOperationsPtrHost
                    ->JxWBasisData()[iElemCount * numberQuadraturePoints + q];
              }

            iElemCount++;
          } // if cellPtr->is_locally_owned() loop

      } // cell loop
    d_vEffJxWDevice.resize(d_vEffJxW.size());
    d_vEffJxWDevice.copyFrom(d_vEffJxW);
  }


  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    computeVEffPrime(
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &rhoValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &rhoPrimeValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &gradRhoValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &gradRhoPrimeValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &                                                  phiPrimeValues,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
      const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues)
  {
    d_basisOperationsPtrHost->reinit(0, 0, dftPtr->d_densityQuadratureId);
    const unsigned int totalLocallyOwnedCells =
      dftPtr->matrix_free_data.n_physical_cells();

    const dealii::Quadrature<3> &quadrature_formula =
      dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId);
    const unsigned int numberQuadraturePoints = quadrature_formula.size();


    d_vEff.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);
    d_vEffJxW.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);
    d_derExcWithSigmaTimesGradRhoJxW.resize(totalLocallyOwnedCells *
                                              numberQuadraturePoints * 3,
                                            0.0);

    typename dealii::DoFHandler<3>::active_cell_iterator cellPtr =
      dftPtr->matrix_free_data.get_dof_handler().begin_active();
    typename dealii::DoFHandler<3>::active_cell_iterator endcPtr =
      dftPtr->matrix_free_data.get_dof_handler().end();
    unsigned int iElemCount = 0;

    std::vector<double> densityValue(numberQuadraturePoints);
    std::vector<double> gradDensityValue(3 * numberQuadraturePoints);
    std::vector<double> densityPrimeValue(numberQuadraturePoints);
    std::vector<double> gradDensityPrimeValue(3 * numberQuadraturePoints);
    std::vector<double> phiPrimeValue(numberQuadraturePoints);

    std::vector<double> sigmaValue(numberQuadraturePoints);
    std::vector<double> derExchEnergyWithSigmaVal(numberQuadraturePoints);
    std::vector<double> derCorrEnergyWithSigmaVal(numberQuadraturePoints);
    std::vector<double> der2ExchEnergyWithSigmaVal(numberQuadraturePoints);
    std::vector<double> der2CorrEnergyWithSigmaVal(numberQuadraturePoints);
    std::vector<double> der2ExchEnergyWithDensitySigmaVal(
      numberQuadraturePoints);
    std::vector<double> der2CorrEnergyWithDensitySigmaVal(
      numberQuadraturePoints);
    std::vector<double> derExchEnergyWithDensityVal(numberQuadraturePoints);
    std::vector<double> derCorrEnergyWithDensityVal(numberQuadraturePoints);
    std::vector<double> der2ExchEnergyWithDensityVal(numberQuadraturePoints);
    std::vector<double> der2CorrEnergyWithDensityVal(numberQuadraturePoints);

    for (; cellPtr != endcPtr; ++cellPtr)
      if (cellPtr->is_locally_owned())
        {
          const auto &tempDensityTotalValues     = rhoValues[0];
          const auto &tempGradDensityTotalValues = gradRhoValues[0];
          for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
            {
              densityValue[q] =
                tempDensityTotalValues[iElemCount * numberQuadraturePoints + q];
              gradDensityValue[3 * q + 0] =
                tempGradDensityTotalValues[3 * iElemCount *
                                             numberQuadraturePoints +
                                           3 * q + 0];
              gradDensityValue[3 * q + 1] =
                tempGradDensityTotalValues[3 * iElemCount *
                                             numberQuadraturePoints +
                                           3 * q + 1];
              gradDensityValue[3 * q + 2] =
                tempGradDensityTotalValues[3 * iElemCount *
                                             numberQuadraturePoints +
                                           3 * q + 2];
            }

          const auto &tempDensityTotalPrimeValues     = rhoPrimeValues[0];
          const auto &tempGradDensityTotalPrimeValues = gradRhoPrimeValues[0];
          for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
            {
              densityPrimeValue[q] =
                tempDensityTotalPrimeValues[iElemCount *
                                              numberQuadraturePoints +
                                            q];
              for (unsigned int idim = 0; idim < 3; ++idim)
                gradDensityPrimeValue[3 * q + idim] =
                  tempGradDensityTotalPrimeValues[3 * iElemCount *
                                                    numberQuadraturePoints +
                                                  3 * q + idim];
            }

          for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
            phiPrimeValue[q] =
              phiPrimeValues[iElemCount * numberQuadraturePoints + q];

          if (dftPtr->d_dftParamsPtr->nonLinearCoreCorrection)
            {
              const std::vector<double> &temp2 =
                rhoCoreValues.find(cellPtr->id())->second;
              const std::vector<double> &temp3 =
                gradRhoCoreValues.find(cellPtr->id())->second;
              for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                {
                  densityValue[q] += temp2[q];
                  gradDensityValue[3 * q + 0] += temp3[3 * q + 0];
                  gradDensityValue[3 * q + 1] += temp3[3 * q + 1];
                  gradDensityValue[3 * q + 2] += temp3[3 * q + 2];
                }
            }


          for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
            {
              const double gradRhoX = gradDensityValue[3 * q + 0];
              const double gradRhoY = gradDensityValue[3 * q + 1];
              const double gradRhoZ = gradDensityValue[3 * q + 2];
              sigmaValue[q] =
                gradRhoX * gradRhoX + gradRhoY * gradRhoY + gradRhoZ * gradRhoZ;
            }
          std::map<rhoDataAttributes, const std::vector<double> *> rhoData;


          std::map<VeffOutputDataAttributes, std::vector<double> *>
            outputDerExchangeEnergy;
          std::map<VeffOutputDataAttributes, std::vector<double> *>
            outputDerCorrEnergy;

          std::map<fxcOutputDataAttributes, std::vector<double> *>
            outputDer2ExchangeEnergy;
          std::map<fxcOutputDataAttributes, std::vector<double> *>
            outputDer2CorrEnergy;


          rhoData[rhoDataAttributes::values]         = &densityValue;
          rhoData[rhoDataAttributes::sigmaGradValue] = &sigmaValue;

          outputDerExchangeEnergy
            [VeffOutputDataAttributes::derEnergyWithDensity] =
              &derExchEnergyWithDensityVal;
          outputDerExchangeEnergy
            [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
              &derExchEnergyWithSigmaVal;

          outputDerCorrEnergy[VeffOutputDataAttributes::derEnergyWithDensity] =
            &derCorrEnergyWithDensityVal;
          outputDerCorrEnergy
            [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
              &derCorrEnergyWithSigmaVal;

          outputDer2ExchangeEnergy
            [fxcOutputDataAttributes::der2EnergyWithDensity] =
              &der2ExchEnergyWithDensityVal;
          outputDer2ExchangeEnergy
            [fxcOutputDataAttributes::der2EnergyWithDensitySigma] =
              &der2ExchEnergyWithDensitySigmaVal;
          outputDer2ExchangeEnergy
            [fxcOutputDataAttributes::der2EnergyWithSigma] =
              &der2ExchEnergyWithSigmaVal;

          outputDer2CorrEnergy[fxcOutputDataAttributes::der2EnergyWithDensity] =
            &der2CorrEnergyWithDensityVal;
          outputDer2CorrEnergy
            [fxcOutputDataAttributes::der2EnergyWithDensitySigma] =
              &der2CorrEnergyWithDensitySigmaVal;
          outputDer2CorrEnergy[fxcOutputDataAttributes::der2EnergyWithSigma] =
            &der2CorrEnergyWithSigmaVal;


          dftPtr->d_excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
            numberQuadraturePoints,
            rhoData,
            outputDerExchangeEnergy,
            outputDerCorrEnergy);

          dftPtr->d_excManagerPtr->getExcDensityObj()->computeDensityBasedFxc(
            numberQuadraturePoints,
            rhoData,
            outputDer2ExchangeEnergy,
            outputDer2CorrEnergy);


          for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
            {
              const double jxw =
                d_basisOperationsPtrHost
                  ->JxWBasisData()[iElemCount * numberQuadraturePoints + q];
              const double gradRhoX = gradDensityValue[3 * q + 0];
              const double gradRhoY = gradDensityValue[3 * q + 1];
              const double gradRhoZ = gradDensityValue[3 * q + 2];

              const double gradRhoPrimeX = gradDensityPrimeValue[3 * q + 0];
              const double gradRhoPrimeY = gradDensityPrimeValue[3 * q + 1];
              const double gradRhoPrimeZ = gradDensityPrimeValue[3 * q + 2];

              const double gradRhoDotGradRhoPrime = gradRhoX * gradRhoPrimeX +
                                                    gradRhoY * gradRhoPrimeY +
                                                    gradRhoZ * gradRhoPrimeZ;

              const double term1 =
                derExchEnergyWithSigmaVal[q] + derCorrEnergyWithSigmaVal[q];
              const double term2 =
                der2ExchEnergyWithSigmaVal[q] + der2CorrEnergyWithSigmaVal[q];
              const double term3 = der2ExchEnergyWithDensitySigmaVal[q] +
                                   der2CorrEnergyWithDensitySigmaVal[q];
              d_derExcWithSigmaTimesGradRhoJxW[iElemCount *
                                                 numberQuadraturePoints * 3 +
                                               3 * q] =
                (term1 * gradRhoPrimeX +
                 2.0 * term2 * gradRhoDotGradRhoPrime * gradRhoX +
                 term3 * densityPrimeValue[q] * gradRhoX) *
                jxw;
              d_derExcWithSigmaTimesGradRhoJxW[iElemCount *
                                                 numberQuadraturePoints * 3 +
                                               3 * q + 1] =
                (term1 * gradRhoPrimeY +
                 2.0 * term2 * gradRhoDotGradRhoPrime * gradRhoY +
                 term3 * densityPrimeValue[q] * gradRhoY) *
                jxw;
              d_derExcWithSigmaTimesGradRhoJxW[iElemCount *
                                                 numberQuadraturePoints * 3 +
                                               3 * q + 2] =
                (term1 * gradRhoPrimeZ +
                 2.0 * term2 * gradRhoDotGradRhoPrime * gradRhoZ +
                 term3 * densityPrimeValue[q] * gradRhoZ) *
                jxw;
            }

          for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
            {
              const double gradRhoX = gradDensityValue[3 * q + 0];
              const double gradRhoY = gradDensityValue[3 * q + 1];
              const double gradRhoZ = gradDensityValue[3 * q + 2];

              const double gradRhoPrimeX = gradDensityPrimeValue[3 * q + 0];
              const double gradRhoPrimeY = gradDensityPrimeValue[3 * q + 1];
              const double gradRhoPrimeZ = gradDensityPrimeValue[3 * q + 2];

              const double gradRhoDotGradRhoPrime = gradRhoX * gradRhoPrimeX +
                                                    gradRhoY * gradRhoPrimeY +
                                                    gradRhoZ * gradRhoPrimeZ;

              // 2.0*del2{exc}/del{sigma}{rho}*\dot{gradrho^{\prime},gradrho}
              const double sigmaDensityMixedDerTerm =
                2.0 *
                (der2ExchEnergyWithDensitySigmaVal[q] +
                 der2CorrEnergyWithDensitySigmaVal[q]) *
                gradRhoDotGradRhoPrime;

              d_vEff[iElemCount * numberQuadraturePoints + q] =
                phiPrimeValue[q] +
                (der2ExchEnergyWithDensityVal[q] +
                 der2CorrEnergyWithDensityVal[q]) *
                  densityPrimeValue[q] +
                sigmaDensityMixedDerTerm;

              d_vEffJxW[iElemCount * numberQuadraturePoints + q] =
                d_vEff[iElemCount * numberQuadraturePoints + q] *
                d_basisOperationsPtrHost
                  ->JxWBasisData()[iElemCount * numberQuadraturePoints + q];
            }

          iElemCount++;
        }

    d_vEffJxWDevice.resize(d_vEffJxW.size());
    d_vEffJxWDevice.copyFrom(d_vEffJxW);

    d_derExcWithSigmaTimesGradRhoJxWDevice.resize(
      d_derExcWithSigmaTimesGradRhoJxW.size());
    d_derExcWithSigmaTimesGradRhoJxWDevice.copyFrom(
      d_derExcWithSigmaTimesGradRhoJxW);
  }


  // Fourth order stencil finite difference stencil used
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    computeVEffPrimeSpinPolarized(
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &rhoValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &rhoPrimeValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &gradRhoValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &gradRhoPrimeValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &                                                  phiPrimeValues,
      const unsigned int                                   spinIndex,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
      const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues)
  {
    d_basisOperationsPtrHost->reinit(0, 0, dftPtr->d_densityQuadratureId);
    const unsigned int totalLocallyOwnedCells =
      dftPtr->matrix_free_data.n_physical_cells();
    const dealii::Quadrature<3> &quadrature_formula =
      dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId);

    const unsigned int numberQuadraturePoints = quadrature_formula.size();

    d_vEffJxW.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);
    d_derExcWithSigmaTimesGradRhoJxW.resize(totalLocallyOwnedCells *
                                              numberQuadraturePoints * 3,
                                            0.0);

    std::vector<double> densityValue(2 * numberQuadraturePoints);
    std::vector<double> gradDensityValue(6 * numberQuadraturePoints);
    std::vector<double> densityPrimeValue(2 * numberQuadraturePoints);
    std::vector<double> gradDensityPrimeValue(6 * numberQuadraturePoints);
    std::vector<double> phiPrimeValue(numberQuadraturePoints);

    std::vector<double> derExchEnergyWithDensityVal(2 * numberQuadraturePoints);
    std::vector<double> derCorrEnergyWithDensityVal(2 * numberQuadraturePoints);
    std::vector<double> derExchEnergyWithSigma(3 * numberQuadraturePoints);
    std::vector<double> derCorrEnergyWithSigma(3 * numberQuadraturePoints);
    std::vector<double> sigmaValue(3 * numberQuadraturePoints);

    typename dealii::DoFHandler<3>::active_cell_iterator
      cellPtr = dftPtr->matrix_free_data
                  .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
                  .begin_active(),
      endcellPtr = dftPtr->matrix_free_data
                     .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
                     .end();
    const double lambda = 1e-2;

    //
    // loop over cell block
    //
    unsigned int iElemCount = 0;
    for (; cellPtr != endcellPtr; ++cellPtr)
      {
        if (cellPtr->is_locally_owned())
          {
            const auto &tempDensityTotalValues     = rhoValues[0];
            const auto &tempDensityMagValues       = rhoValues[1];
            const auto &tempGradDensityTotalValues = gradRhoValues[0];
            const auto &tempGradDensityMagValues   = gradRhoValues[1];
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[2 * q + 0] =
                  0.5 *
                  (tempDensityTotalValues[iElemCount * numberQuadraturePoints +
                                          q] +
                   tempDensityMagValues[iElemCount * numberQuadraturePoints +
                                        q]);
                densityValue[2 * q + 1] =
                  0.5 *
                  (tempDensityTotalValues[iElemCount * numberQuadraturePoints +
                                          q] -
                   tempDensityMagValues[iElemCount * numberQuadraturePoints +
                                        q]);
                for (unsigned int idim = 0; idim < 3; ++idim)
                  {
                    gradDensityValue[6 * q + idim] =
                      0.5 *
                      (tempGradDensityTotalValues[3 * iElemCount *
                                                    numberQuadraturePoints +
                                                  3 * q + idim] +
                       tempGradDensityMagValues[3 * iElemCount *
                                                  numberQuadraturePoints +
                                                3 * q + idim]);
                    gradDensityValue[6 * q + 3 + idim] =
                      0.5 *
                      (tempGradDensityTotalValues[3 * iElemCount *
                                                    numberQuadraturePoints +
                                                  3 * q + idim] -
                       tempGradDensityMagValues[3 * iElemCount *
                                                  numberQuadraturePoints +
                                                3 * q + idim]);
                  }
              }


            if (dftPtr->d_dftParamsPtr->nonLinearCoreCorrection)
              {
                const std::vector<double> &temp2 =
                  rhoCoreValues.find(cellPtr->id())->second;

                const std::vector<double> &temp3 =
                  gradRhoCoreValues.find(cellPtr->id())->second;

                for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                  {
                    densityValue[2 * q] += temp2[q] / 2.0;
                    densityValue[2 * q + 1] += temp2[q] / 2.0;
                    gradDensityValue[6 * q + 0] += temp3[3 * q + 0] / 2.0;
                    gradDensityValue[6 * q + 1] += temp3[3 * q + 1] / 2.0;
                    gradDensityValue[6 * q + 2] += temp3[3 * q + 2] / 2.0;
                    gradDensityValue[6 * q + 3] += temp3[3 * q + 0] / 2.0;
                    gradDensityValue[6 * q + 4] += temp3[3 * q + 1] / 2.0;
                    gradDensityValue[6 * q + 5] += temp3[3 * q + 2] / 2.0;
                  }
              }

            const auto &tempDensityTotalPrimeValues     = rhoPrimeValues[0];
            const auto &tempDensityMagPrimeValues       = rhoPrimeValues[1];
            const auto &tempGradDensityTotalPrimeValues = gradRhoPrimeValues[0];
            const auto &tempGradDensityMagPrimeValues   = gradRhoPrimeValues[1];
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityPrimeValue[2 * q + 0] =
                  0.5 * (tempDensityTotalPrimeValues[iElemCount *
                                                       numberQuadraturePoints +
                                                     q] +
                         tempDensityMagPrimeValues[iElemCount *
                                                     numberQuadraturePoints +
                                                   q]);
                densityPrimeValue[2 * q + 1] =
                  0.5 * (tempDensityTotalPrimeValues[iElemCount *
                                                       numberQuadraturePoints +
                                                     q] -
                         tempDensityMagPrimeValues[iElemCount *
                                                     numberQuadraturePoints +
                                                   q]);
                for (unsigned int idim = 0; idim < 3; ++idim)
                  {
                    gradDensityPrimeValue[6 * q + idim] =
                      0.5 *
                      (tempGradDensityTotalPrimeValues
                         [3 * iElemCount * numberQuadraturePoints + 3 * q +
                          idim] +
                       tempGradDensityMagPrimeValues[3 * iElemCount *
                                                       numberQuadraturePoints +
                                                     3 * q + idim]);
                    gradDensityPrimeValue[6 * q + 3 + idim] =
                      0.5 *
                      (tempGradDensityTotalPrimeValues
                         [3 * iElemCount * numberQuadraturePoints + 3 * q +
                          idim] -
                       tempGradDensityMagPrimeValues[3 * iElemCount *
                                                       numberQuadraturePoints +
                                                     3 * q + idim]);
                  }
              }

            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[2 * q] += 2.0 * lambda * densityPrimeValue[2 * q];
                densityValue[2 * q + 1] +=
                  2.0 * lambda * densityPrimeValue[2 * q + 1];
                gradDensityValue[6 * q + 0] +=
                  2.0 * lambda * gradDensityPrimeValue[6 * q + 0];
                gradDensityValue[6 * q + 1] +=
                  2.0 * lambda * gradDensityPrimeValue[6 * q + 1];
                gradDensityValue[6 * q + 2] +=
                  2.0 * lambda * gradDensityPrimeValue[6 * q + 2];
                gradDensityValue[6 * q + 3] +=
                  2.0 * lambda * gradDensityPrimeValue[6 * q + 3];
                gradDensityValue[6 * q + 4] +=
                  2.0 * lambda * gradDensityPrimeValue[6 * q + 4];
                gradDensityValue[6 * q + 5] +=
                  2.0 * lambda * gradDensityPrimeValue[6 * q + 5];
              }


            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                const double gradRhoX1 = gradDensityValue[6 * q + 0];
                const double gradRhoY1 = gradDensityValue[6 * q + 1];
                const double gradRhoZ1 = gradDensityValue[6 * q + 2];
                const double gradRhoX2 = gradDensityValue[6 * q + 3];
                const double gradRhoY2 = gradDensityValue[6 * q + 4];
                const double gradRhoZ2 = gradDensityValue[6 * q + 5];

                sigmaValue[3 * q + 0] = gradRhoX1 * gradRhoX1 +
                                        gradRhoY1 * gradRhoY1 +
                                        gradRhoZ1 * gradRhoZ1;
                sigmaValue[3 * q + 1] = gradRhoX1 * gradRhoX2 +
                                        gradRhoY1 * gradRhoY2 +
                                        gradRhoZ1 * gradRhoZ2;
                sigmaValue[3 * q + 2] = gradRhoX2 * gradRhoX2 +
                                        gradRhoY2 * gradRhoY2 +
                                        gradRhoZ2 * gradRhoZ2;
              }

            std::map<rhoDataAttributes, const std::vector<double> *> rhoData;

            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerExchangeEnergy;
            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerCorrEnergy;


            rhoData[rhoDataAttributes::values]         = &densityValue;
            rhoData[rhoDataAttributes::sigmaGradValue] = &sigmaValue;

            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derExchEnergyWithDensityVal;
            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
                &derExchEnergyWithSigma;

            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derCorrEnergyWithDensityVal;
            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
                &derCorrEnergyWithSigma;

            dftPtr->d_excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
              numberQuadraturePoints,
              rhoData,
              outputDerExchangeEnergy,
              outputDerCorrEnergy);



            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                d_vEffJxW[iElemCount * numberQuadraturePoints + q] =
                  -(derExchEnergyWithDensityVal[2 * q + spinIndex] +
                    derCorrEnergyWithDensityVal[2 * q + spinIndex]) *
                  d_basisOperationsPtrHost
                    ->JxWBasisData()[iElemCount * numberQuadraturePoints + q];
              }

            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                const double jxw =
                  d_basisOperationsPtrHost
                    ->JxWBasisData()[iElemCount * numberQuadraturePoints + q];
                const double gradRhoX =
                  gradDensityValue[6 * q + 0 + 3 * spinIndex];
                const double gradRhoY =
                  gradDensityValue[6 * q + 1 + 3 * spinIndex];
                const double gradRhoZ =
                  gradDensityValue[6 * q + 2 + 3 * spinIndex];
                const double gradRhoOtherX =
                  gradDensityValue[6 * q + 0 + 3 * (1 - spinIndex)];
                const double gradRhoOtherY =
                  gradDensityValue[6 * q + 1 + 3 * (1 - spinIndex)];
                const double gradRhoOtherZ =
                  gradDensityValue[6 * q + 2 + 3 * (1 - spinIndex)];
                const double term =
                  derExchEnergyWithSigma[3 * q + 2 * spinIndex] +
                  derCorrEnergyWithSigma[3 * q + 2 * spinIndex];
                const double termOff = derExchEnergyWithSigma[3 * q + 1] +
                                       derCorrEnergyWithSigma[3 * q + 1];

                d_derExcWithSigmaTimesGradRhoJxW[iElemCount *
                                                   numberQuadraturePoints * 3 +
                                                 3 * q] =
                  -1.0 * (term * gradRhoX + 0.5 * termOff * gradRhoOtherX) *
                  jxw;
                d_derExcWithSigmaTimesGradRhoJxW[iElemCount *
                                                   numberQuadraturePoints * 3 +
                                                 3 * q + 1] =
                  -1.0 * (term * gradRhoY + 0.5 * termOff * gradRhoOtherY) *
                  jxw;
                d_derExcWithSigmaTimesGradRhoJxW[iElemCount *
                                                   numberQuadraturePoints * 3 +
                                                 3 * q + 2] =
                  -1.0 * (term * gradRhoZ + 0.5 * termOff * gradRhoOtherZ) *
                  jxw;
              }
            iElemCount++;
          } // if cellPtr->is_locally_owned() loop

      } // cell loop

    cellPtr =
      dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex)
        .begin_active();
    iElemCount = 0;
    for (; cellPtr != endcellPtr; ++cellPtr)
      {
        if (cellPtr->is_locally_owned())
          {
            const auto &tempDensityTotalValues     = rhoValues[0];
            const auto &tempDensityMagValues       = rhoValues[1];
            const auto &tempGradDensityTotalValues = gradRhoValues[0];
            const auto &tempGradDensityMagValues   = gradRhoValues[1];
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[2 * q + 0] =
                  0.5 *
                  (tempDensityTotalValues[iElemCount * numberQuadraturePoints +
                                          q] +
                   tempDensityMagValues[iElemCount * numberQuadraturePoints +
                                        q]);
                densityValue[2 * q + 1] =
                  0.5 *
                  (tempDensityTotalValues[iElemCount * numberQuadraturePoints +
                                          q] -
                   tempDensityMagValues[iElemCount * numberQuadraturePoints +
                                        q]);
                for (unsigned int idim = 0; idim < 3; ++idim)
                  {
                    gradDensityValue[6 * q + idim] =
                      0.5 *
                      (tempGradDensityTotalValues[3 * iElemCount *
                                                    numberQuadraturePoints +
                                                  3 * q + idim] +
                       tempGradDensityMagValues[3 * iElemCount *
                                                  numberQuadraturePoints +
                                                3 * q + idim]);
                    gradDensityValue[6 * q + 3 + idim] =
                      0.5 *
                      (tempGradDensityTotalValues[3 * iElemCount *
                                                    numberQuadraturePoints +
                                                  3 * q + idim] -
                       tempGradDensityMagValues[3 * iElemCount *
                                                  numberQuadraturePoints +
                                                3 * q + idim]);
                  }
              }


            if (dftPtr->d_dftParamsPtr->nonLinearCoreCorrection)
              {
                const std::vector<double> &temp2 =
                  rhoCoreValues.find(cellPtr->id())->second;

                const std::vector<double> &temp3 =
                  gradRhoCoreValues.find(cellPtr->id())->second;

                for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                  {
                    densityValue[2 * q] += temp2[q] / 2.0;
                    densityValue[2 * q + 1] += temp2[q] / 2.0;
                    gradDensityValue[6 * q + 0] += temp3[3 * q + 0] / 2.0;
                    gradDensityValue[6 * q + 1] += temp3[3 * q + 1] / 2.0;
                    gradDensityValue[6 * q + 2] += temp3[3 * q + 2] / 2.0;
                    gradDensityValue[6 * q + 3] += temp3[3 * q + 0] / 2.0;
                    gradDensityValue[6 * q + 4] += temp3[3 * q + 1] / 2.0;
                    gradDensityValue[6 * q + 5] += temp3[3 * q + 2] / 2.0;
                  }
              }


            const auto &tempDensityTotalPrimeValues     = rhoPrimeValues[0];
            const auto &tempDensityMagPrimeValues       = rhoPrimeValues[1];
            const auto &tempGradDensityTotalPrimeValues = gradRhoPrimeValues[0];
            const auto &tempGradDensityMagPrimeValues   = gradRhoPrimeValues[1];
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityPrimeValue[2 * q + 0] =
                  0.5 * (tempDensityTotalPrimeValues[iElemCount *
                                                       numberQuadraturePoints +
                                                     q] +
                         tempDensityMagPrimeValues[iElemCount *
                                                     numberQuadraturePoints +
                                                   q]);
                densityPrimeValue[2 * q + 1] =
                  0.5 * (tempDensityTotalPrimeValues[iElemCount *
                                                       numberQuadraturePoints +
                                                     q] -
                         tempDensityMagPrimeValues[iElemCount *
                                                     numberQuadraturePoints +
                                                   q]);
                for (unsigned int idim = 0; idim < 3; ++idim)
                  {
                    gradDensityPrimeValue[6 * q + idim] =
                      0.5 *
                      (tempGradDensityTotalPrimeValues
                         [3 * iElemCount * numberQuadraturePoints + 3 * q +
                          idim] +
                       tempGradDensityMagPrimeValues[3 * iElemCount *
                                                       numberQuadraturePoints +
                                                     3 * q + idim]);
                    gradDensityPrimeValue[6 * q + 3 + idim] =
                      0.5 *
                      (tempGradDensityTotalPrimeValues
                         [3 * iElemCount * numberQuadraturePoints + 3 * q +
                          idim] -
                       tempGradDensityMagPrimeValues[3 * iElemCount *
                                                       numberQuadraturePoints +
                                                     3 * q + idim]);
                  }
              }

            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[2 * q] += lambda * densityPrimeValue[2 * q];
                densityValue[2 * q + 1] +=
                  lambda * densityPrimeValue[2 * q + 1];
                gradDensityValue[6 * q + 0] +=
                  lambda * gradDensityPrimeValue[6 * q + 0];
                gradDensityValue[6 * q + 1] +=
                  lambda * gradDensityPrimeValue[6 * q + 1];
                gradDensityValue[6 * q + 2] +=
                  lambda * gradDensityPrimeValue[6 * q + 2];
                gradDensityValue[6 * q + 3] +=
                  lambda * gradDensityPrimeValue[6 * q + 3];
                gradDensityValue[6 * q + 4] +=
                  lambda * gradDensityPrimeValue[6 * q + 4];
                gradDensityValue[6 * q + 5] +=
                  lambda * gradDensityPrimeValue[6 * q + 5];
              }


            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                const double gradRhoX1 = gradDensityValue[6 * q + 0];
                const double gradRhoY1 = gradDensityValue[6 * q + 1];
                const double gradRhoZ1 = gradDensityValue[6 * q + 2];
                const double gradRhoX2 = gradDensityValue[6 * q + 3];
                const double gradRhoY2 = gradDensityValue[6 * q + 4];
                const double gradRhoZ2 = gradDensityValue[6 * q + 5];

                sigmaValue[3 * q + 0] = gradRhoX1 * gradRhoX1 +
                                        gradRhoY1 * gradRhoY1 +
                                        gradRhoZ1 * gradRhoZ1;
                sigmaValue[3 * q + 1] = gradRhoX1 * gradRhoX2 +
                                        gradRhoY1 * gradRhoY2 +
                                        gradRhoZ1 * gradRhoZ2;
                sigmaValue[3 * q + 2] = gradRhoX2 * gradRhoX2 +
                                        gradRhoY2 * gradRhoY2 +
                                        gradRhoZ2 * gradRhoZ2;
              }

            std::map<rhoDataAttributes, const std::vector<double> *> rhoData;

            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerExchangeEnergy;
            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerCorrEnergy;


            rhoData[rhoDataAttributes::values]         = &densityValue;
            rhoData[rhoDataAttributes::sigmaGradValue] = &sigmaValue;

            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derExchEnergyWithDensityVal;
            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
                &derExchEnergyWithSigma;

            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derCorrEnergyWithDensityVal;
            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
                &derCorrEnergyWithSigma;

            dftPtr->d_excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
              numberQuadraturePoints,
              rhoData,
              outputDerExchangeEnergy,
              outputDerCorrEnergy);



            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                d_vEffJxW[iElemCount * numberQuadraturePoints + q] +=
                  8.0 *
                  (derExchEnergyWithDensityVal[2 * q + spinIndex] +
                   derCorrEnergyWithDensityVal[2 * q + spinIndex]) *
                  d_basisOperationsPtrHost
                    ->JxWBasisData()[iElemCount * numberQuadraturePoints + q];
              }

            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                const double jxw =
                  d_basisOperationsPtrHost
                    ->JxWBasisData()[iElemCount * numberQuadraturePoints + q];
                const double gradRhoX =
                  gradDensityValue[6 * q + 0 + 3 * spinIndex];
                const double gradRhoY =
                  gradDensityValue[6 * q + 1 + 3 * spinIndex];
                const double gradRhoZ =
                  gradDensityValue[6 * q + 2 + 3 * spinIndex];
                const double gradRhoOtherX =
                  gradDensityValue[6 * q + 0 + 3 * (1 - spinIndex)];
                const double gradRhoOtherY =
                  gradDensityValue[6 * q + 1 + 3 * (1 - spinIndex)];
                const double gradRhoOtherZ =
                  gradDensityValue[6 * q + 2 + 3 * (1 - spinIndex)];
                const double term =
                  derExchEnergyWithSigma[3 * q + 2 * spinIndex] +
                  derCorrEnergyWithSigma[3 * q + 2 * spinIndex];
                const double termOff = derExchEnergyWithSigma[3 * q + 1] +
                                       derCorrEnergyWithSigma[3 * q + 1];

                d_derExcWithSigmaTimesGradRhoJxW[iElemCount *
                                                   numberQuadraturePoints * 3 +
                                                 3 * q] +=
                  8.0 * (term * gradRhoX + 0.5 * termOff * gradRhoOtherX) * jxw;
                d_derExcWithSigmaTimesGradRhoJxW[iElemCount *
                                                   numberQuadraturePoints * 3 +
                                                 3 * q + 1] +=
                  8.0 * (term * gradRhoY + 0.5 * termOff * gradRhoOtherY) * jxw;
                d_derExcWithSigmaTimesGradRhoJxW[iElemCount *
                                                   numberQuadraturePoints * 3 +
                                                 3 * q + 2] +=
                  8.0 * (term * gradRhoZ + 0.5 * termOff * gradRhoOtherZ) * jxw;
              }
            iElemCount++;
          } // if cellPtr->is_locally_owned() loop

      } // cell loop


    cellPtr =
      dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex)
        .begin_active();
    iElemCount = 0;
    for (; cellPtr != endcellPtr; ++cellPtr)
      {
        if (cellPtr->is_locally_owned())
          {
            const auto &tempDensityTotalValues     = rhoValues[0];
            const auto &tempDensityMagValues       = rhoValues[1];
            const auto &tempGradDensityTotalValues = gradRhoValues[0];
            const auto &tempGradDensityMagValues   = gradRhoValues[1];
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[2 * q + 0] =
                  0.5 *
                  (tempDensityTotalValues[iElemCount * numberQuadraturePoints +
                                          q] +
                   tempDensityMagValues[iElemCount * numberQuadraturePoints +
                                        q]);
                densityValue[2 * q + 1] =
                  0.5 *
                  (tempDensityTotalValues[iElemCount * numberQuadraturePoints +
                                          q] -
                   tempDensityMagValues[iElemCount * numberQuadraturePoints +
                                        q]);
                for (unsigned int idim = 0; idim < 3; ++idim)
                  {
                    gradDensityValue[6 * q + idim] =
                      0.5 *
                      (tempGradDensityTotalValues[3 * iElemCount *
                                                    numberQuadraturePoints +
                                                  3 * q + idim] +
                       tempGradDensityMagValues[3 * iElemCount *
                                                  numberQuadraturePoints +
                                                3 * q + idim]);
                    gradDensityValue[6 * q + 3 + idim] =
                      0.5 *
                      (tempGradDensityTotalValues[3 * iElemCount *
                                                    numberQuadraturePoints +
                                                  3 * q + idim] -
                       tempGradDensityMagValues[3 * iElemCount *
                                                  numberQuadraturePoints +
                                                3 * q + idim]);
                  }
              }


            if (dftPtr->d_dftParamsPtr->nonLinearCoreCorrection)
              {
                const std::vector<double> &temp2 =
                  rhoCoreValues.find(cellPtr->id())->second;

                const std::vector<double> &temp3 =
                  gradRhoCoreValues.find(cellPtr->id())->second;

                for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                  {
                    densityValue[2 * q] += temp2[q] / 2.0;
                    densityValue[2 * q + 1] += temp2[q] / 2.0;
                    gradDensityValue[6 * q + 0] += temp3[3 * q + 0] / 2.0;
                    gradDensityValue[6 * q + 1] += temp3[3 * q + 1] / 2.0;
                    gradDensityValue[6 * q + 2] += temp3[3 * q + 2] / 2.0;
                    gradDensityValue[6 * q + 3] += temp3[3 * q + 0] / 2.0;
                    gradDensityValue[6 * q + 4] += temp3[3 * q + 1] / 2.0;
                    gradDensityValue[6 * q + 5] += temp3[3 * q + 2] / 2.0;
                  }
              }


            const auto &tempDensityTotalPrimeValues     = rhoPrimeValues[0];
            const auto &tempDensityMagPrimeValues       = rhoPrimeValues[1];
            const auto &tempGradDensityTotalPrimeValues = gradRhoPrimeValues[0];
            const auto &tempGradDensityMagPrimeValues   = gradRhoPrimeValues[1];
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityPrimeValue[2 * q + 0] =
                  0.5 * (tempDensityTotalPrimeValues[iElemCount *
                                                       numberQuadraturePoints +
                                                     q] +
                         tempDensityMagPrimeValues[iElemCount *
                                                     numberQuadraturePoints +
                                                   q]);
                densityPrimeValue[2 * q + 1] =
                  0.5 * (tempDensityTotalPrimeValues[iElemCount *
                                                       numberQuadraturePoints +
                                                     q] -
                         tempDensityMagPrimeValues[iElemCount *
                                                     numberQuadraturePoints +
                                                   q]);
                for (unsigned int idim = 0; idim < 3; ++idim)
                  {
                    gradDensityPrimeValue[6 * q + idim] =
                      0.5 *
                      (tempGradDensityTotalPrimeValues
                         [3 * iElemCount * numberQuadraturePoints + 3 * q +
                          idim] +
                       tempGradDensityMagPrimeValues[3 * iElemCount *
                                                       numberQuadraturePoints +
                                                     3 * q + idim]);
                    gradDensityPrimeValue[6 * q + 3 + idim] =
                      0.5 *
                      (tempGradDensityTotalPrimeValues
                         [3 * iElemCount * numberQuadraturePoints + 3 * q +
                          idim] -
                       tempGradDensityMagPrimeValues[3 * iElemCount *
                                                       numberQuadraturePoints +
                                                     3 * q + idim]);
                  }
              }

            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[2 * q] -= 1.0 * lambda * densityPrimeValue[2 * q];
                densityValue[2 * q + 1] -=
                  1.0 * lambda * densityPrimeValue[2 * q + 1];
                gradDensityValue[6 * q + 0] -=
                  1.0 * lambda * gradDensityPrimeValue[6 * q + 0];
                gradDensityValue[6 * q + 1] -=
                  1.0 * lambda * gradDensityPrimeValue[6 * q + 1];
                gradDensityValue[6 * q + 2] -=
                  1.0 * lambda * gradDensityPrimeValue[6 * q + 2];
                gradDensityValue[6 * q + 3] -=
                  1.0 * lambda * gradDensityPrimeValue[6 * q + 3];
                gradDensityValue[6 * q + 4] -=
                  1.0 * lambda * gradDensityPrimeValue[6 * q + 4];
                gradDensityValue[6 * q + 5] -=
                  1.0 * lambda * gradDensityPrimeValue[6 * q + 5];
              }


            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                const double gradRhoX1 = gradDensityValue[6 * q + 0];
                const double gradRhoY1 = gradDensityValue[6 * q + 1];
                const double gradRhoZ1 = gradDensityValue[6 * q + 2];
                const double gradRhoX2 = gradDensityValue[6 * q + 3];
                const double gradRhoY2 = gradDensityValue[6 * q + 4];
                const double gradRhoZ2 = gradDensityValue[6 * q + 5];

                sigmaValue[3 * q + 0] = gradRhoX1 * gradRhoX1 +
                                        gradRhoY1 * gradRhoY1 +
                                        gradRhoZ1 * gradRhoZ1;
                sigmaValue[3 * q + 1] = gradRhoX1 * gradRhoX2 +
                                        gradRhoY1 * gradRhoY2 +
                                        gradRhoZ1 * gradRhoZ2;
                sigmaValue[3 * q + 2] = gradRhoX2 * gradRhoX2 +
                                        gradRhoY2 * gradRhoY2 +
                                        gradRhoZ2 * gradRhoZ2;
              }

            std::map<rhoDataAttributes, const std::vector<double> *> rhoData;

            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerExchangeEnergy;
            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerCorrEnergy;


            rhoData[rhoDataAttributes::values]         = &densityValue;
            rhoData[rhoDataAttributes::sigmaGradValue] = &sigmaValue;

            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derExchEnergyWithDensityVal;
            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
                &derExchEnergyWithSigma;

            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derCorrEnergyWithDensityVal;
            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
                &derCorrEnergyWithSigma;

            dftPtr->d_excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
              numberQuadraturePoints,
              rhoData,
              outputDerExchangeEnergy,
              outputDerCorrEnergy);

            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                d_vEffJxW[iElemCount * numberQuadraturePoints + q] -=
                  8.0 *
                  (derExchEnergyWithDensityVal[2 * q + spinIndex] +
                   derCorrEnergyWithDensityVal[2 * q + spinIndex]) *
                  d_basisOperationsPtrHost
                    ->JxWBasisData()[iElemCount * numberQuadraturePoints + q];
              }

            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                const double jxw =
                  d_basisOperationsPtrHost
                    ->JxWBasisData()[iElemCount * numberQuadraturePoints + q];
                const double gradRhoX =
                  gradDensityValue[6 * q + 0 + 3 * spinIndex];
                const double gradRhoY =
                  gradDensityValue[6 * q + 1 + 3 * spinIndex];
                const double gradRhoZ =
                  gradDensityValue[6 * q + 2 + 3 * spinIndex];
                const double gradRhoOtherX =
                  gradDensityValue[6 * q + 0 + 3 * (1 - spinIndex)];
                const double gradRhoOtherY =
                  gradDensityValue[6 * q + 1 + 3 * (1 - spinIndex)];
                const double gradRhoOtherZ =
                  gradDensityValue[6 * q + 2 + 3 * (1 - spinIndex)];
                const double term =
                  derExchEnergyWithSigma[3 * q + 2 * spinIndex] +
                  derCorrEnergyWithSigma[3 * q + 2 * spinIndex];
                const double termOff = derExchEnergyWithSigma[3 * q + 1] +
                                       derCorrEnergyWithSigma[3 * q + 1];

                d_derExcWithSigmaTimesGradRhoJxW[iElemCount *
                                                   numberQuadraturePoints * 3 +
                                                 3 * q] -=
                  8.0 * (term * gradRhoX + 0.5 * termOff * gradRhoOtherX) * jxw;
                d_derExcWithSigmaTimesGradRhoJxW[iElemCount *
                                                   numberQuadraturePoints * 3 +
                                                 3 * q + 1] -=
                  8.0 * (term * gradRhoY + 0.5 * termOff * gradRhoOtherY) * jxw;
                d_derExcWithSigmaTimesGradRhoJxW[iElemCount *
                                                   numberQuadraturePoints * 3 +
                                                 3 * q + 2] -=
                  8.0 * (term * gradRhoZ + 0.5 * termOff * gradRhoOtherZ) * jxw;
              }
            iElemCount++;
          } // if cellPtr->is_locally_owned() loop

      } // cell loop


    cellPtr =
      dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex)
        .begin_active();
    iElemCount = 0;
    for (; cellPtr != endcellPtr; ++cellPtr)
      {
        if (cellPtr->is_locally_owned())
          {
            const auto &tempDensityTotalValues     = rhoValues[0];
            const auto &tempDensityMagValues       = rhoValues[1];
            const auto &tempGradDensityTotalValues = gradRhoValues[0];
            const auto &tempGradDensityMagValues   = gradRhoValues[1];
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[2 * q + 0] =
                  0.5 *
                  (tempDensityTotalValues[iElemCount * numberQuadraturePoints +
                                          q] +
                   tempDensityMagValues[iElemCount * numberQuadraturePoints +
                                        q]);
                densityValue[2 * q + 1] =
                  0.5 *
                  (tempDensityTotalValues[iElemCount * numberQuadraturePoints +
                                          q] -
                   tempDensityMagValues[iElemCount * numberQuadraturePoints +
                                        q]);
                for (unsigned int idim = 0; idim < 3; ++idim)
                  {
                    gradDensityValue[6 * q + idim] =
                      0.5 *
                      (tempGradDensityTotalValues[3 * iElemCount *
                                                    numberQuadraturePoints +
                                                  3 * q + idim] +
                       tempGradDensityMagValues[3 * iElemCount *
                                                  numberQuadraturePoints +
                                                3 * q + idim]);
                    gradDensityValue[6 * q + 3 + idim] =
                      0.5 *
                      (tempGradDensityTotalValues[3 * iElemCount *
                                                    numberQuadraturePoints +
                                                  3 * q + idim] -
                       tempGradDensityMagValues[3 * iElemCount *
                                                  numberQuadraturePoints +
                                                3 * q + idim]);
                  }
              }


            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              phiPrimeValue[q] =
                phiPrimeValues[iElemCount * numberQuadraturePoints + q];

            if (dftPtr->d_dftParamsPtr->nonLinearCoreCorrection)
              {
                const std::vector<double> &temp2 =
                  rhoCoreValues.find(cellPtr->id())->second;

                const std::vector<double> &temp3 =
                  gradRhoCoreValues.find(cellPtr->id())->second;

                for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                  {
                    densityValue[2 * q] += temp2[q] / 2.0;
                    densityValue[2 * q + 1] += temp2[q] / 2.0;
                    gradDensityValue[6 * q + 0] += temp3[3 * q + 0] / 2.0;
                    gradDensityValue[6 * q + 1] += temp3[3 * q + 1] / 2.0;
                    gradDensityValue[6 * q + 2] += temp3[3 * q + 2] / 2.0;
                    gradDensityValue[6 * q + 3] += temp3[3 * q + 0] / 2.0;
                    gradDensityValue[6 * q + 4] += temp3[3 * q + 1] / 2.0;
                    gradDensityValue[6 * q + 5] += temp3[3 * q + 2] / 2.0;
                  }
              }


            const auto &tempDensityTotalPrimeValues     = rhoPrimeValues[0];
            const auto &tempDensityMagPrimeValues       = rhoPrimeValues[1];
            const auto &tempGradDensityTotalPrimeValues = gradRhoPrimeValues[0];
            const auto &tempGradDensityMagPrimeValues   = gradRhoPrimeValues[1];
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityPrimeValue[2 * q + 0] =
                  0.5 * (tempDensityTotalPrimeValues[iElemCount *
                                                       numberQuadraturePoints +
                                                     q] +
                         tempDensityMagPrimeValues[iElemCount *
                                                     numberQuadraturePoints +
                                                   q]);
                densityPrimeValue[2 * q + 1] =
                  0.5 * (tempDensityTotalPrimeValues[iElemCount *
                                                       numberQuadraturePoints +
                                                     q] -
                         tempDensityMagPrimeValues[iElemCount *
                                                     numberQuadraturePoints +
                                                   q]);
                for (unsigned int idim = 0; idim < 3; ++idim)
                  {
                    gradDensityPrimeValue[6 * q + idim] =
                      0.5 *
                      (tempGradDensityTotalPrimeValues
                         [3 * iElemCount * numberQuadraturePoints + 3 * q +
                          idim] +
                       tempGradDensityMagPrimeValues[3 * iElemCount *
                                                       numberQuadraturePoints +
                                                     3 * q + idim]);
                    gradDensityPrimeValue[6 * q + 3 + idim] =
                      0.5 *
                      (tempGradDensityTotalPrimeValues
                         [3 * iElemCount * numberQuadraturePoints + 3 * q +
                          idim] -
                       tempGradDensityMagPrimeValues[3 * iElemCount *
                                                       numberQuadraturePoints +
                                                     3 * q + idim]);
                  }
              }

            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[2 * q] -= 2.0 * lambda * densityPrimeValue[2 * q];
                densityValue[2 * q + 1] -=
                  2.0 * lambda * densityPrimeValue[2 * q + 1];
                gradDensityValue[6 * q + 0] -=
                  2.0 * lambda * gradDensityPrimeValue[6 * q + 0];
                gradDensityValue[6 * q + 1] -=
                  2.0 * lambda * gradDensityPrimeValue[6 * q + 1];
                gradDensityValue[6 * q + 2] -=
                  2.0 * lambda * gradDensityPrimeValue[6 * q + 2];
                gradDensityValue[6 * q + 3] -=
                  2.0 * lambda * gradDensityPrimeValue[6 * q + 3];
                gradDensityValue[6 * q + 4] -=
                  2.0 * lambda * gradDensityPrimeValue[6 * q + 4];
                gradDensityValue[6 * q + 5] -=
                  2.0 * lambda * gradDensityPrimeValue[6 * q + 5];
              }


            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                const double gradRhoX1 = gradDensityValue[6 * q + 0];
                const double gradRhoY1 = gradDensityValue[6 * q + 1];
                const double gradRhoZ1 = gradDensityValue[6 * q + 2];
                const double gradRhoX2 = gradDensityValue[6 * q + 3];
                const double gradRhoY2 = gradDensityValue[6 * q + 4];
                const double gradRhoZ2 = gradDensityValue[6 * q + 5];

                sigmaValue[3 * q + 0] = gradRhoX1 * gradRhoX1 +
                                        gradRhoY1 * gradRhoY1 +
                                        gradRhoZ1 * gradRhoZ1;
                sigmaValue[3 * q + 1] = gradRhoX1 * gradRhoX2 +
                                        gradRhoY1 * gradRhoY2 +
                                        gradRhoZ1 * gradRhoZ2;
                sigmaValue[3 * q + 2] = gradRhoX2 * gradRhoX2 +
                                        gradRhoY2 * gradRhoY2 +
                                        gradRhoZ2 * gradRhoZ2;
              }

            std::map<rhoDataAttributes, const std::vector<double> *> rhoData;

            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerExchangeEnergy;
            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerCorrEnergy;


            rhoData[rhoDataAttributes::values]         = &densityValue;
            rhoData[rhoDataAttributes::sigmaGradValue] = &sigmaValue;

            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derExchEnergyWithDensityVal;
            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
                &derExchEnergyWithSigma;

            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derCorrEnergyWithDensityVal;
            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
                &derCorrEnergyWithSigma;

            dftPtr->d_excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
              numberQuadraturePoints,
              rhoData,
              outputDerExchangeEnergy,
              outputDerCorrEnergy);


            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                d_vEffJxW[iElemCount * numberQuadraturePoints + q] +=
                  1.0 *
                  (derExchEnergyWithDensityVal[2 * q + spinIndex] +
                   derCorrEnergyWithDensityVal[2 * q + spinIndex]) *
                  d_basisOperationsPtrHost
                    ->JxWBasisData()[iElemCount * numberQuadraturePoints + q];

                d_vEffJxW[iElemCount * numberQuadraturePoints + q] *=
                  1.0 / 12.0 / lambda;
                d_vEffJxW[iElemCount * numberQuadraturePoints + q] +=
                  phiPrimeValue[q] *
                  d_basisOperationsPtrHost
                    ->JxWBasisData()[iElemCount * numberQuadraturePoints + q];
              }

            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                const double jxw =
                  d_basisOperationsPtrHost
                    ->JxWBasisData()[iElemCount * numberQuadraturePoints + q];
                const double gradRhoX =
                  gradDensityValue[6 * q + 0 + 3 * spinIndex];
                const double gradRhoY =
                  gradDensityValue[6 * q + 1 + 3 * spinIndex];
                const double gradRhoZ =
                  gradDensityValue[6 * q + 2 + 3 * spinIndex];
                const double gradRhoOtherX =
                  gradDensityValue[6 * q + 0 + 3 * (1 - spinIndex)];
                const double gradRhoOtherY =
                  gradDensityValue[6 * q + 1 + 3 * (1 - spinIndex)];
                const double gradRhoOtherZ =
                  gradDensityValue[6 * q + 2 + 3 * (1 - spinIndex)];
                const double term =
                  derExchEnergyWithSigma[3 * q + 2 * spinIndex] +
                  derCorrEnergyWithSigma[3 * q + 2 * spinIndex];
                const double termOff = derExchEnergyWithSigma[3 * q + 1] +
                                       derCorrEnergyWithSigma[3 * q + 1];

                d_derExcWithSigmaTimesGradRhoJxW[iElemCount *
                                                   numberQuadraturePoints * 3 +
                                                 3 * q] +=
                  1.0 * (term * gradRhoX + 0.5 * termOff * gradRhoOtherX) * jxw;
                d_derExcWithSigmaTimesGradRhoJxW[iElemCount *
                                                   numberQuadraturePoints * 3 +
                                                 3 * q + 1] +=
                  1.0 * (term * gradRhoY + 0.5 * termOff * gradRhoOtherY) * jxw;
                d_derExcWithSigmaTimesGradRhoJxW[iElemCount *
                                                   numberQuadraturePoints * 3 +
                                                 3 * q + 2] +=
                  1.0 * (term * gradRhoZ + 0.5 * termOff * gradRhoOtherZ) * jxw;

                d_derExcWithSigmaTimesGradRhoJxW[iElemCount *
                                                   numberQuadraturePoints * 3 +
                                                 3 * q] *= 1.0 / 12.0 / lambda;

                d_derExcWithSigmaTimesGradRhoJxW[iElemCount *
                                                   numberQuadraturePoints * 3 +
                                                 3 * q + 1] *=
                  1.0 / 12.0 / lambda;

                d_derExcWithSigmaTimesGradRhoJxW[iElemCount *
                                                   numberQuadraturePoints * 3 +
                                                 3 * q + 2] *=
                  1.0 / 12.0 / lambda;
              }
            iElemCount++;
          } // if cellPtr->is_locally_owned() loop

      } // cell loop
    d_vEffJxWDevice.resize(d_vEffJxW.size());
    d_vEffJxWDevice.copyFrom(d_vEffJxW);

    d_derExcWithSigmaTimesGradRhoJxWDevice.resize(
      d_derExcWithSigmaTimesGradRhoJxW.size());
    d_derExcWithSigmaTimesGradRhoJxWDevice.copyFrom(
      d_derExcWithSigmaTimesGradRhoJxW);
  }


  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::HX(
    distributedDeviceVec<dataTypes::number> &    src,
    distributedDeviceVec<dataTypes::numberFP32> &tempFloatArray,
    distributedDeviceVec<dataTypes::number> &    projectorKetTimesVector,
    const unsigned int                           localVectorSize,
    const unsigned int                           numberWaveFunctions,
    const bool                                   scaleFlag,
    const double                                 scalar,
    distributedDeviceVec<dataTypes::number> &    dst,
    const bool                                   doUnscalingSrc,
    const bool                                   singlePrecCommun,
    const bool onlyHPrimePartForFirstOrderDensityMatResponse)
  {
    const unsigned int n_ghosts =
      dftPtr->matrix_free_data
        .get_vector_partitioner(dftPtr->d_densityDofHandlerIndex)
        ->n_ghost_indices();
    const unsigned int localSize =
      dftPtr->matrix_free_data
        .get_vector_partitioner(dftPtr->d_densityDofHandlerIndex)
        ->local_size();
    const unsigned int totalSize = localSize + n_ghosts;
    //
    // scale src vector with M^{-1/2}
    //
    dftfe::utils::deviceKernelsGeneric::stridedBlockScale(
      numberWaveFunctions,
      localVectorSize,
      scalar,
      d_invSqrtMassVectorDevice.begin(),
      src.begin());


    if (scaleFlag)
      {
        dftfe::utils::deviceKernelsGeneric::stridedBlockScale(
          numberWaveFunctions,
          localVectorSize,
          1.0,
          d_sqrtMassVectorDevice.begin(),
          dst.begin());
      }


    if (singlePrecCommun)
      {
        dftfe::utils::deviceKernelsGeneric::copyValueType1ArrToValueType2Arr(
          numberWaveFunctions * localSize, src.begin(), tempFloatArray.begin());
        tempFloatArray.updateGhostValues();

        if (n_ghosts != 0)
          dftfe::utils::deviceKernelsGeneric::copyValueType1ArrToValueType2Arr(
            numberWaveFunctions * n_ghosts,
            tempFloatArray.begin() + localSize * numberWaveFunctions,
            src.begin() + localSize * numberWaveFunctions);
      }
    else
      {
        src.updateGhostValues();
      }
    getOverloadedConstraintMatrix()->distribute(src);

    computeLocalHamiltonianTimesX(
      src.begin(),
      numberWaveFunctions,
      dst.begin(),
      onlyHPrimePartForFirstOrderDensityMatResponse);

    // H^{nloc}*M^{-1/2}*X
    if (dftPtr->d_dftParamsPtr->isPseudopotential &&
        (d_totalNonlocalElemsPseudo > 0) &&
        !onlyHPrimePartForFirstOrderDensityMatResponse)
      {
        computeNonLocalHamiltonianTimesX(src.begin(),
                                         projectorKetTimesVector,
                                         numberWaveFunctions,
                                         dst.begin());
      }

    if (std::is_same<dataTypes::number, std::complex<double>>::value)
      getOverloadedConstraintMatrix()->distribute_slave_to_master(
        dst, d_tempRealVec.begin(), d_tempImagVec.begin());
    else
      getOverloadedConstraintMatrix()->distribute_slave_to_master(dst);


    src.zeroOutGhosts();
    if (singlePrecCommun)
      {
        dftfe::utils::deviceKernelsGeneric::copyValueType1ArrToValueType2Arr(
          numberWaveFunctions * totalSize, dst.begin(), tempFloatArray.begin());

        tempFloatArray.accumulateAddLocallyOwned();

        // copy locally owned processor boundary nodes only to dst vector
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
        copyFloatArrToDoubleArrLocallyOwned<<<
          (numberWaveFunctions + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
            dftfe::utils::DEVICE_BLOCK_SIZE * localSize,
          dftfe::utils::DEVICE_BLOCK_SIZE>>>(
          numberWaveFunctions,
          localSize,
          dftfe::utils::makeDataTypeDeviceCompatible(tempFloatArray.begin()),
          d_locallyOwnedProcBoundaryNodesVectorDevice.begin(),
          dftfe::utils::makeDataTypeDeviceCompatible(dst.begin()));
#elif DFTFE_WITH_DEVICE_LANG_HIP
        hipLaunchKernelGGL(
          copyFloatArrToDoubleArrLocallyOwned,
          (numberWaveFunctions + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
            dftfe::utils::DEVICE_BLOCK_SIZE * localSize,
          dftfe::utils::DEVICE_BLOCK_SIZE,
          0,
          0,
          numberWaveFunctions,
          localSize,
          dftfe::utils::makeDataTypeDeviceCompatible(tempFloatArray.begin()),
          d_locallyOwnedProcBoundaryNodesVectorDevice.begin(),
          dftfe::utils::makeDataTypeDeviceCompatible(dst.begin()));
#endif

        dst.zeroOutGhosts();
      }
    else
      {
        dst.accumulateAddLocallyOwned();
      }

    //
    // M^{-1/2}*H*M^{-1/2}*X
    //
    dftfe::utils::deviceKernelsGeneric::stridedBlockScale(
      numberWaveFunctions,
      localVectorSize,
      1.0,
      d_invSqrtMassVectorDevice.begin(),
      dst.begin());



    //
    // unscale src M^{1/2}*X
    //
    if (doUnscalingSrc)
      dftfe::utils::deviceKernelsGeneric::stridedBlockScale(
        numberWaveFunctions,
        localVectorSize,
        1.0 / scalar,
        d_sqrtMassVectorDevice.begin(),
        src.begin());
  }



  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::HX(
    distributedDeviceVec<dataTypes::number> &src,
    distributedDeviceVec<dataTypes::number> &projectorKetTimesVector,
    const unsigned int                       localVectorSize,
    const unsigned int                       numberWaveFunctions,
    const bool                               scaleFlag,
    const double                             scalar,
    distributedDeviceVec<dataTypes::number> &dst,
    const bool                               doUnscalingSrc,
    const bool onlyHPrimePartForFirstOrderDensityMatResponse)
  {
    const unsigned int n_ghosts =
      dftPtr->matrix_free_data
        .get_vector_partitioner(dftPtr->d_densityDofHandlerIndex)
        ->n_ghost_indices();
    const unsigned int localSize =
      dftPtr->matrix_free_data
        .get_vector_partitioner(dftPtr->d_densityDofHandlerIndex)
        ->local_size();
    const unsigned int totalSize = localSize + n_ghosts;
    //
    // scale src vector with M^{-1/2}
    //
    dftfe::utils::deviceKernelsGeneric::stridedBlockScale(
      numberWaveFunctions,
      localVectorSize,
      scalar,
      d_invSqrtMassVectorDevice.begin(),
      src.begin());


    if (scaleFlag)
      {
        dftfe::utils::deviceKernelsGeneric::stridedBlockScale(
          numberWaveFunctions,
          localVectorSize,
          1.0,
          d_sqrtMassVectorDevice.begin(),
          dst.begin());
      }


    src.updateGhostValues();
    getOverloadedConstraintMatrix()->distribute(src);

    computeLocalHamiltonianTimesX(
      src.begin(),
      numberWaveFunctions,
      dst.begin(),
      onlyHPrimePartForFirstOrderDensityMatResponse);

    // H^{nloc}*M^{-1/2}*X
    if (dftPtr->d_dftParamsPtr->isPseudopotential &&
        (d_totalNonlocalElemsPseudo > 0) &&
        !onlyHPrimePartForFirstOrderDensityMatResponse)
      {
        computeNonLocalHamiltonianTimesX(src.begin(),
                                         projectorKetTimesVector,
                                         numberWaveFunctions,
                                         dst.begin());
      }

    if (std::is_same<dataTypes::number, std::complex<double>>::value)
      getOverloadedConstraintMatrix()->distribute_slave_to_master(
        dst, d_tempRealVec.begin(), d_tempImagVec.begin());
    else
      getOverloadedConstraintMatrix()->distribute_slave_to_master(dst);


    src.zeroOutGhosts();
    dst.accumulateAddLocallyOwned();

    //
    // M^{-1/2}*H*M^{-1/2}*X
    //
    dftfe::utils::deviceKernelsGeneric::stridedBlockScale(
      numberWaveFunctions,
      localVectorSize,
      1.0,
      d_invSqrtMassVectorDevice.begin(),
      dst.begin());

    //
    // unscale src M^{1/2}*X
    //
    if (doUnscalingSrc)
      dftfe::utils::deviceKernelsGeneric::stridedBlockScale(
        numberWaveFunctions,
        localVectorSize,
        1.0 / scalar,
        d_sqrtMassVectorDevice.begin(),
        src.begin());
  }


  // computePart1 and computePart2 are flags used by chebyshevFilter function to
  // perform overlap of computation and communication. When either computePart1
  // or computePart1 flags are set to true all communication calls are skipped
  // as they are directly called in chebyshevFilter. Only either of computePart1
  // or computePart2 can be set to true at one time. When computePart1 is set to
  // true distrubute, computeLocalHamiltonianTimesX, and first compute part of
  // nonlocalHX are performed before the control returns back to
  // chebyshevFilter. When computePart2 is set to true, the computations in
  // computePart1 are skipped and only computations performed are: second
  // compute part of nonlocalHX, assembly (only local processor), and
  // distribute_slave_to_master.
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::HXCheby(
    distributedDeviceVec<dataTypes::number> &    src,
    distributedDeviceVec<dataTypes::numberFP32> &tempFloatArray,
    distributedDeviceVec<dataTypes::number> &    projectorKetTimesVector,
    const unsigned int                           localVectorSize,
    const unsigned int                           numberWaveFunctions,
    distributedDeviceVec<dataTypes::number> &    dst,
    bool                                         chebMixedPrec,
    bool                                         computePart1,
    bool                                         computePart2)
  {
    const unsigned int n_ghosts =
      dftPtr->matrix_free_data
        .get_vector_partitioner(dftPtr->d_densityDofHandlerIndex)
        ->n_ghost_indices();
    const unsigned int localSize =
      dftPtr->matrix_free_data
        .get_vector_partitioner(dftPtr->d_densityDofHandlerIndex)
        ->local_size();
    const unsigned int totalSize = localSize + n_ghosts;

    if (!(computePart1 || computePart2))
      {
        if (chebMixedPrec)
          {
            dftfe::utils::deviceKernelsGeneric::
              copyValueType1ArrToValueType2Arr(numberWaveFunctions * localSize,
                                               src.begin(),
                                               tempFloatArray.begin());

            tempFloatArray.updateGhostValues();

            if (n_ghosts != 0)
              dftfe::utils::deviceKernelsGeneric::
                copyValueType1ArrToValueType2Arr(
                  numberWaveFunctions * n_ghosts,
                  tempFloatArray.begin() + localSize * numberWaveFunctions,
                  src.begin() + localSize * numberWaveFunctions);
          }
        else
          {
            src.updateGhostValues();
          }
      }

    if (!computePart2)
      getOverloadedConstraintMatrix()->distribute(src);


    if (!computePart2)
      computeLocalHamiltonianTimesX(src.begin(),
                                    numberWaveFunctions,
                                    dst.begin());


    // H^{nloc}*M^{-1/2}*X
    if (dftPtr->d_dftParamsPtr->isPseudopotential &&
        d_totalNonlocalElemsPseudo > 0)
      {
        computeNonLocalHamiltonianTimesX(src.begin(),
                                         projectorKetTimesVector,
                                         numberWaveFunctions,
                                         dst.begin(),
                                         computePart2,
                                         computePart1);
      }

    if (computePart1)
      return;


    if (std::is_same<dataTypes::number, std::complex<double>>::value)
      getOverloadedConstraintMatrix()->distribute_slave_to_master(
        dst, d_tempRealVec.begin(), d_tempImagVec.begin());
    else
      getOverloadedConstraintMatrix()->distribute_slave_to_master(dst);

    if (computePart2)
      return;

    src.zeroOutGhosts();

    if (chebMixedPrec)
      {
        dftfe::utils::deviceKernelsGeneric::copyValueType1ArrToValueType2Arr(
          numberWaveFunctions * totalSize, dst.begin(), tempFloatArray.begin());

        tempFloatArray.accumulateAddLocallyOwned();

        // copy locally owned processor boundary nodes only to dst vector
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
        copyFloatArrToDoubleArrLocallyOwned<<<
          (numberWaveFunctions + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
            dftfe::utils::DEVICE_BLOCK_SIZE * localSize,
          dftfe::utils::DEVICE_BLOCK_SIZE>>>(
          numberWaveFunctions,
          localSize,
          dftfe::utils::makeDataTypeDeviceCompatible(tempFloatArray.begin()),
          d_locallyOwnedProcBoundaryNodesVectorDevice.begin(),
          dftfe::utils::makeDataTypeDeviceCompatible(dst.begin()));
#elif DFTFE_WITH_DEVICE_LANG_HIP
        hipLaunchKernelGGL(
          copyFloatArrToDoubleArrLocallyOwned,
          (numberWaveFunctions + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
            dftfe::utils::DEVICE_BLOCK_SIZE * localSize,
          dftfe::utils::DEVICE_BLOCK_SIZE,
          0,
          0,
          numberWaveFunctions,
          localSize,
          dftfe::utils::makeDataTypeDeviceCompatible(tempFloatArray.begin()),
          d_locallyOwnedProcBoundaryNodesVectorDevice.begin(),
          dftfe::utils::makeDataTypeDeviceCompatible(dst.begin()));
#endif
        dst.zeroOutGhosts();
      }
    else
      {
        dst.accumulateAddLocallyOwned();
      }
  }


  // X^{T}*HConj*XConj
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::XtHX(
    const dataTypes::number *                        X,
    distributedDeviceVec<dataTypes::number> &        XBlock,
    distributedDeviceVec<dataTypes::number> &        HXBlock,
    distributedDeviceVec<dataTypes::number> &        projectorKetTimesVector,
    const unsigned int                               M,
    const unsigned int                               N,
    dftfe::utils::deviceBlasHandle_t &               handle,
    const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
    dftfe::ScaLAPACKMatrix<dataTypes::number> &      projHamPar,
    utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
    const bool onlyHPrimePartForFirstOrderDensityMatResponse)
  {
    std::unordered_map<unsigned int, unsigned int> globalToLocalColumnIdMap;
    std::unordered_map<unsigned int, unsigned int> globalToLocalRowIdMap;
    linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
      processGrid, projHamPar, globalToLocalRowIdMap, globalToLocalColumnIdMap);

    // band group parallelization data structures
    const unsigned int numberBandGroups =
      dealii::Utilities::MPI::n_mpi_processes(dftPtr->interBandGroupComm);
    const unsigned int bandGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(dftPtr->interBandGroupComm);
    std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
    dftUtils::createBandParallelizationIndices(dftPtr->interBandGroupComm,
                                               N,
                                               bandGroupLowHighPlusOneIndices);



    const unsigned int vectorsBlockSize =
      std::min(dftPtr->d_dftParamsPtr->wfcBlockSize, N);

    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::HOST_PINNED>
      projHamBlockHost;
    projHamBlockHost.resize(vectorsBlockSize * N, 0);
    std::memset(projHamBlockHost.begin(),
                0,
                vectorsBlockSize * N * sizeof(dataTypes::number));

    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      HXBlockFull(vectorsBlockSize * M, dataTypes::number(0.0));
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      projHamBlock(vectorsBlockSize * N, dataTypes::number(0.0));

    for (unsigned int jvec = 0; jvec < N; jvec += vectorsBlockSize)
      {
        // Correct block dimensions if block "goes off edge of" the matrix
        const unsigned int B = std::min(vectorsBlockSize, N - jvec);

        if ((jvec + B) <=
              bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
            (jvec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
          {
            const unsigned int chebyBlockSize =
              std::min(dftPtr->d_dftParamsPtr->chebyWfcBlockSize, N);

            for (unsigned int k = jvec; k < jvec + B; k += chebyBlockSize)
              {
                dftfe::utils::deviceKernelsGeneric::
                  stridedCopyToBlockConstantStride(
                    chebyBlockSize, N, M, k, X, XBlock.begin());

                // evaluate XBlock^{T} times H^{T} and store in HXBlock
                HXBlock.setValue(0);
                const bool   scaleFlag = false;
                const double scalar    = 1.0;
                HX(XBlock,
                   projectorKetTimesVector,
                   M,
                   chebyBlockSize,
                   scaleFlag,
                   scalar,
                   HXBlock,
                   false,
                   onlyHPrimePartForFirstOrderDensityMatResponse);

                dftfe::utils::deviceKernelsGeneric::
                  stridedCopyFromBlockConstantStride(B,
                                                     chebyBlockSize,
                                                     M,
                                                     k - jvec,
                                                     HXBlock.begin(),
                                                     HXBlockFull.begin());
              }

            // Comptute local XTrunc^{T}*HConj*XConj.
            const dataTypes::number alpha = dataTypes::number(1.0),
                                    beta  = dataTypes::number(0.0);
            const unsigned int D          = N - jvec;
            dftfe::utils::deviceBlasWrapper::gemm(
              handle,
              dftfe::utils::DEVICEBLAS_OP_N,
              std::is_same<dataTypes::number, std::complex<double>>::value ?
                dftfe::utils::DEVICEBLAS_OP_C :
                dftfe::utils::DEVICEBLAS_OP_T,
              D,
              B,
              M,
              &alpha,
              X + jvec,
              N,
              HXBlockFull.begin(),
              B,
              &beta,
              projHamBlock.begin(),
              D);

            dftfe::utils::deviceMemcpyD2H(
              projHamBlockHost.begin(),
              dftfe::utils::makeDataTypeDeviceCompatible(projHamBlock.begin()),
              D * B * sizeof(dataTypes::number));


            // Sum local projHamBlock across domain decomposition processors
            MPI_Allreduce(MPI_IN_PLACE,
                          projHamBlockHost.begin(),
                          D * B,
                          dataTypes::mpi_type_id(projHamBlockHost.begin()),
                          MPI_SUM,
                          mpi_communicator);

            // Copying only the lower triangular part to the ScaLAPACK projected
            // Hamiltonian matrix
            if (processGrid->is_process_active())
              for (unsigned int j = 0; j < B; ++j)
                if (globalToLocalColumnIdMap.find(j + jvec) !=
                    globalToLocalColumnIdMap.end())
                  {
                    const unsigned int localColumnId =
                      globalToLocalColumnIdMap[j + jvec];
                    for (unsigned int i = j + jvec; i < N; ++i)
                      {
                        std::unordered_map<unsigned int, unsigned int>::iterator
                          it = globalToLocalRowIdMap.find(i);
                        if (it != globalToLocalRowIdMap.end())
                          projHamPar.local_el(it->second, localColumnId) =
                            projHamBlockHost[j * D + i - jvec];
                      }
                  }

          } // band parallelization
      }


    if (numberBandGroups > 1)
      {
        MPI_Barrier(dftPtr->interBandGroupComm);
        linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
          processGrid, projHamPar, dftPtr->interBandGroupComm);
      }
  }

  // X^{T}*HConj*XConj  with overlap of computation and
  // communication
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    XtHXOverlapComputeCommun(
      const dataTypes::number *                        X,
      distributedDeviceVec<dataTypes::number> &        XBlock,
      distributedDeviceVec<dataTypes::number> &        HXBlock,
      distributedDeviceVec<dataTypes::number> &        projectorKetTimesVector,
      const unsigned int                               M,
      const unsigned int                               N,
      dftfe::utils::deviceBlasHandle_t &               handle,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      projHamPar,
      utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
      const bool onlyHPrimePartForFirstOrderDensityMatResponse)
  {
    /////////////PSEUDO CODE for the implementation below for Overlapping
    /// compute and communication/////////////////
    //
    // In the algorithm below the communication and computation of two
    // consecutive blocks of wavefunctions: block i and block i+1 are
    // overlapped.
    // ----------------------------------------------------------
    // CMP denotes computuation of X^{T} times HXBlock
    // COP denotes Device->CPU copy of X^{T} times HXBlock
    // COM denotes blocking MPI_Allreduce on X^{T}HXBlock and copy to scalapack
    // matrix
    // ----------------------------------------------------------
    // Two Device streams are created: compute and copy
    // CMP is performed in compute Device stream and COP is performed in copy
    // Device stream. COP for a block can only start after the CMP for that
    // block in the compute stream is completed. COM is performed for a block
    // only after COP even for that block is completed.
    //
    // In a blocked loop do:
    // 1) [CMP] Call compute on first block (edge case only for first iteration)
    // 2) Wait for CMP event for current block to be completed.
    // 3) Swap current and next block memory (all iterations except edge case)
    // 4) [COP] Call copy on current block
    // 5) [CMP] Call compute on next block
    // 6) Wait for COP event for current block to be completed
    // 7) [COM] Perform blocking MPI_Allreduce on curent block and copy to
    // scalapack matrix
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    std::unordered_map<unsigned int, unsigned int> globalToLocalColumnIdMap;
    std::unordered_map<unsigned int, unsigned int> globalToLocalRowIdMap;
    linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
      processGrid, projHamPar, globalToLocalRowIdMap, globalToLocalColumnIdMap);

    // band group parallelization data structures
    const unsigned int numberBandGroups =
      dealii::Utilities::MPI::n_mpi_processes(dftPtr->interBandGroupComm);
    const unsigned int bandGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(dftPtr->interBandGroupComm);
    std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
    dftUtils::createBandParallelizationIndices(dftPtr->interBandGroupComm,
                                               N,
                                               bandGroupLowHighPlusOneIndices);



    const unsigned int vectorsBlockSize =
      std::min(dftPtr->d_dftParamsPtr->wfcBlockSize, N);
    const unsigned int numberBlocks = N / vectorsBlockSize;

    // create separate Device streams for Device->CPU copy and computation
    dftfe::utils::deviceStream_t streamCompute, streamDataMove;
    dftfe::utils::deviceStreamCreate(&streamCompute);
    dftfe::utils::deviceStreamCreate(&streamDataMove);

    // attach deviceblas handle to compute stream
    dftfe::utils::deviceBlasWrapper::setStream(handle, streamCompute);

    // create array of compute and copy events on Devices
    // for all the blocks. These are required for synchronization
    // between compute, copy and communication as discussed above in the
    // pseudo code
    dftfe::utils::deviceEvent_t computeEvents[numberBlocks];
    dftfe::utils::deviceEvent_t copyEvents[numberBlocks];

    for (int i = 0; i < numberBlocks; ++i)
      {
        dftfe::utils::deviceEventCreate(&computeEvents[i]);
        dftfe::utils::deviceEventCreate(&copyEvents[i]);
      }

    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::HOST_PINNED>
      projHamBlockHost;
    projHamBlockHost.resize(vectorsBlockSize * N, 0);
    std::memset(projHamBlockHost.begin(),
                0,
                vectorsBlockSize * N * sizeof(dataTypes::number));

    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      HXBlockFull(vectorsBlockSize * M, dataTypes::number(0.0));
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      projHamBlock(vectorsBlockSize * N, dataTypes::number(0.0));
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      projHamBlockNext(vectorsBlockSize * N, dataTypes::number(0.0));

    dftfe::utils::MemoryStorage<dataTypes::numberValueType,
                                dftfe::utils::MemorySpace::DEVICE>
      tempReal;
    dftfe::utils::MemoryStorage<dataTypes::numberValueType,
                                dftfe::utils::MemorySpace::DEVICE>
      tempImag;
    if (std::is_same<dataTypes::number, std::complex<double>>::value)
      {
        tempReal.resize(vectorsBlockSize * N, 0);
        tempImag.resize(vectorsBlockSize * N, 0);
      }

    unsigned int blockCount = 0;
    for (unsigned int jvec = 0; jvec < N; jvec += vectorsBlockSize)
      {
        // Correct block dimensions if block "goes off edge of" the matrix
        const unsigned int B = std::min(vectorsBlockSize, N - jvec);

        if ((jvec + B) <=
              bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
            (jvec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
          {
            const unsigned int chebyBlockSize =
              std::min(dftPtr->d_dftParamsPtr->chebyWfcBlockSize, N);

            const dataTypes::number alpha = dataTypes::number(1.0),
                                    beta  = dataTypes::number(0.0);
            const unsigned int D          = N - jvec;

            // handle edge case for the first block or the first block in the
            // band group in case of band parallelization
            if (jvec == bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
              {
                // compute HXBlockFull in an inner loop over blocks of B
                // wavefunction vectors
                for (unsigned int k = jvec; k < jvec + B; k += chebyBlockSize)
                  {
                    dftfe::utils::deviceKernelsGeneric::
                      stridedCopyToBlockConstantStride(
                        chebyBlockSize, N, M, k, X, XBlock.begin());

                    // evaluate H times XBlock^{T} and store in HXBlock^{T}
                    HXBlock.setValue(0);
                    const bool   scaleFlag = false;
                    const double scalar    = 1.0;
                    HX(XBlock,
                       projectorKetTimesVector,
                       M,
                       chebyBlockSize,
                       scaleFlag,
                       scalar,
                       HXBlock,
                       false,
                       onlyHPrimePartForFirstOrderDensityMatResponse);

                    dftfe::utils::deviceKernelsGeneric::
                      stridedCopyFromBlockConstantStride(B,
                                                         chebyBlockSize,
                                                         M,
                                                         k - jvec,
                                                         HXBlock.begin(),
                                                         HXBlockFull.begin());
                  }

                // evalute X^{T} times HXBlock
                dftfe::utils::deviceBlasWrapper::gemm(
                  handle,
                  dftfe::utils::DEVICEBLAS_OP_N,
                  std::is_same<dataTypes::number, std::complex<double>>::value ?
                    dftfe::utils::DEVICEBLAS_OP_C :
                    dftfe::utils::DEVICEBLAS_OP_T,
                  D,
                  B,
                  M,
                  &alpha,
                  X + jvec,
                  N,
                  HXBlockFull.begin(),
                  B,
                  &beta,
                  projHamBlock.begin(),
                  D);

                // record completion of compute for first block
                dftfe::utils::deviceEventRecord(computeEvents[blockCount],
                                                streamCompute);
              }


            // Before swap host thread needs to wait till compute on
            // currentblock is over. Since swap occurs on the null stream, any
            // future calls in the streamDataMove will only occur after both the
            // compute on currentblock and swap is over. Note that at this point
            // there is nothing queued in the streamDataMove as all previous
            // operations in that stream are over.
            if ((dftfe::utils::deviceEventSynchronize(
                   computeEvents[blockCount]) == dftfe::utils::deviceSuccess) &&
                (jvec > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId]))
              projHamBlock.swap(projHamBlockNext);

            const unsigned int jvecNew = jvec + vectorsBlockSize;
            const unsigned int DNew    = N - jvecNew;

            // start computations on the next block
            if (jvecNew <
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1])
              {
                for (unsigned int k = jvecNew; k < jvecNew + B;
                     k += chebyBlockSize)
                  {
                    dftfe::utils::deviceKernelsGeneric::
                      stridedCopyToBlockConstantStride(
                        chebyBlockSize, N, M, k, X, XBlock.begin());

                    // evaluate H times XBlock^{T} and store in HXBlock^{T}
                    HXBlock.setValue(0);
                    const bool   scaleFlag = false;
                    const double scalar    = 1.0;
                    HX(XBlock,
                       projectorKetTimesVector,
                       M,
                       chebyBlockSize,
                       scaleFlag,
                       scalar,
                       HXBlock,
                       false,
                       onlyHPrimePartForFirstOrderDensityMatResponse);

                    dftfe::utils::deviceKernelsGeneric::
                      stridedCopyFromBlockConstantStride(B,
                                                         chebyBlockSize,
                                                         M,
                                                         k - jvecNew,
                                                         HXBlock.begin(),
                                                         HXBlockFull.begin());
                  }

                // evalute X^{T} times HXBlock
                dftfe::utils::deviceBlasWrapper::gemm(
                  handle,
                  dftfe::utils::DEVICEBLAS_OP_N,
                  std::is_same<dataTypes::number, std::complex<double>>::value ?
                    dftfe::utils::DEVICEBLAS_OP_C :
                    dftfe::utils::DEVICEBLAS_OP_T,
                  DNew,
                  B,
                  M,
                  &alpha,
                  X + jvecNew,
                  N,
                  HXBlockFull.begin(),
                  B,
                  &beta,
                  projHamBlockNext.begin(),
                  DNew);

                // record completion of compute for next block
                dftfe::utils::deviceEventRecord(computeEvents[blockCount + 1],
                                                streamCompute);
              }

            if (dftPtr->d_dftParamsPtr->useDeviceDirectAllReduce)
              {
                // Sum local projHamBlock across domain decomposition processors
                if (std::is_same<dataTypes::number,
                                 std::complex<double>>::value)
                  {
                    devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                      projHamBlock.begin(),
                      projHamBlock.begin(),
                      D * B,
                      tempReal.begin(),
                      tempImag.begin(),
                      streamDataMove);
                  }
                else
                  devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                    projHamBlock.begin(),
                    projHamBlock.begin(),
                    D * B,
                    streamDataMove);
              }

            dftfe::utils::deviceMemcpyAsyncD2H(
              projHamBlockHost.begin(),
              dftfe::utils::makeDataTypeDeviceCompatible(projHamBlock.begin()),
              D * B * sizeof(dataTypes::number),
              streamDataMove);

            // record completion of Device->CPU copy for current block
            dftfe::utils::deviceEventRecord(copyEvents[blockCount],
                                            streamDataMove);

            // Check that Device->CPU on the current block has been completed.
            // If completed, perform blocking MPI commmunication on the current
            // block and copy to ScaLAPACK matrix
            if (dftfe::utils::deviceEventSynchronize(copyEvents[blockCount]) ==
                dftfe::utils::deviceSuccess)
              {
                // Sum local projHamBlock across domain decomposition processors
                if (!dftPtr->d_dftParamsPtr->useDeviceDirectAllReduce)
                  MPI_Allreduce(MPI_IN_PLACE,
                                projHamBlockHost.begin(),
                                D * B,
                                dataTypes::mpi_type_id(
                                  projHamBlockHost.begin()),
                                MPI_SUM,
                                mpi_communicator);

                // Copying only the lower triangular part to the ScaLAPACK
                // projected Hamiltonian matrix
                if (processGrid->is_process_active())
                  for (unsigned int j = 0; j < B; ++j)
                    if (globalToLocalColumnIdMap.find(j + jvec) !=
                        globalToLocalColumnIdMap.end())
                      {
                        const unsigned int localColumnId =
                          globalToLocalColumnIdMap[j + jvec];
                        for (unsigned int i = j + jvec; i < N; ++i)
                          {
                            std::unordered_map<unsigned int,
                                               unsigned int>::iterator it =
                              globalToLocalRowIdMap.find(i);
                            if (it != globalToLocalRowIdMap.end())
                              projHamPar.local_el(it->second, localColumnId) =
                                projHamBlockHost[j * D + i - jvec];
                          }
                      }
              }

          } // band parallelization
        blockCount += 1;
      }

    // return deviceblas handle to default stream
    dftfe::utils::deviceBlasWrapper::setStream(handle, NULL);

    for (int i = 0; i < numberBlocks; ++i)
      {
        dftfe::utils::deviceEventDestroy(computeEvents[i]);
        dftfe::utils::deviceEventDestroy(copyEvents[i]);
      }

    dftfe::utils::deviceStreamDestroy(streamCompute);
    dftfe::utils::deviceStreamDestroy(streamDataMove);

    if (numberBandGroups > 1)
      {
        MPI_Barrier(dftPtr->interBandGroupComm);
        linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
          processGrid, projHamPar, dftPtr->interBandGroupComm);
      }
  }


  // X^{T}*HConj*XConj  (Xc denotes complex conjugate)
  /////////////PSEUDO CODE for the implementation below for Overlapping compute
  /// and communication/////////////////
  //
  // In the algorithm below the communication and computation of two consecutive
  // blocks of wavefunctions: block i and block i+1 are overlapped.
  // ----------------------------------------------------------
  // CMP denotes computuation of X^{T} times HXBlock
  // COP denotes Device->CPU copy of X^{T} times HXBlock
  // COM denotes blocking MPI_Allreduce on X^{T}HXBlock and copy to scalapack
  // matrix
  // ----------------------------------------------------------
  // Two Device streams are created: compute and copy
  // CMP is performed in compute Device stream and COP is performed in copy
  // Device stream. COP for a block can only start after the CMP for that block
  // in the compute stream is completed. COM is performed for a block only after
  // COP even for that block is completed.
  //
  // In a blocked loop do:
  // 1) [CMP] Call compute on first block (edge case only for first iteration)
  // 2) Wait for CMP event for current block to be completed.
  // 3) Swap current and next block memory (all iterations except edge case)
  // 4) [COP] Call copy on current block
  // 5) [CMP] Call compute on next block
  // 6) Wait for COP event for current block to be completed
  // 7) [COM] Perform blocking MPI_Allreduce on curent block and copy to
  // scalapack matrix
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    XtHXMixedPrecOverlapComputeCommun(
      const dataTypes::number *                        X,
      distributedDeviceVec<dataTypes::number> &        XBlock,
      distributedDeviceVec<dataTypes::numberFP32> &    tempFloatBlock,
      distributedDeviceVec<dataTypes::number> &        HXBlock,
      distributedDeviceVec<dataTypes::number> &        projectorKetTimesVector,
      const unsigned int                               M,
      const unsigned int                               N,
      const unsigned int                               Noc,
      dftfe::utils::deviceBlasHandle_t &               handle,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      projHamPar,
      utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
      const bool onlyHPrimePartForFirstOrderDensityMatResponse)
  {
    std::unordered_map<unsigned int, unsigned int> globalToLocalColumnIdMap;
    std::unordered_map<unsigned int, unsigned int> globalToLocalRowIdMap;
    linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
      processGrid, projHamPar, globalToLocalRowIdMap, globalToLocalColumnIdMap);

    // band group parallelization data structures
    const unsigned int numberBandGroups =
      dealii::Utilities::MPI::n_mpi_processes(dftPtr->interBandGroupComm);
    const unsigned int bandGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(dftPtr->interBandGroupComm);
    std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
    dftUtils::createBandParallelizationIndices(dftPtr->interBandGroupComm,
                                               N,
                                               bandGroupLowHighPlusOneIndices);


    const unsigned int vectorsBlockSize =
      std::min(dftPtr->d_dftParamsPtr->wfcBlockSize, N);

    const unsigned int numberBlocks = N / vectorsBlockSize;

    // create device compute and copy streams
    dftfe::utils::deviceStream_t streamCompute, streamDataMove;
    dftfe::utils::deviceStreamCreate(&streamCompute);
    dftfe::utils::deviceStreamCreate(&streamDataMove);

    // attach deviceblas handle to compute stream
    dftfe::utils::deviceBlasWrapper::setStream(handle, streamCompute);

    // create array of compute and copy events on Devices
    // for all the blocks. These are required for synchronization
    // between compute, copy and communication as discussed above in the
    // pseudo code
    dftfe::utils::deviceEvent_t computeEvents[numberBlocks];
    dftfe::utils::deviceEvent_t copyEvents[numberBlocks];

    for (int i = 0; i < numberBlocks; ++i)
      {
        dftfe::utils::deviceEventCreate(&computeEvents[i]);
        dftfe::utils::deviceEventCreate(&copyEvents[i]);
      }

    dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                dftfe::utils::MemorySpace::DEVICE>
      XFP32(M * N, dataTypes::numberFP32(0.0));

    dftfe::utils::deviceKernelsGeneric::copyValueType1ArrToValueType2Arr(
      N * M, X, XFP32.begin());

    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::HOST_PINNED>
      projHamBlockHost;
    projHamBlockHost.resize(vectorsBlockSize * N, 0);
    std::memset(projHamBlockHost.begin(),
                0,
                vectorsBlockSize * N * sizeof(dataTypes::number));

    dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                dftfe::utils::MemorySpace::HOST_PINNED>
      projHamBlockHostFP32;
    projHamBlockHostFP32.resize(vectorsBlockSize * N, 0);
    std::memset(projHamBlockHostFP32.begin(),
                0,
                vectorsBlockSize * N * sizeof(dataTypes::numberFP32));

    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      HXBlockFull(vectorsBlockSize * M, dataTypes::number(0.0));
    dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                dftfe::utils::MemorySpace::DEVICE>
      HXBlockFullFP32(vectorsBlockSize * M, dataTypes::numberFP32(0.0));
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      projHamBlock(vectorsBlockSize * N, dataTypes::number(0.0));
    dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                dftfe::utils::MemorySpace::DEVICE>
      projHamBlockFP32(vectorsBlockSize * N, dataTypes::numberFP32(0.0));
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      projHamBlockNext(vectorsBlockSize * N, dataTypes::number(0.0));
    dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                dftfe::utils::MemorySpace::DEVICE>
      projHamBlockFP32Next(vectorsBlockSize * N, dataTypes::numberFP32(0.0));


    dftfe::utils::MemoryStorage<dataTypes::numberValueType,
                                dftfe::utils::MemorySpace::DEVICE>
      tempReal;
    dftfe::utils::MemoryStorage<dataTypes::numberValueType,
                                dftfe::utils::MemorySpace::DEVICE>
      tempImag;

    dftfe::utils::MemoryStorage<dataTypes::numberFP32ValueType,
                                dftfe::utils::MemorySpace::DEVICE>
      tempRealFP32;
    dftfe::utils::MemoryStorage<dataTypes::numberFP32ValueType,
                                dftfe::utils::MemorySpace::DEVICE>
      tempImagFP32;
    if (std::is_same<dataTypes::number, std::complex<double>>::value)
      {
        tempReal.resize(vectorsBlockSize * N, 0);
        tempImag.resize(vectorsBlockSize * N, 0);
        tempRealFP32.resize(vectorsBlockSize * N, 0);
        tempImagFP32.resize(vectorsBlockSize * N, 0);
      }

    unsigned int blockCount = 0;
    for (unsigned int jvec = 0; jvec < N; jvec += vectorsBlockSize)
      {
        // Correct block dimensions if block "goes off edge of" the matrix
        const unsigned int B = std::min(vectorsBlockSize, N - jvec);

        if ((jvec + B) <=
              bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
            (jvec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
          {
            const unsigned int chebyBlockSize =
              std::min(dftPtr->d_dftParamsPtr->chebyWfcBlockSize, N);

            const dataTypes::number alpha         = dataTypes::number(1.0),
                                    beta          = dataTypes::number(0.0);
            const dataTypes::numberFP32 alphaFP32 = dataTypes::numberFP32(1.0),
                                        betaFP32  = dataTypes::numberFP32(0.0);
            const unsigned int D                  = N - jvec;

            // handle edge case for the first block or the first block in the
            // band group in case of band parallelization
            if (jvec == bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
              {
                // compute HXBlockFull or HXBlockFullFP32 in an inner loop over
                // blocks of B wavefunction vectors
                for (unsigned int k = jvec; k < jvec + B; k += chebyBlockSize)
                  {
                    dftfe::utils::deviceKernelsGeneric::
                      stridedCopyToBlockConstantStride(
                        chebyBlockSize, N, M, k, X, XBlock.begin());

                    // evaluate H times XBlock^{T} and store in HXBlock^{T}
                    HXBlock.setValue(0);
                    const bool   scaleFlag = false;
                    const double scalar    = 1.0;
                    if (jvec + B > Noc)
                      HX(XBlock,
                         projectorKetTimesVector,
                         M,
                         chebyBlockSize,
                         scaleFlag,
                         scalar,
                         HXBlock,
                         false,
                         onlyHPrimePartForFirstOrderDensityMatResponse);
                    else
                      HX(XBlock,
                         tempFloatBlock,
                         projectorKetTimesVector,
                         M,
                         chebyBlockSize,
                         scaleFlag,
                         scalar,
                         HXBlock,
                         false,
                         true,
                         onlyHPrimePartForFirstOrderDensityMatResponse);

                    if (jvec + B > Noc)
                      dftfe::utils::deviceKernelsGeneric::
                        stridedCopyFromBlockConstantStride(B,
                                                           chebyBlockSize,
                                                           M,
                                                           k - jvec,
                                                           HXBlock.begin(),
                                                           HXBlockFull.begin());
                    else
                      dftfe::utils::deviceKernelsGeneric::
                        stridedCopyFromBlockConstantStride(
                          B,
                          chebyBlockSize,
                          M,
                          k - jvec,
                          HXBlock.begin(),
                          HXBlockFullFP32.begin());
                  }

                // evaluate X^{T} times HXBlockFullConj or XFP32^{T} times
                // HXBlockFullFP32Conj
                if (jvec + B > Noc)
                  dftfe::utils::deviceBlasWrapper::gemm(
                    handle,
                    dftfe::utils::DEVICEBLAS_OP_N,
                    std::is_same<dataTypes::number,
                                 std::complex<double>>::value ?
                      dftfe::utils::DEVICEBLAS_OP_C :
                      dftfe::utils::DEVICEBLAS_OP_T,
                    D,
                    B,
                    M,
                    &alpha,
                    X + jvec,
                    N,
                    HXBlockFull.begin(),
                    B,
                    &beta,
                    projHamBlock.begin(),
                    D);
                else
                  dftfe::utils::deviceBlasWrapper::gemm(
                    handle,
                    dftfe::utils::DEVICEBLAS_OP_N,
                    std::is_same<dataTypes::numberFP32,
                                 std::complex<float>>::value ?
                      dftfe::utils::DEVICEBLAS_OP_C :
                      dftfe::utils::DEVICEBLAS_OP_T,
                    D,
                    B,
                    M,
                    &alphaFP32,
                    XFP32.begin() + jvec,
                    N,
                    HXBlockFullFP32.begin(),
                    B,
                    &betaFP32,
                    projHamBlockFP32.begin(),
                    D);

                // record completion of compute for next block
                dftfe::utils::deviceEventRecord(computeEvents[blockCount],
                                                streamCompute);
              }

            // Before swap host thread needs to wait till compute on
            // currentblock is over. Since swap occurs on the null stream, any
            // future calls in the streamDataMove will only occur after both the
            // compute on currentblock and swap is over. Note that at this point
            // there is nothing queued in the streamDataMove as all previous
            // operations in that stream are over.
            if ((dftfe::utils::deviceEventSynchronize(
                   computeEvents[blockCount]) == dftfe::utils::deviceSuccess) &&
                (jvec > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId]))
              {
                if (jvec + B > Noc)
                  projHamBlock.swap(projHamBlockNext);
                else
                  projHamBlockFP32.swap(projHamBlockFP32Next);
              }

            const unsigned int jvecNew = jvec + vectorsBlockSize;
            const unsigned int DNew    = N - jvecNew;

            if (jvecNew <
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1])
              {
                // compute HXBlockFull or HXBlockFullFP32 in an inner loop over
                // blocks of B wavefunction vectors
                for (unsigned int k = jvecNew; k < jvecNew + B;
                     k += chebyBlockSize)
                  {
                    dftfe::utils::deviceKernelsGeneric::
                      stridedCopyToBlockConstantStride(
                        chebyBlockSize, N, M, k, X, XBlock.begin());

                    // evaluate H times XBlock^{T} and store in HXBlock^{T}
                    HXBlock.setValue(0);
                    const bool   scaleFlag = false;
                    const double scalar    = 1.0;
                    if (jvecNew + B > Noc)
                      HX(XBlock,
                         projectorKetTimesVector,
                         M,
                         chebyBlockSize,
                         scaleFlag,
                         scalar,
                         HXBlock,
                         false,
                         onlyHPrimePartForFirstOrderDensityMatResponse);
                    else
                      HX(XBlock,
                         tempFloatBlock,
                         projectorKetTimesVector,
                         M,
                         chebyBlockSize,
                         scaleFlag,
                         scalar,
                         HXBlock,
                         false,
                         true,
                         onlyHPrimePartForFirstOrderDensityMatResponse);

                    if (jvecNew + B > Noc)
                      dftfe::utils::deviceKernelsGeneric::
                        stridedCopyFromBlockConstantStride(B,
                                                           chebyBlockSize,
                                                           M,
                                                           k - jvecNew,
                                                           HXBlock.begin(),
                                                           HXBlockFull.begin());
                    else
                      dftfe::utils::deviceKernelsGeneric::
                        stridedCopyFromBlockConstantStride(
                          B,
                          chebyBlockSize,
                          M,
                          k - jvecNew,
                          HXBlock.begin(),
                          HXBlockFullFP32.begin());
                  }

                // evaluate X^{T} times HXBlockFullConj or XFP32^{T} times
                // HXBlockFullFP32Conj
                if (jvecNew + B > Noc)
                  dftfe::utils::deviceBlasWrapper::gemm(
                    handle,
                    dftfe::utils::DEVICEBLAS_OP_N,
                    std::is_same<dataTypes::number,
                                 std::complex<double>>::value ?
                      dftfe::utils::DEVICEBLAS_OP_C :
                      dftfe::utils::DEVICEBLAS_OP_T,
                    DNew,
                    B,
                    M,
                    &alpha,
                    X + jvecNew,
                    N,
                    HXBlockFull.begin(),
                    B,
                    &beta,
                    projHamBlockNext.begin(),
                    DNew);
                else
                  dftfe::utils::deviceBlasWrapper::gemm(
                    handle,
                    dftfe::utils::DEVICEBLAS_OP_N,
                    std::is_same<dataTypes::numberFP32,
                                 std::complex<float>>::value ?
                      dftfe::utils::DEVICEBLAS_OP_C :
                      dftfe::utils::DEVICEBLAS_OP_T,
                    DNew,
                    B,
                    M,
                    &alphaFP32,
                    XFP32.begin() + jvecNew,
                    N,
                    HXBlockFullFP32.begin(),
                    B,
                    &betaFP32,
                    projHamBlockFP32Next.begin(),
                    DNew);

                // record completion of compute for next block
                dftfe::utils::deviceEventRecord(computeEvents[blockCount + 1],
                                                streamCompute);
              }

            if (dftPtr->d_dftParamsPtr->useDeviceDirectAllReduce)
              {
                if (jvec + B > Noc)
                  {
                    if (std::is_same<dataTypes::number,
                                     std::complex<double>>::value)
                      devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                        projHamBlock.begin(),
                        projHamBlock.begin(),
                        D * B,
                        tempReal.begin(),
                        tempImag.begin(),
                        streamDataMove);
                    else
                      devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                        projHamBlock.begin(),
                        projHamBlock.begin(),
                        D * B,
                        streamDataMove);
                  }
                else
                  {
                    if (std::is_same<dataTypes::number,
                                     std::complex<double>>::value)
                      devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                        projHamBlockFP32.begin(),
                        projHamBlockFP32.begin(),
                        D * B,
                        tempRealFP32.begin(),
                        tempImagFP32.begin(),
                        streamDataMove);
                    else
                      devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                        projHamBlockFP32.begin(),
                        projHamBlockFP32.begin(),
                        D * B,
                        streamDataMove);
                  }
              }

            if (jvec + B > Noc)
              dftfe::utils::deviceMemcpyAsyncD2H(
                projHamBlockHost.begin(),
                dftfe::utils::makeDataTypeDeviceCompatible(
                  projHamBlock.begin()),
                D * B * sizeof(dataTypes::number),
                streamDataMove);
            else
              dftfe::utils::deviceMemcpyAsyncD2H(
                projHamBlockHostFP32.begin(),
                dftfe::utils::makeDataTypeDeviceCompatible(
                  projHamBlockFP32.begin()),
                D * B * sizeof(dataTypes::numberFP32),
                streamDataMove);

            // record completion of Device->CPU copy for current block
            dftfe::utils::deviceEventRecord(copyEvents[blockCount],
                                            streamDataMove);

            // Check that Device->CPU on the current block has been completed.
            // If completed, perform blocking MPI commmunication on the current
            // block and copy to ScaLAPACK matrix
            if (dftfe::utils::deviceEventSynchronize(copyEvents[blockCount]) ==
                dftfe::utils::deviceSuccess)
              {
                if (jvec + B > Noc)
                  {
                    // Sum local projHamBlock across domain decomposition
                    // processors
                    if (!dftPtr->d_dftParamsPtr->useDeviceDirectAllReduce)
                      MPI_Allreduce(MPI_IN_PLACE,
                                    projHamBlockHost.begin(),
                                    D * B,
                                    dataTypes::mpi_type_id(
                                      projHamBlockHost.begin()),
                                    MPI_SUM,
                                    mpi_communicator);

                    // Copying only the lower triangular part to the ScaLAPACK
                    // projected Hamiltonian matrix
                    if (processGrid->is_process_active())
                      for (unsigned int j = 0; j < B; ++j)
                        if (globalToLocalColumnIdMap.find(j + jvec) !=
                            globalToLocalColumnIdMap.end())
                          {
                            const unsigned int localColumnId =
                              globalToLocalColumnIdMap[j + jvec];
                            for (unsigned int i = j + jvec; i < N; ++i)
                              {
                                std::unordered_map<unsigned int,
                                                   unsigned int>::iterator it =
                                  globalToLocalRowIdMap.find(i);
                                if (it != globalToLocalRowIdMap.end())
                                  projHamPar.local_el(it->second,
                                                      localColumnId) =
                                    projHamBlockHost[j * D + i - jvec];
                              }
                          }
                  }
                else
                  {
                    // Sum local projHamBlock across domain decomposition
                    // processors
                    if (!dftPtr->d_dftParamsPtr->useDeviceDirectAllReduce)
                      MPI_Allreduce(MPI_IN_PLACE,
                                    projHamBlockHostFP32.begin(),
                                    D * B,
                                    dataTypes::mpi_type_id(
                                      projHamBlockHostFP32.begin()),
                                    MPI_SUM,
                                    mpi_communicator);

                    // Copying only the lower triangular part to the ScaLAPACK
                    // projected Hamiltonian matrix
                    if (processGrid->is_process_active())
                      for (unsigned int j = 0; j < B; ++j)
                        if (globalToLocalColumnIdMap.find(j + jvec) !=
                            globalToLocalColumnIdMap.end())
                          {
                            const unsigned int localColumnId =
                              globalToLocalColumnIdMap[j + jvec];
                            for (unsigned int i = j + jvec; i < N; ++i)
                              {
                                std::unordered_map<unsigned int,
                                                   unsigned int>::iterator it =
                                  globalToLocalRowIdMap.find(i);
                                if (it != globalToLocalRowIdMap.end())
                                  projHamPar.local_el(it->second,
                                                      localColumnId) =
                                    projHamBlockHostFP32[j * D + i - jvec];
                              }
                          }
                  }
              }
          } // band parallelization
        blockCount += 1;
      }

    // return deviceblas handle to default stream
    dftfe::utils::deviceBlasWrapper::setStream(handle, NULL);

    for (int i = 0; i < numberBlocks; ++i)
      {
        dftfe::utils::deviceEventDestroy(computeEvents[i]);
        dftfe::utils::deviceEventDestroy(copyEvents[i]);
      }

    dftfe::utils::deviceStreamDestroy(streamCompute);
    dftfe::utils::deviceStreamDestroy(streamDataMove);

    if (numberBandGroups > 1)
      {
        MPI_Barrier(dftPtr->interBandGroupComm);
        linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
          processGrid, projHamPar, dftPtr->interBandGroupComm);
      }
  }

  // X^{T}*HConj*XConj  with overlap of computation and
  // communication
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro, memorySpace>::
    XtHXMixedPrecCommunOverlapComputeCommun(
      const dataTypes::number *                        X,
      distributedDeviceVec<dataTypes::number> &        XBlock,
      distributedDeviceVec<dataTypes::number> &        HXBlock,
      distributedDeviceVec<dataTypes::number> &        projectorKetTimesVector,
      const unsigned int                               M,
      const unsigned int                               N,
      const unsigned int                               Noc,
      dftfe::utils::deviceBlasHandle_t &               handle,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      projHamPar,
      utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
      const bool onlyHPrimePartForFirstOrderDensityMatResponse)
  {
    /////////////PSEUDO CODE for the implementation below for Overlapping
    /// compute and communication/////////////////
    //
    // In the algorithm below the communication and computation of two
    // consecutive blocks of wavefunctions: block i and block i+1 are
    // overlapped.
    // ----------------------------------------------------------
    // CMP denotes computuation of X^{T} times HXBlock
    // COP denotes Device->CPU copy of X^{T} times HXBlock
    // COM denotes blocking MPI_Allreduce on X^{T}HXBlock and copy to scalapack
    // matrix
    // ----------------------------------------------------------
    // Two Device streams are created: compute and copy
    // CMP is performed in compute Device stream and COP is performed in copy
    // Device stream. COP for a block can only start after the CMP for that
    // block in the compute stream is completed. COM is performed for a block
    // only after COP even for that block is completed.
    //
    // In a blocked loop do:
    // 1) [CMP] Call compute on first block (edge case only for first iteration)
    // 2) Wait for CMP event for current block to be completed.
    // 3) Swap current and next block memory (all iterations except edge case)
    // 4) [COP] Call copy on current block
    // 5) [CMP] Call compute on next block
    // 6) Wait for COP event for current block to be completed
    // 7) [COM] Perform blocking MPI_Allreduce on curent block and copy to
    // scalapack matrix
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    std::unordered_map<unsigned int, unsigned int> globalToLocalColumnIdMap;
    std::unordered_map<unsigned int, unsigned int> globalToLocalRowIdMap;
    linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
      processGrid, projHamPar, globalToLocalRowIdMap, globalToLocalColumnIdMap);

    // band group parallelization data structures
    const unsigned int numberBandGroups =
      dealii::Utilities::MPI::n_mpi_processes(dftPtr->interBandGroupComm);
    const unsigned int bandGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(dftPtr->interBandGroupComm);
    std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
    dftUtils::createBandParallelizationIndices(dftPtr->interBandGroupComm,
                                               N,
                                               bandGroupLowHighPlusOneIndices);



    const unsigned int vectorsBlockSize =
      std::min(dftPtr->d_dftParamsPtr->wfcBlockSize, N);
    const unsigned int numberBlocks = N / vectorsBlockSize;

    // create separate Device streams for Device->CPU copy and computation
    dftfe::utils::deviceStream_t streamCompute, streamDataMove;
    dftfe::utils::deviceStreamCreate(&streamCompute);
    dftfe::utils::deviceStreamCreate(&streamDataMove);

    // attach deviceblas handle to compute stream
    dftfe::utils::deviceBlasWrapper::setStream(handle, streamCompute);

    // create array of compute and copy events on Devices
    // for all the blocks. These are required for synchronization
    // between compute, copy and communication as discussed above in the
    // pseudo code
    dftfe::utils::deviceEvent_t computeEvents[numberBlocks];
    dftfe::utils::deviceEvent_t copyEvents[numberBlocks];

    for (int i = 0; i < numberBlocks; ++i)
      {
        dftfe::utils::deviceEventCreate(&computeEvents[i]);
        dftfe::utils::deviceEventCreate(&copyEvents[i]);
      }

    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::HOST_PINNED>
      projHamBlockHost;
    projHamBlockHost.resize(vectorsBlockSize * N, 0);
    std::memset(projHamBlockHost.begin(),
                0,
                vectorsBlockSize * N * sizeof(dataTypes::number));


    dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                dftfe::utils::MemorySpace::HOST_PINNED>
      projHamBlockHostFP32;
    projHamBlockHostFP32.resize(vectorsBlockSize * N, 0);
    std::memset(projHamBlockHostFP32.begin(),
                0,
                vectorsBlockSize * N * sizeof(dataTypes::numberFP32));

    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      HXBlockFull(vectorsBlockSize * M, dataTypes::number(0.0));
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      projHamBlock(vectorsBlockSize * N, dataTypes::number(0.0));
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      projHamBlockNext(vectorsBlockSize * N, dataTypes::number(0.0));

    dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                dftfe::utils::MemorySpace::DEVICE>
      projHamBlockFP32(vectorsBlockSize * N, dataTypes::numberFP32(0.0));

    dftfe::utils::MemoryStorage<dataTypes::numberValueType,
                                dftfe::utils::MemorySpace::DEVICE>
      tempReal;
    dftfe::utils::MemoryStorage<dataTypes::numberValueType,
                                dftfe::utils::MemorySpace::DEVICE>
      tempImag;

    dftfe::utils::MemoryStorage<dataTypes::numberFP32ValueType,
                                dftfe::utils::MemorySpace::DEVICE>
      tempRealFP32;
    dftfe::utils::MemoryStorage<dataTypes::numberFP32ValueType,
                                dftfe::utils::MemorySpace::DEVICE>
      tempImagFP32;
    if (std::is_same<dataTypes::number, std::complex<double>>::value)
      {
        tempReal.resize(vectorsBlockSize * N, 0);
        tempImag.resize(vectorsBlockSize * N, 0);
        tempRealFP32.resize(vectorsBlockSize * N, 0);
        tempImagFP32.resize(vectorsBlockSize * N, 0);
      }

    unsigned int blockCount = 0;
    for (unsigned int jvec = 0; jvec < N; jvec += vectorsBlockSize)
      {
        // Correct block dimensions if block "goes off edge of" the matrix
        const unsigned int B = std::min(vectorsBlockSize, N - jvec);

        if ((jvec + B) <=
              bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
            (jvec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
          {
            const unsigned int chebyBlockSize =
              std::min(dftPtr->d_dftParamsPtr->chebyWfcBlockSize, N);

            const dataTypes::number alpha = dataTypes::number(1.0),
                                    beta  = dataTypes::number(0.0);
            const unsigned int D          = N - jvec;

            // handle edge case for the first block or the first block in the
            // band group in case of band parallelization
            if (jvec == bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
              {
                // compute HXBlockFull in an inner loop over blocks of B
                // wavefunction vectors
                for (unsigned int k = jvec; k < jvec + B; k += chebyBlockSize)
                  {
                    dftfe::utils::deviceKernelsGeneric::
                      stridedCopyToBlockConstantStride(
                        chebyBlockSize, N, M, k, X, XBlock.begin());

                    // evaluate H times XBlock^{T} and store in HXBlock^{T}
                    HXBlock.setValue(0);
                    const bool   scaleFlag = false;
                    const double scalar    = 1.0;
                    HX(XBlock,
                       projectorKetTimesVector,
                       M,
                       chebyBlockSize,
                       scaleFlag,
                       scalar,
                       HXBlock,
                       false,
                       onlyHPrimePartForFirstOrderDensityMatResponse);

                    dftfe::utils::deviceKernelsGeneric::
                      stridedCopyFromBlockConstantStride(B,
                                                         chebyBlockSize,
                                                         M,
                                                         k - jvec,
                                                         HXBlock.begin(),
                                                         HXBlockFull.begin());
                  }

                // evalute X^{T} times HXBlock
                dftfe::utils::deviceBlasWrapper::gemm(
                  handle,
                  dftfe::utils::DEVICEBLAS_OP_N,
                  std::is_same<dataTypes::number, std::complex<double>>::value ?
                    dftfe::utils::DEVICEBLAS_OP_C :
                    dftfe::utils::DEVICEBLAS_OP_T,
                  D,
                  B,
                  M,
                  &alpha,
                  X + jvec,
                  N,
                  HXBlockFull.begin(),
                  B,
                  &beta,
                  projHamBlock.begin(),
                  D);

                // record completion of compute for first block
                dftfe::utils::deviceEventRecord(computeEvents[blockCount],
                                                streamCompute);
              }


            // Before swap host thread needs to wait till compute on
            // currentblock is over. Since swap occurs on the null stream, any
            // future calls in the streamDataMove will only occur after both the
            // compute on currentblock and swap is over. Note that at this point
            // there is nothing queued in the streamDataMove as all previous
            // operations in that stream are over.
            if ((dftfe::utils::deviceEventSynchronize(
                   computeEvents[blockCount]) == dftfe::utils::deviceSuccess) &&
                (jvec > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId]))
              projHamBlock.swap(projHamBlockNext);

            const unsigned int jvecNew = jvec + vectorsBlockSize;
            const unsigned int DNew    = N - jvecNew;

            // start computations on the next block
            if (jvecNew <
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1])
              {
                for (unsigned int k = jvecNew; k < jvecNew + B;
                     k += chebyBlockSize)
                  {
                    dftfe::utils::deviceKernelsGeneric::
                      stridedCopyToBlockConstantStride(
                        chebyBlockSize, N, M, k, X, XBlock.begin());

                    // evaluate H times XBlock^{T} and store in HXBlock^{T}
                    HXBlock.setValue(0);
                    const bool   scaleFlag = false;
                    const double scalar    = 1.0;
                    HX(XBlock,
                       projectorKetTimesVector,
                       M,
                       chebyBlockSize,
                       scaleFlag,
                       scalar,
                       HXBlock,
                       false,
                       onlyHPrimePartForFirstOrderDensityMatResponse);

                    dftfe::utils::deviceKernelsGeneric::
                      stridedCopyFromBlockConstantStride(B,
                                                         chebyBlockSize,
                                                         M,
                                                         k - jvecNew,
                                                         HXBlock.begin(),
                                                         HXBlockFull.begin());
                  }

                // evalute X^{T} times HXBlock
                dftfe::utils::deviceBlasWrapper::gemm(
                  handle,
                  dftfe::utils::DEVICEBLAS_OP_N,
                  std::is_same<dataTypes::number, std::complex<double>>::value ?
                    dftfe::utils::DEVICEBLAS_OP_C :
                    dftfe::utils::DEVICEBLAS_OP_T,
                  DNew,
                  B,
                  M,
                  &alpha,
                  X + jvecNew,
                  N,
                  HXBlockFull.begin(),
                  B,
                  &beta,
                  projHamBlockNext.begin(),
                  DNew);

                // record completion of compute for next block
                dftfe::utils::deviceEventRecord(computeEvents[blockCount + 1],
                                                streamCompute);
              }

            if (!(jvec + B > Noc))
              {
                dftfe::utils::deviceKernelsGeneric::
                  copyValueType1ArrToValueType2Arr(D * B,
                                                   projHamBlock.begin(),
                                                   projHamBlockFP32.begin(),
                                                   streamDataMove);
              }

            if (dftPtr->d_dftParamsPtr->useDeviceDirectAllReduce)
              {
                if (jvec + B > Noc)
                  {
                    if (std::is_same<dataTypes::number,
                                     std::complex<double>>::value)
                      devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                        projHamBlock.begin(),
                        projHamBlock.begin(),
                        D * B,
                        tempReal.begin(),
                        tempImag.begin(),
                        streamDataMove);
                    else
                      devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                        projHamBlock.begin(),
                        projHamBlock.begin(),
                        D * B,
                        streamDataMove);
                  }
                else
                  {
                    if (std::is_same<dataTypes::number,
                                     std::complex<double>>::value)
                      devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                        projHamBlockFP32.begin(),
                        projHamBlockFP32.begin(),
                        D * B,
                        tempRealFP32.begin(),
                        tempImagFP32.begin(),
                        streamDataMove);
                    else
                      devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                        projHamBlockFP32.begin(),
                        projHamBlockFP32.begin(),
                        D * B,
                        streamDataMove);
                  }
              }

            if (jvec + B > Noc)
              dftfe::utils::deviceMemcpyAsyncD2H(
                projHamBlockHost.begin(),
                dftfe::utils::makeDataTypeDeviceCompatible(
                  projHamBlock.begin()),
                D * B * sizeof(dataTypes::number),
                streamDataMove);
            else
              dftfe::utils::deviceMemcpyAsyncD2H(
                projHamBlockHostFP32.begin(),
                dftfe::utils::makeDataTypeDeviceCompatible(
                  projHamBlockFP32.begin()),
                D * B * sizeof(dataTypes::numberFP32),
                streamDataMove);

            // record completion of Device->CPU copy for current block
            dftfe::utils::deviceEventRecord(copyEvents[blockCount],
                                            streamDataMove);

            // Check that Device->CPU on the current block has been completed.
            // If completed, perform blocking MPI commmunication on the current
            // block and copy to ScaLAPACK matrix
            if (dftfe::utils::deviceEventSynchronize(copyEvents[blockCount]) ==
                dftfe::utils::deviceSuccess)
              {
                if (jvec + B > Noc)
                  {
                    // Sum local projHamBlock across domain decomposition
                    // processors
                    if (!dftPtr->d_dftParamsPtr->useDeviceDirectAllReduce)
                      MPI_Allreduce(MPI_IN_PLACE,
                                    projHamBlockHost.begin(),
                                    D * B,
                                    dataTypes::mpi_type_id(
                                      projHamBlockHost.begin()),
                                    MPI_SUM,
                                    mpi_communicator);

                    // Copying only the lower triangular part to the ScaLAPACK
                    // projected Hamiltonian matrix
                    if (processGrid->is_process_active())
                      for (unsigned int j = 0; j < B; ++j)
                        if (globalToLocalColumnIdMap.find(j + jvec) !=
                            globalToLocalColumnIdMap.end())
                          {
                            const unsigned int localColumnId =
                              globalToLocalColumnIdMap[j + jvec];
                            for (unsigned int i = j + jvec; i < N; ++i)
                              {
                                std::unordered_map<unsigned int,
                                                   unsigned int>::iterator it =
                                  globalToLocalRowIdMap.find(i);
                                if (it != globalToLocalRowIdMap.end())
                                  projHamPar.local_el(it->second,
                                                      localColumnId) =
                                    projHamBlockHost[j * D + i - jvec];
                              }
                          }
                  }
                else
                  {
                    // Sum local projHamBlock across domain decomposition
                    // processors
                    if (!dftPtr->d_dftParamsPtr->useDeviceDirectAllReduce)
                      MPI_Allreduce(MPI_IN_PLACE,
                                    projHamBlockHostFP32.begin(),
                                    D * B,
                                    dataTypes::mpi_type_id(
                                      projHamBlockHostFP32.begin()),
                                    MPI_SUM,
                                    mpi_communicator);

                    // Copying only the lower triangular part to the ScaLAPACK
                    // projected Hamiltonian matrix
                    if (processGrid->is_process_active())
                      for (unsigned int j = 0; j < B; ++j)
                        if (globalToLocalColumnIdMap.find(j + jvec) !=
                            globalToLocalColumnIdMap.end())
                          {
                            const unsigned int localColumnId =
                              globalToLocalColumnIdMap[j + jvec];
                            for (unsigned int i = j + jvec; i < N; ++i)
                              {
                                std::unordered_map<unsigned int,
                                                   unsigned int>::iterator it =
                                  globalToLocalRowIdMap.find(i);
                                if (it != globalToLocalRowIdMap.end())
                                  projHamPar.local_el(it->second,
                                                      localColumnId) =
                                    projHamBlockHostFP32[j * D + i - jvec];
                              }
                          }
                  }
              }

          } // band parallelization
        blockCount += 1;
      }

    // return deviceblas handle to default stream
    dftfe::utils::deviceBlasWrapper::setStream(handle, NULL);

    for (int i = 0; i < numberBlocks; ++i)
      {
        dftfe::utils::deviceEventDestroy(computeEvents[i]);
        dftfe::utils::deviceEventDestroy(copyEvents[i]);
      }

    dftfe::utils::deviceStreamDestroy(streamCompute);
    dftfe::utils::deviceStreamDestroy(streamDataMove);

    if (numberBandGroups > 1)
      {
        MPI_Barrier(dftPtr->interBandGroupComm);
        linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
          processGrid, projHamPar, dftPtr->interBandGroupComm);
      }
  }

#include "computeNonLocalHamiltonianTimesXMemoryOptBatchGEMMDevice.cc"
#include "hamiltonianMatrixCalculatorFlattenedDevice.cc"
#include "matrixVectorProductImplementationsDevice.cc"
#include "shapeFunctionDataCalculatorDevice.cc"
#include "instDevice.cc"
} // namespace dftfe
