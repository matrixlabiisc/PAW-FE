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

#include <chebyshevOrthogonalizedSubspaceIterationSolverDevice.h>
#include <dftUtils.h>
#include <deviceHelpers.h>
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <linearAlgebraOperations.h>
#include <linearAlgebraOperationsDevice.h>
#include <vectorUtilities.h>

static const unsigned int order_lookup[][2] = {
  {500, 24}, // <= 500 ~> chebyshevOrder = 24
  {750, 30},
  {1000, 39},
  {1500, 50},
  {2000, 53},
  {3000, 57},
  {4000, 62},
  {5000, 69},
  {9000, 77},
  {14000, 104},
  {20000, 119},
  {30000, 162},
  {50000, 300},
  {80000, 450},
  {100000, 550},
  {200000, 700},
  {500000, 1000}};

namespace dftfe
{
  namespace
  {
    template <typename NumberType>
    __global__ void
    stridedCopyToBlockKernel(const unsigned int BVec,
                             const unsigned int M,
                             const NumberType * xVec,
                             const unsigned int N,
                             NumberType *       yVec,
                             const unsigned int startingXVecId)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numGangsPerBVec = (BVec + blockDim.x - 1) / blockDim.x;
      const unsigned int gangBlockId     = blockIdx.x / numGangsPerBVec;
      const unsigned int localThreadId =
        globalThreadId - gangBlockId * numGangsPerBVec * blockDim.x;

      if (globalThreadId < M * numGangsPerBVec * blockDim.x &&
          localThreadId < BVec)
        {
          *(yVec + gangBlockId * BVec + localThreadId) =
            *(xVec + gangBlockId * N + startingXVecId + localThreadId);
        }
    }

    template <typename NumberType>
    __global__ void
    stridedCopyFromBlockKernel(const unsigned int BVec,
                               const unsigned int M,
                               const NumberType * xVec,
                               const unsigned int N,
                               NumberType *       yVec,
                               const unsigned int startingXVecId)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numGangsPerBVec = (BVec + blockDim.x - 1) / blockDim.x;
      const unsigned int gangBlockId     = blockIdx.x / numGangsPerBVec;
      const unsigned int localThreadId =
        globalThreadId - gangBlockId * numGangsPerBVec * blockDim.x;

      if (globalThreadId < M * numGangsPerBVec * blockDim.x &&
          localThreadId < BVec)
        {
          *(yVec + gangBlockId * N + startingXVecId + localThreadId) =
            *(xVec + gangBlockId * BVec + localThreadId);
        }
    }


    __global__ void
    scaleDeviceKernel(const unsigned int contiguousBlockSize,
                      const unsigned int numContiguousBlocks,
                      double *           srcArray,
                      const double *     scalingVector)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numGangsPerContiguousBlock =
        (contiguousBlockSize + (blockDim.x - 1)) / blockDim.x;
      const unsigned int gangBlockId = blockIdx.x / numGangsPerContiguousBlock;
      const unsigned int localThreadId =
        globalThreadId - gangBlockId * numGangsPerContiguousBlock * blockDim.x;
      if (globalThreadId <
            numContiguousBlocks * numGangsPerContiguousBlock * blockDim.x &&
          localThreadId < contiguousBlockSize)
        {
          *(srcArray + (localThreadId + gangBlockId * contiguousBlockSize)) =
            *(srcArray + (localThreadId + gangBlockId * contiguousBlockSize)) *
            (*(scalingVector + gangBlockId));
        }
    }


    __global__ void
    scaleDeviceKernel(const unsigned int contiguousBlockSize,
                      const unsigned int numContiguousBlocks,
                      cuDoubleComplex *  srcArray,
                      const double *     scalingVector)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numGangsPerContiguousBlock =
        (contiguousBlockSize + (blockDim.x - 1)) / blockDim.x;
      const unsigned int gangBlockId = blockIdx.x / numGangsPerContiguousBlock;
      const unsigned int localThreadId =
        globalThreadId - gangBlockId * numGangsPerContiguousBlock * blockDim.x;
      if (globalThreadId <
            numContiguousBlocks * numGangsPerContiguousBlock * blockDim.x &&
          localThreadId < contiguousBlockSize)
        {
          *(srcArray + (localThreadId + gangBlockId * contiguousBlockSize)) =
            make_cuDoubleComplex(
              (srcArray + (localThreadId + gangBlockId * contiguousBlockSize))
                  ->x *
                (*(scalingVector + gangBlockId)),
              (srcArray + (localThreadId + gangBlockId * contiguousBlockSize))
                  ->y *
                (*(scalingVector + gangBlockId)));
        }
    }


    __global__ void
    setZeroKernel(const unsigned int BVec,
                  const unsigned int M,
                  const unsigned int N,
                  double *           yVec,
                  const unsigned int startingXVecId)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numGangsPerBVec = (BVec + blockDim.x - 1) / blockDim.x;
      const unsigned int gangBlockId     = blockIdx.x / numGangsPerBVec;
      const unsigned int localThreadId =
        globalThreadId - gangBlockId * numGangsPerBVec * blockDim.x;

      if (globalThreadId < M * numGangsPerBVec * blockDim.x &&
          localThreadId < BVec)
        {
          *(yVec + gangBlockId * N + startingXVecId + localThreadId) = 0.0;
        }
    }


    __global__ void
    setZeroKernel(const unsigned int BVec,
                  const unsigned int M,
                  const unsigned int N,
                  cuDoubleComplex *  yVec,
                  const unsigned int startingXVecId)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numGangsPerBVec = (BVec + blockDim.x - 1) / blockDim.x;
      const unsigned int gangBlockId     = blockIdx.x / numGangsPerBVec;
      const unsigned int localThreadId =
        globalThreadId - gangBlockId * numGangsPerBVec * blockDim.x;

      if (globalThreadId < M * numGangsPerBVec * blockDim.x &&
          localThreadId < BVec)
        {
          *(yVec + gangBlockId * N + startingXVecId + localThreadId) =
            make_cuDoubleComplex(0.0, 0.0);
        }
    }

    namespace internal
    {
      unsigned int
      setChebyshevOrder(const unsigned int d_upperBoundUnWantedSpectrum)
      {
        for (int i = 0; i < sizeof(order_lookup) / sizeof(order_lookup[0]); i++)
          {
            if (d_upperBoundUnWantedSpectrum <= order_lookup[i][0])
              return order_lookup[i][1];
          }
        return 1250;
      }
    } // namespace internal
  }   // namespace

  //
  // Constructor.
  //
  chebyshevOrthogonalizedSubspaceIterationSolverDevice::
    chebyshevOrthogonalizedSubspaceIterationSolverDevice(
      const MPI_Comm &     mpi_comm_parent,
      const MPI_Comm &     mpi_comm_domain,
      double               lowerBoundWantedSpectrum,
      double               lowerBoundUnWantedSpectrum,
      double               upperBoundUnWantedSpectrum,
      const dftParameters &dftParams)
    : d_lowerBoundWantedSpectrum(lowerBoundWantedSpectrum)
    , d_lowerBoundUnWantedSpectrum(lowerBoundUnWantedSpectrum)
    , d_upperBoundUnWantedSpectrum(upperBoundUnWantedSpectrum)
    , d_isTemporaryParallelVectorsCreated(false)
    , d_mpiCommParent(mpi_comm_parent)
    , d_dftParams(dftParams)
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
    , computing_timer(mpi_comm_domain,
                      pcout,
                      dftParams.reproducible_output || dftParams.verbosity < 4 ?
                        dealii::TimerOutput::never :
                        dealii::TimerOutput::summary,
                      dealii::TimerOutput::wall_times)
  {}


  //
  // reinitialize spectrum bounds
  //
  void
  chebyshevOrthogonalizedSubspaceIterationSolverDevice::reinitSpectrumBounds(
    double lowerBoundWantedSpectrum,
    double lowerBoundUnWantedSpectrum,
    double upperBoundUnWantedSpectrum)
  {
    d_lowerBoundWantedSpectrum   = lowerBoundWantedSpectrum;
    d_lowerBoundUnWantedSpectrum = lowerBoundUnWantedSpectrum;
    d_upperBoundUnWantedSpectrum = upperBoundUnWantedSpectrum;
  }


  //
  // solve
  //
  double
  chebyshevOrthogonalizedSubspaceIterationSolverDevice::solve(
    operatorDFTDeviceClass & operatorMatrix,
    elpaScalaManager &       elpaScala,
    dataTypes::numberDevice *eigenVectorsFlattenedDevice,
    dataTypes::numberDevice *eigenVectorsRotFracDensityFlattenedDevice,
    const unsigned int       flattenedSize,
    const unsigned int       totalNumberWaveFunctions,
    std::vector<double> &    eigenValues,
    std::vector<double> &    residualNorms,
    DeviceCCLWrapper &       devicecclMpiCommDomain,
    const MPI_Comm &         interBandGroupComm,
    const bool               isFirstFilteringCall,
    const bool               computeResidual,
    const bool               useMixedPrecOverall,
    const bool               isFirstScf)
  {
    dealii::TimerOutput computingTimerStandard(
      operatorMatrix.getMPICommunicator(),
      pcout,
      d_dftParams.reproducible_output || d_dftParams.verbosity < 2 ?
        dealii::TimerOutput::never :
        dealii::TimerOutput::every_call,
      dealii::TimerOutput::wall_times);

    cublasHandle_t &cublasHandle = operatorMatrix.getCublasHandle();

    //
    // allocate memory for full flattened array on device and fill it up
    //
    const unsigned int localVectorSize =
      flattenedSize / totalNumberWaveFunctions;

    // band group parallelization data structures
    const unsigned int numberBandGroups =
      dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);


    const unsigned int bandGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
    std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
    dftUtils::createBandParallelizationIndices(interBandGroupComm,
                                               totalNumberWaveFunctions,
                                               bandGroupLowHighPlusOneIndices);


    const unsigned int vectorsBlockSize =
      std::min(d_dftParams.chebyWfcBlockSize, totalNumberWaveFunctions);

    distributedDeviceVec<dataTypes::numberDevice> &deviceFlattenedArrayBlock =
      operatorMatrix.getParallelChebyBlockVectorDevice();

    distributedDeviceVec<dataTypes::numberDevice> &projectorKetTimesVector =
      operatorMatrix.getParallelProjectorKetTimesBlockVectorDevice();


    if (isFirstFilteringCall || !d_isTemporaryParallelVectorsCreated)
      {
        d_YArray.reinit(deviceFlattenedArrayBlock);

        d_deviceFlattenedFloatArrayBlock.reinit(
          operatorMatrix.getMatrixFreeData()->get_vector_partitioner(),
          vectorsBlockSize);

        if (d_dftParams.isPseudopotential)
          {
            if (d_dftParams.overlapComputeCommunCheby)
              d_projectorKetTimesVector2.reinit(projectorKetTimesVector);
          }


        if (d_dftParams.overlapComputeCommunCheby)
          d_deviceFlattenedArrayBlock2.reinit(deviceFlattenedArrayBlock);


        if (d_dftParams.overlapComputeCommunCheby)
          d_YArray2.reinit(d_deviceFlattenedArrayBlock2);


        d_isTemporaryParallelVectorsCreated = true;
      }

    if (isFirstFilteringCall)
      {
        if (d_dftParams.deviceFineGrainedTimings)
          {
            dftfe::utils::deviceSynchronize();
            computingTimerStandard.enter_subsection("Lanczos upper bound");
          }

        const std::pair<double, double> bounds =
          linearAlgebraOperationsDevice::lanczosLowerUpperBoundEigenSpectrum(
            operatorMatrix,
            deviceFlattenedArrayBlock,
            d_YArray,
            projectorKetTimesVector,
            vectorsBlockSize,
            d_dftParams);

        if (d_dftParams.deviceFineGrainedTimings)
          {
            dftfe::utils::deviceSynchronize();
            computingTimerStandard.leave_subsection("Lanczos upper bound");
          }

        d_lowerBoundWantedSpectrum   = bounds.first;
        d_upperBoundUnWantedSpectrum = bounds.second;
        d_lowerBoundUnWantedSpectrum =
          d_lowerBoundWantedSpectrum +
          (d_upperBoundUnWantedSpectrum - d_lowerBoundWantedSpectrum) *
            totalNumberWaveFunctions /
            operatorMatrix.getParallelVecSingleComponent().size() *
            (d_dftParams.reproducible_output ? 10.0 : 200.0);
      }
    else if (!d_dftParams.reuseLanczosUpperBoundFromFirstCall)
      {
        if (d_dftParams.deviceFineGrainedTimings)
          {
            dftfe::utils::deviceSynchronize();
            computingTimerStandard.enter_subsection("Lanczos upper bound");
          }

        const std::pair<double, double> bounds =
          linearAlgebraOperationsDevice::lanczosLowerUpperBoundEigenSpectrum(
            operatorMatrix,
            deviceFlattenedArrayBlock,
            d_YArray,
            projectorKetTimesVector,
            vectorsBlockSize,
            d_dftParams);

        if (d_dftParams.deviceFineGrainedTimings)
          {
            dftfe::utils::deviceSynchronize();
            computingTimerStandard.leave_subsection("Lanczos upper bound");
          }

        d_upperBoundUnWantedSpectrum = bounds.second;
      }

    if (d_dftParams.deviceFineGrainedTimings)
      {
        dftfe::utils::deviceSynchronize();
        computingTimerStandard.enter_subsection(
          "Chebyshev filtering on Device");
      }


    unsigned int chebyshevOrder = d_dftParams.chebyshevOrder;

    //
    // set Chebyshev order
    //
    if (chebyshevOrder == 0)
      {
        chebyshevOrder =
          internal::setChebyshevOrder(d_upperBoundUnWantedSpectrum);

        if (d_dftParams.orthogType.compare("CGS") == 0 &&
            !d_dftParams.isPseudopotential)
          chebyshevOrder *= 0.5;
      }

    chebyshevOrder =
      (isFirstScf && d_dftParams.isPseudopotential) ?
        chebyshevOrder *
          d_dftParams.chebyshevFilterPolyDegreeFirstScfScalingFactor :
        chebyshevOrder;


    //
    // output statements
    //
    if (d_dftParams.verbosity >= 2)
      {
        char buffer[100];

        sprintf(buffer,
                "%s:%18.10e\n",
                "upper bound of unwanted spectrum",
                d_upperBoundUnWantedSpectrum);
        pcout << buffer;
        sprintf(buffer,
                "%s:%18.10e\n",
                "lower bound of unwanted spectrum",
                d_lowerBoundUnWantedSpectrum);
        pcout << buffer;
        sprintf(buffer,
                "%s: %u\n\n",
                "Chebyshev polynomial degree",
                chebyshevOrder);
        pcout << buffer;
      }


    //
    // scale the eigenVectors (initial guess of single atom wavefunctions or
    // previous guess) to convert into Lowden Orthonormalized FE basis
    // multiply by M^{1/2}
    scaleDeviceKernel<<<
      (totalNumberWaveFunctions + (deviceConstants::blockSize - 1)) /
        deviceConstants::blockSize * localVectorSize,
      deviceConstants::blockSize>>>(totalNumberWaveFunctions,
                                    localVectorSize,
                                    eigenVectorsFlattenedDevice,
                                    operatorMatrix.getSqrtMassVec());


    // two blocks of wavefunctions are filtered simultaneously when overlap
    // compute communication in chebyshev filtering is toggled on
    const unsigned int numSimultaneousBlocks =
      d_dftParams.overlapComputeCommunCheby ? 2 : 1;
    unsigned int       numSimultaneousBlocksCurrent = numSimultaneousBlocks;
    const unsigned int numWfcsInBandGroup =
      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] -
      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId];
    int startIndexBandParal = totalNumberWaveFunctions;
    int numVectorsBandParal = 0;
    for (unsigned int jvec = 0; jvec < totalNumberWaveFunctions;
         jvec += numSimultaneousBlocksCurrent * vectorsBlockSize)
      {
        // Correct block dimensions if block "goes off edge of" the matrix
        const unsigned int BVec =
          vectorsBlockSize; // std::min(vectorsBlockSize,
                            // totalNumberWaveFunctions-jvec);

        // handle edge case when total number of blocks in a given band
        // group is not even in case of overlapping computation and
        // communciation in chebyshev filtering
        const unsigned int leftIndexBandGroupMargin =
          (jvec / numWfcsInBandGroup) * numWfcsInBandGroup;
        numSimultaneousBlocksCurrent =
          ((jvec + numSimultaneousBlocks * BVec - leftIndexBandGroupMargin) <=
             numWfcsInBandGroup &&
           numSimultaneousBlocks == 2) ?
            2 :
            1;

        if ((jvec + numSimultaneousBlocksCurrent * BVec) <=
              bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
            (jvec + numSimultaneousBlocksCurrent * BVec) >
              bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
          {
            if (jvec < startIndexBandParal)
              startIndexBandParal = jvec;
            numVectorsBandParal =
              jvec + numSimultaneousBlocksCurrent * BVec - startIndexBandParal;

            // copy from vector containg all wavefunction vectors to current
            // wavefunction vectors block
            stridedCopyToBlockKernel<<<
              (BVec + (deviceConstants::blockSize - 1)) /
                deviceConstants::blockSize * localVectorSize,
              deviceConstants::blockSize>>>(BVec,
                                            localVectorSize,
                                            eigenVectorsFlattenedDevice,
                                            totalNumberWaveFunctions,
                                            deviceFlattenedArrayBlock.begin(),
                                            jvec);

            if (d_dftParams.overlapComputeCommunCheby &&
                numSimultaneousBlocksCurrent == 2)
              stridedCopyToBlockKernel<<<
                (BVec + (deviceConstants::blockSize - 1)) /
                  deviceConstants::blockSize * localVectorSize,
                deviceConstants::blockSize>>>(
                BVec,
                localVectorSize,
                eigenVectorsFlattenedDevice,
                totalNumberWaveFunctions,
                d_deviceFlattenedArrayBlock2.begin(),
                jvec + BVec);

            //
            // call Chebyshev filtering function only for the current block
            // or two simulataneous blocks (in case of overlap computation
            // and communication) to be filtered and does in-place filtering
            if (d_dftParams.overlapComputeCommunCheby &&
                numSimultaneousBlocksCurrent == 2)
              {
                linearAlgebraOperationsDevice::chebyshevFilter(
                  operatorMatrix,
                  deviceFlattenedArrayBlock,
                  d_YArray,
                  d_deviceFlattenedFloatArrayBlock,
                  projectorKetTimesVector,
                  d_deviceFlattenedArrayBlock2,
                  d_YArray2,
                  d_projectorKetTimesVector2,
                  localVectorSize,
                  BVec,
                  chebyshevOrder,
                  d_lowerBoundUnWantedSpectrum,
                  d_upperBoundUnWantedSpectrum,
                  d_lowerBoundWantedSpectrum,
                  useMixedPrecOverall,
                  d_dftParams);
              }
            else
              {
                linearAlgebraOperationsDevice::chebyshevFilter(
                  operatorMatrix,
                  deviceFlattenedArrayBlock,
                  d_YArray,
                  d_deviceFlattenedFloatArrayBlock,
                  projectorKetTimesVector,
                  localVectorSize,
                  BVec,
                  chebyshevOrder,
                  d_lowerBoundUnWantedSpectrum,
                  d_upperBoundUnWantedSpectrum,
                  d_lowerBoundWantedSpectrum,
                  useMixedPrecOverall,
                  d_dftParams);
              }

            // copy current wavefunction vectors block to vector containing
            // all wavefunction vectors
            stridedCopyFromBlockKernel<<<
              (BVec + (deviceConstants::blockSize - 1)) /
                deviceConstants::blockSize * localVectorSize,
              deviceConstants::blockSize>>>(BVec,
                                            localVectorSize,
                                            deviceFlattenedArrayBlock.begin(),
                                            totalNumberWaveFunctions,
                                            eigenVectorsFlattenedDevice,
                                            jvec);

            if (d_dftParams.overlapComputeCommunCheby &&
                numSimultaneousBlocksCurrent == 2)
              stridedCopyFromBlockKernel<<<
                (BVec + (deviceConstants::blockSize - 1)) /
                  deviceConstants::blockSize * localVectorSize,
                deviceConstants::blockSize>>>(
                BVec,
                localVectorSize,
                d_deviceFlattenedArrayBlock2.begin(),
                totalNumberWaveFunctions,
                eigenVectorsFlattenedDevice,
                jvec + BVec);
          }
        else
          {
            // set to zero wavefunctions which wont go through chebyshev
            // filtering inside a given band group
            setZeroKernel<<<(numSimultaneousBlocksCurrent * BVec +
                             (deviceConstants::blockSize - 1)) /
                              deviceConstants::blockSize * localVectorSize,
                            deviceConstants::blockSize>>>(
              numSimultaneousBlocksCurrent * BVec,
              localVectorSize,
              totalNumberWaveFunctions,
              eigenVectorsFlattenedDevice,
              jvec);
          }

      } // block loop

    if (d_dftParams.deviceFineGrainedTimings)
      {
        dftfe::utils::deviceSynchronize();
        computingTimerStandard.leave_subsection(
          "Chebyshev filtering on Device");

        if (d_dftParams.verbosity >= 4)
          pcout << "ChebyShev Filtering Done: " << std::endl;
      }


    if (numberBandGroups > 1)
      {
        std::vector<dataTypes::number> eigenVectorsFlattened(
          totalNumberWaveFunctions * localVectorSize, dataTypes::number(0.0));

        dftfe::utils::deviceMemcpyD2H(dftfe::utils::makeDataTypeDeviceCompatible(
                     &eigenVectorsFlattened[0]),
                   eigenVectorsFlattenedDevice,
                   totalNumberWaveFunctions * localVectorSize *
                     sizeof(dataTypes::numberDevice));

        MPI_Barrier(interBandGroupComm);


        MPI_Allreduce(MPI_IN_PLACE,
                      &eigenVectorsFlattened[0],
                      totalNumberWaveFunctions * localVectorSize,
                      dataTypes::mpi_type_id(&eigenVectorsFlattened[0]),
                      MPI_SUM,
                      interBandGroupComm);

        MPI_Barrier(interBandGroupComm);

        dftfe::utils::deviceMemcpyH2D(eigenVectorsFlattenedDevice,
                   dftfe::utils::makeDataTypeDeviceCompatible(
                     &eigenVectorsFlattened[0]),
                   totalNumberWaveFunctions * localVectorSize *
                     sizeof(dataTypes::numberDevice));
      }

    // if (d_dftParams.measureOnlyChebyTime)
    //  exit(0);

    /*
       int inc=1;
       double result=0.0;
       cublasDnrm2(cublasHandle,
       flattenedSize,
       eigenVectorsFlattenedDevice,
       inc,
       &result);
       result=result*result;
       result=dealii::Utilities::MPI::sum(result,operatorMatrix.getMPICommunicator());
       std::cout<<"l2 norm Chebyshev filtered x:
       "<<std::sqrt(result)<<std::endl;
     */

    if (d_dftParams.orthogType.compare("GS") == 0)
      {
        AssertThrow(
          false,
          dealii::ExcMessage(
            "Classical Gram-Schmidt Orthonormalization not implemented in Device:"));
      }

    std::fill(eigenValues.begin(), eigenValues.end(), 0.0);

    if (eigenValues.size() != totalNumberWaveFunctions)
      {
        linearAlgebraOperationsDevice::rayleighRitzGEPSpectrumSplitDirect(
          operatorMatrix,
          elpaScala,
          eigenVectorsFlattenedDevice,
          eigenVectorsRotFracDensityFlattenedDevice,
          deviceFlattenedArrayBlock,
          d_deviceFlattenedFloatArrayBlock,
          d_YArray,
          projectorKetTimesVector,
          localVectorSize,
          totalNumberWaveFunctions,
          totalNumberWaveFunctions - eigenValues.size(),
          d_mpiCommParent,
          operatorMatrix.getMPICommunicator(),
          devicecclMpiCommDomain,
          interBandGroupComm,
          eigenValues,
          cublasHandle,
          d_dftParams,
          useMixedPrecOverall);
      }
    else
      {
        linearAlgebraOperationsDevice::rayleighRitzGEP(
          operatorMatrix,
          elpaScala,
          eigenVectorsFlattenedDevice,
          deviceFlattenedArrayBlock,
          d_deviceFlattenedFloatArrayBlock,
          d_YArray,
          projectorKetTimesVector,
          localVectorSize,
          totalNumberWaveFunctions,
          d_mpiCommParent,
          operatorMatrix.getMPICommunicator(),
          devicecclMpiCommDomain,
          interBandGroupComm,
          eigenValues,
          cublasHandle,
          d_dftParams,
          useMixedPrecOverall);
      }


    if (computeResidual)
      {
        if (d_dftParams.deviceFineGrainedTimings)
          {
            dftfe::utils::deviceSynchronize();
            computingTimerStandard.enter_subsection("Residual norm");
          }

        if (eigenValues.size() != totalNumberWaveFunctions)
          linearAlgebraOperationsDevice::computeEigenResidualNorm(
            operatorMatrix,
            eigenVectorsRotFracDensityFlattenedDevice,
            deviceFlattenedArrayBlock,
            d_YArray,
            projectorKetTimesVector,
            localVectorSize,
            eigenValues.size(),
            eigenValues,
            operatorMatrix.getMPICommunicator(),
            interBandGroupComm,
            cublasHandle,
            residualNorms,
            d_dftParams);
        else
          linearAlgebraOperationsDevice::computeEigenResidualNorm(
            operatorMatrix,
            eigenVectorsFlattenedDevice,
            deviceFlattenedArrayBlock,
            d_YArray,
            projectorKetTimesVector,
            localVectorSize,
            totalNumberWaveFunctions,
            eigenValues,
            operatorMatrix.getMPICommunicator(),
            interBandGroupComm,
            cublasHandle,
            residualNorms,
            d_dftParams,
            true);

        if (d_dftParams.deviceFineGrainedTimings)
          {
            dftfe::utils::deviceSynchronize();
            computingTimerStandard.leave_subsection("Residual norm");
          }
      }

    //
    // scale the eigenVectors with M^{-1/2} to represent the wavefunctions in
    // the usual FE basis
    //
    scaleDeviceKernel<<<
      (totalNumberWaveFunctions + (deviceConstants::blockSize - 1)) /
        deviceConstants::blockSize * localVectorSize,
      deviceConstants::blockSize>>>(totalNumberWaveFunctions,
                                    localVectorSize,
                                    eigenVectorsFlattenedDevice,
                                    operatorMatrix.getInvSqrtMassVec());

    if (eigenValues.size() != totalNumberWaveFunctions)
      scaleDeviceKernel<<<
        (eigenValues.size() + (deviceConstants::blockSize - 1)) /
          deviceConstants::blockSize * localVectorSize,
        deviceConstants::blockSize>>>(eigenValues.size(),
                                      localVectorSize,
                                      eigenVectorsRotFracDensityFlattenedDevice,
                                      operatorMatrix.getInvSqrtMassVec());

    return d_upperBoundUnWantedSpectrum;
  }

  //
  // solve
  //
  void
  chebyshevOrthogonalizedSubspaceIterationSolverDevice::solveNoRR(
    operatorDFTDeviceClass & operatorMatrix,
    elpaScalaManager &       elpaScala,
    dataTypes::numberDevice *eigenVectorsFlattenedDevice,
    const unsigned int       flattenedSize,
    const unsigned int       totalNumberWaveFunctions,
    std::vector<double> &    eigenValues,
    DeviceCCLWrapper &       devicecclMpiCommDomain,
    const MPI_Comm &         interBandGroupComm,
    const unsigned int       numberPasses,
    const bool               useMixedPrecOverall)
  {
    cublasHandle_t &cublasHandle = operatorMatrix.getCublasHandle();

    //
    // allocate memory for full flattened array on device and fill it up
    //
    const unsigned int localVectorSize =
      flattenedSize / totalNumberWaveFunctions;


    // band group parallelization data structures
    const unsigned int numberBandGroups =
      dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);


    const unsigned int bandGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
    std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
    dftUtils::createBandParallelizationIndices(interBandGroupComm,
                                               totalNumberWaveFunctions,
                                               bandGroupLowHighPlusOneIndices);

    const unsigned int wfcBlockSize =
      std::min(d_dftParams.wfcBlockSize, totalNumberWaveFunctions);


    const unsigned int chebyBlockSize =
      std::min(d_dftParams.chebyWfcBlockSize, totalNumberWaveFunctions);

    distributedDeviceVec<dataTypes::numberDevice> &deviceFlattenedArrayBlock =
      operatorMatrix.getParallelChebyBlockVectorDevice();

    distributedDeviceVec<dataTypes::numberDevice> &projectorKetTimesVector =
      operatorMatrix.getParallelProjectorKetTimesBlockVectorDevice();


    if (!d_isTemporaryParallelVectorsCreated)
      {
        d_YArray.reinit(deviceFlattenedArrayBlock);

        d_deviceFlattenedFloatArrayBlock.reinit(
          operatorMatrix.getMatrixFreeData()->get_vector_partitioner(),
          chebyBlockSize);


        if (d_dftParams.overlapComputeCommunCheby)
          d_deviceFlattenedArrayBlock2.reinit(deviceFlattenedArrayBlock);


        if (d_dftParams.overlapComputeCommunCheby)
          d_YArray2.reinit(d_deviceFlattenedArrayBlock2);


        if (d_dftParams.overlapComputeCommunCheby)
          d_projectorKetTimesVector2.reinit(projectorKetTimesVector);
      }

    if (!d_dftParams.reuseLanczosUpperBoundFromFirstCall)
      {
        const std::pair<double, double> bounds =
          linearAlgebraOperationsDevice::lanczosLowerUpperBoundEigenSpectrum(
            operatorMatrix,
            deviceFlattenedArrayBlock,
            d_YArray,
            projectorKetTimesVector,
            chebyBlockSize,
            d_dftParams);

        d_upperBoundUnWantedSpectrum = bounds.second;
      }

    unsigned int chebyshevOrder = d_dftParams.chebyshevOrder;

    //
    // set Chebyshev order
    //
    if (chebyshevOrder == 0)
      chebyshevOrder =
        internal::setChebyshevOrder(d_upperBoundUnWantedSpectrum);

    chebyshevOrder =
      (d_dftParams.isPseudopotential) ?
        chebyshevOrder *
          d_dftParams.chebyshevFilterPolyDegreeFirstScfScalingFactor :
        chebyshevOrder;


    //
    // output statements
    //
    if (d_dftParams.verbosity >= 2)
      {
        char buffer[100];

        sprintf(buffer,
                "%s:%18.10e\n",
                "upper bound of unwanted spectrum",
                d_upperBoundUnWantedSpectrum);
        pcout << buffer;
        sprintf(buffer,
                "%s:%18.10e\n",
                "lower bound of unwanted spectrum",
                d_lowerBoundUnWantedSpectrum);
        pcout << buffer;
        sprintf(buffer,
                "%s: %u\n\n",
                "Chebyshev polynomial degree",
                chebyshevOrder);
        pcout << buffer;
      }


    //
    // scale the eigenVectors (initial guess of single atom wavefunctions or
    // previous guess) to convert into Lowden Orthonormalized FE basis multiply
    // by M^{1/2}
    scaleDeviceKernel<<<
      (totalNumberWaveFunctions + (deviceConstants::blockSize - 1)) /
        deviceConstants::blockSize * localVectorSize,
      deviceConstants::blockSize>>>(totalNumberWaveFunctions,
                                    localVectorSize,
                                    eigenVectorsFlattenedDevice,
                                    operatorMatrix.getSqrtMassVec());

    for (unsigned int ipass = 0; ipass < numberPasses; ipass++)
      {
        pcout << "Beginning no RR Chebyshev filter subpspace iteration pass: "
              << ipass + 1 << std::endl;

        for (unsigned int ivec = 0; ivec < totalNumberWaveFunctions;
             ivec += wfcBlockSize)
          {
            // two blocks of wavefunctions are filtered simultaneously when
            // overlap compute communication in chebyshev filtering is toggled
            // on
            const unsigned int numSimultaneousBlocks =
              d_dftParams.overlapComputeCommunCheby ? 2 : 1;
            unsigned int numSimultaneousBlocksCurrent = numSimultaneousBlocks;
            const unsigned int numWfcsInBandGroup =
              bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] -
              bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId];
            for (unsigned int jvec = ivec; jvec < (ivec + wfcBlockSize);
                 jvec += numSimultaneousBlocksCurrent * chebyBlockSize)
              {
                // Correct block dimensions if block "goes off edge of" the
                // matrix
                const unsigned int BVec =
                  chebyBlockSize; // std::min(vectorsBlockSize,
                                  // totalNumberWaveFunctions-jvec);

                // handle edge case when total number of blocks in a given band
                // group is not even in case of overlapping computation and
                // communciation in chebyshev filtering
                const unsigned int leftIndexBandGroupMargin =
                  (jvec / numWfcsInBandGroup) * numWfcsInBandGroup;
                numSimultaneousBlocksCurrent =
                  ((jvec + numSimultaneousBlocks * BVec -
                    leftIndexBandGroupMargin) <= numWfcsInBandGroup &&
                   numSimultaneousBlocks == 2) ?
                    2 :
                    1;

                if ((jvec + numSimultaneousBlocksCurrent * BVec) <=
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                    (jvec + numSimultaneousBlocksCurrent * BVec) >
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                  {
                    // copy from vector containg all wavefunction vectors to
                    // current wavefunction vectors block
                    stridedCopyToBlockKernel<<<
                      (BVec + (deviceConstants::blockSize - 1)) /
                        deviceConstants::blockSize * localVectorSize,
                      deviceConstants::blockSize>>>(
                      BVec,
                      localVectorSize,
                      eigenVectorsFlattenedDevice,
                      totalNumberWaveFunctions,
                      deviceFlattenedArrayBlock.begin(),
                      jvec);

                    if (d_dftParams.overlapComputeCommunCheby &&
                        numSimultaneousBlocksCurrent == 2)
                      stridedCopyToBlockKernel<<<
                        (BVec + (deviceConstants::blockSize - 1)) /
                          deviceConstants::blockSize * localVectorSize,
                        deviceConstants::blockSize>>>(
                        BVec,
                        localVectorSize,
                        eigenVectorsFlattenedDevice,
                        totalNumberWaveFunctions,
                        d_deviceFlattenedArrayBlock2.begin(),
                        jvec + BVec);

                    //
                    // call Chebyshev filtering function only for the current
                    // block or two simulataneous blocks (in case of overlap
                    // computation and communication) to be filtered and does
                    // in-place filtering
                    if (d_dftParams.overlapComputeCommunCheby &&
                        numSimultaneousBlocksCurrent == 2)
                      {
                        linearAlgebraOperationsDevice::chebyshevFilter(
                          operatorMatrix,
                          deviceFlattenedArrayBlock,
                          d_YArray,
                          d_deviceFlattenedFloatArrayBlock,
                          projectorKetTimesVector,
                          d_deviceFlattenedArrayBlock2,
                          d_YArray2,
                          d_projectorKetTimesVector2,
                          localVectorSize,
                          BVec,
                          chebyshevOrder,
                          d_lowerBoundUnWantedSpectrum,
                          d_upperBoundUnWantedSpectrum,
                          d_lowerBoundWantedSpectrum,
                          useMixedPrecOverall,
                          d_dftParams);
                      }
                    else
                      {
                        linearAlgebraOperationsDevice::chebyshevFilter(
                          operatorMatrix,
                          deviceFlattenedArrayBlock,
                          d_YArray,
                          d_deviceFlattenedFloatArrayBlock,
                          projectorKetTimesVector,
                          localVectorSize,
                          BVec,
                          chebyshevOrder,
                          d_lowerBoundUnWantedSpectrum,
                          d_upperBoundUnWantedSpectrum,
                          d_lowerBoundWantedSpectrum,
                          useMixedPrecOverall,
                          d_dftParams);
                      }

                    // copy current wavefunction vectors block to vector
                    // containing all wavefunction vectors
                    stridedCopyFromBlockKernel<<<
                      (BVec + (deviceConstants::blockSize - 1)) /
                        deviceConstants::blockSize * localVectorSize,
                      deviceConstants::blockSize>>>(
                      BVec,
                      localVectorSize,
                      deviceFlattenedArrayBlock.begin(),
                      totalNumberWaveFunctions,
                      eigenVectorsFlattenedDevice,
                      jvec);

                    if (d_dftParams.overlapComputeCommunCheby &&
                        numSimultaneousBlocksCurrent == 2)
                      stridedCopyFromBlockKernel<<<
                        (BVec + (deviceConstants::blockSize - 1)) /
                          deviceConstants::blockSize * localVectorSize,
                        deviceConstants::blockSize>>>(
                        BVec,
                        localVectorSize,
                        d_deviceFlattenedArrayBlock2.begin(),
                        totalNumberWaveFunctions,
                        eigenVectorsFlattenedDevice,
                        jvec + BVec);
                  }
                else
                  {
                    // set to zero wavefunctions which wont go through chebyshev
                    // filtering inside a given band group
                    setZeroKernel<<<(numSimultaneousBlocksCurrent * BVec +
                                     (deviceConstants::blockSize - 1)) /
                                      deviceConstants::blockSize *
                                      localVectorSize,
                                    deviceConstants::blockSize>>>(
                      numSimultaneousBlocksCurrent * BVec,
                      localVectorSize,
                      totalNumberWaveFunctions,
                      eigenVectorsFlattenedDevice,
                      jvec);
                  }

              } // cheby block loop
          }     // wfc block loop

        if (d_dftParams.verbosity >= 4)
          pcout << "ChebyShev Filtering Done: " << std::endl;


        linearAlgebraOperationsDevice::pseudoGramSchmidtOrthogonalization(
          elpaScala,
          eigenVectorsFlattenedDevice,
          localVectorSize,
          totalNumberWaveFunctions,
          d_mpiCommParent,
          operatorMatrix.getMPICommunicator(),
          devicecclMpiCommDomain,
          interBandGroupComm,
          cublasHandle,
          d_dftParams,
          useMixedPrecOverall);
      }

    //
    // scale the eigenVectors with M^{-1/2} to represent the wavefunctions in
    // the usual FE basis
    //
    scaleDeviceKernel<<<
      (totalNumberWaveFunctions + (deviceConstants::blockSize - 1)) /
        deviceConstants::blockSize * localVectorSize,
      deviceConstants::blockSize>>>(totalNumberWaveFunctions,
                                    localVectorSize,
                                    eigenVectorsFlattenedDevice,
                                    operatorMatrix.getInvSqrtMassVec());
  }


  //
  //
  //
  void
  chebyshevOrthogonalizedSubspaceIterationSolverDevice::
    densityMatrixEigenBasisFirstOrderResponse(
      operatorDFTDeviceClass &   operatorMatrix,
      dataTypes::numberDevice *  eigenVectorsFlattenedDevice,
      const unsigned int         flattenedSize,
      const unsigned int         totalNumberWaveFunctions,
      const std::vector<double> &eigenValues,
      const double               fermiEnergy,
      std::vector<double> &      densityMatDerFermiEnergy,
      DeviceCCLWrapper &         devicecclMpiCommDomain,
      const MPI_Comm &           interBandGroupComm,
      dftfe::elpaScalaManager &  elpaScala)
  {
    dealii::TimerOutput computingTimerStandard(
      operatorMatrix.getMPICommunicator(),
      pcout,
      d_dftParams.reproducible_output || d_dftParams.verbosity < 2 ?
        dealii::TimerOutput::never :
        dealii::TimerOutput::every_call,
      dealii::TimerOutput::wall_times);

    dftfe::utils::deviceSynchronize();
    computingTimerStandard.enter_subsection(
      "Density matrix first order response on Device");

    cublasHandle_t &cublasHandle = operatorMatrix.getCublasHandle();

    //
    // allocate memory for full flattened array on device and fill it up
    //
    const unsigned int localVectorSize =
      flattenedSize / totalNumberWaveFunctions;


    const unsigned int vectorsBlockSize =
      std::min(d_dftParams.chebyWfcBlockSize, totalNumberWaveFunctions);

    distributedDeviceVec<dataTypes::numberDevice> &deviceFlattenedArrayBlock =
      operatorMatrix.getParallelChebyBlockVectorDevice();

    distributedDeviceVec<dataTypes::numberDevice> &projectorKetTimesVector =
      operatorMatrix.getParallelProjectorKetTimesBlockVectorDevice();

    if (!d_isTemporaryParallelVectorsCreated)
      {
        d_YArray.reinit(deviceFlattenedArrayBlock);

        d_deviceFlattenedFloatArrayBlock.reinit(
          operatorMatrix.getMatrixFreeData()->get_vector_partitioner(),
          vectorsBlockSize);
      }

    //
    // scale the eigenVectors (initial guess of single atom wavefunctions or
    // previous guess) to convert into Lowden Orthonormalized FE basis
    // multiply by M^{1/2}
    scaleDeviceKernel<<<
      (totalNumberWaveFunctions + (deviceConstants::blockSize - 1)) /
        deviceConstants::blockSize * localVectorSize,
      deviceConstants::blockSize>>>(totalNumberWaveFunctions,
                                    localVectorSize,
                                    eigenVectorsFlattenedDevice,
                                    operatorMatrix.getSqrtMassVec());


    linearAlgebraOperationsDevice::densityMatrixEigenBasisFirstOrderResponse(
      operatorMatrix,
      eigenVectorsFlattenedDevice,
      deviceFlattenedArrayBlock,
      d_deviceFlattenedFloatArrayBlock,
      d_YArray,
      projectorKetTimesVector,
      localVectorSize,
      totalNumberWaveFunctions,
      d_mpiCommParent,
      operatorMatrix.getMPICommunicator(),
      devicecclMpiCommDomain,
      interBandGroupComm,
      eigenValues,
      fermiEnergy,
      densityMatDerFermiEnergy,
      elpaScala,
      cublasHandle,
      d_dftParams);



    //
    // scale the eigenVectors with M^{-1/2} to represent the wavefunctions in
    // the usual FE basis
    //
    scaleDeviceKernel<<<
      (totalNumberWaveFunctions + (deviceConstants::blockSize - 1)) /
        deviceConstants::blockSize * localVectorSize,
      deviceConstants::blockSize>>>(totalNumberWaveFunctions,
                                    localVectorSize,
                                    eigenVectorsFlattenedDevice,
                                    operatorMatrix.getInvSqrtMassVec());

    dftfe::utils::deviceSynchronize();
    computingTimerStandard.leave_subsection(
      "Density matrix first order response on Device");

    return;
  }

} // namespace dftfe
