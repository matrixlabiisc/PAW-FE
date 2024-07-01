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
// @author  Kartick Ramakrishnan
//

#ifndef DFTFE_PAWCLASS_H
#define DFTFE_PAWCLASS_H

#include "vector"
#include "map"
#include "AtomCenteredSphericalFunctionValenceDensitySpline.h"
#include "AtomCenteredSphericalFunctionCoreDensitySpline.h"
#include "AtomCenteredSphericalFunctionLocalPotentialSpline.h"
#include "AtomCenteredSphericalFunctionProjectorSpline.h"
#include "AtomCenteredSphericalFunctionPAWProjectorSpline.h"
#include "AtomCenteredSphericalFunctionPAWProjectorSpline2.h"
#include "AtomCenteredSphericalFunctionZeroPotentialSpline.h"
#include "AtomCenteredSphericalFunctionContainer.h"
#include "AtomicCenteredNonLocalOperator.h"
#include "AtomCenteredSphericalFunctionGaussian.h"
#include "AtomCenteredSphericalFunctionSinc.h"
#include "AtomCenteredSphericalFunctionBessel.h"
#include "wigner/gaunt.hpp"
#include <memory>
#include <MemorySpaceType.h>
#include <headers.h>
#include <TypeConfig.h>
#include <dftUtils.h>
#include "FEBasisOperations.h"
#include <BLASWrapper.h>
#include <xc.h>
#include <excManager.h>
#ifdef _OPENMP
#  include <omp.h>
#else
#  define omp_get_thread_num() 0
#endif
namespace dftfe
{
  enum class CouplingType
  {
    HamiltonianEntries,
    pawOverlapEntries,
    inversePawOverlapEntries
  };

  enum class TypeOfField
  {
    In,
    Out,
    Residual
  };


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  class pawClass
  {
  public:
    pawClass(const MPI_Comm &                            mpi_comm_parent,
             const std::string &                         scratchFolderName,
             dftParameters *                             dftParamsPtr,
             const std::set<unsigned int> &              atomTypes,
             const bool                                  floatingNuclearCharges,
             const unsigned int                          nOMPThreads,
             const std::map<unsigned int, unsigned int> &atomAttributes,
             const bool                                  reproducibleOutput,
             const int                                   verbosity,
             const bool                                  useDevice);
    /**
     * @brief Initialises all the data members with addresses/values to/of dftClass.
     * @param[in] densityQuadratureId quadratureId for density.
     * @param[in] localContributionQuadratureId quadratureId for local/zero
     * potential
     * @param[in] nuclearChargeQuadratureIdElectro quadratureId for nuclear
     * charges
     * @param[in] densityQuadratureIdElectro quadratureId for density in
     * Electrostatics mesh
     * @param[in] excFunctionalPtr address XC functional pointer
     * @param[in] numEigenValues number of eigenvalues
     * @param[in] atomLocations atomic Coordinates
     * @param[in] imageIds image IDs of periodic cell
     * @param[in] periodicCoords coordinates of image atoms
     */

    void
    initialise(
      std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
        basisOperationsHostPtr,
#if defined(DFTFE_WITH_DEVICE)
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<ValueType,
                                        double,
                                        dftfe::utils::MemorySpace::DEVICE>>
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
      std::map<dealii::CellId, std::vector<double>> &bQuadValuesAllAtoms,
      const bool singlePrecNonLocalOperator);

    /**
     * @brief Initialises all the data members with addresses/values to/of dftClass.
     * @param[in] densityQuadratureId quadratureId for density.
     * @param[in] localContributionQuadratureId quadratureId for local/zero
     * potential
     * @param[in] nuclearChargeQuadratureIdElectro quadratureId for nuclear
     * charges
     * @param[in] densityQuadratureIdElectro quadratureId for density in
     * Electrostatics mesh
     * @param[in] bQuadValuesAllAtoms address of nuclear charge field
     * @param[in] excFunctionalPtr address XC functional pointer
     * @param[in] numEigenValues number of eigenvalues
     * @param[in] atomLocations atomic Coordinates
     * @param[in] imageIds image IDs of periodic cell
     * @param[in] periodicCoords coordinates of image atoms
     */
    void
    initialiseNonLocalContribution(
      const std::vector<std::vector<double>> &atomLocations,
      const std::vector<int> &                imageIds,
      const std::vector<std::vector<double>> &periodicCoords,
      const std::vector<double> &             kPointWeights,
      const std::vector<double> &             kPointCoordinates,
      const bool                              updateNonlocalSparsity,
      const unsigned int                      dofHanderId);


    void
    initialiseNonLocalContribution(
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
      unsigned int                     numberElements);



    void
    computeCompensationCharge(TypeOfField typeOfField);

    void
    computeCompensationChargeMemoryOpt(TypeOfField typeOfField);

    /**
     * @brief pawclass omputecompensationchargel0:
     *
     */
    double
    TotalCompensationCharge();

    void
    computeDij(const bool         isDijOut,
               const unsigned int startVectorIndex,
               const unsigned int vectorBlockSize,
               const unsigned int spinIndex,
               const unsigned int kpointIndex);

    /**
     * @brief Initialises local potential
     */
    void
    initLocalPotential();

    void
    getRadialValenceDensity(unsigned int         Znum,
                            double               rad,
                            std::vector<double> &Val);

    double
    getRadialValenceDensity(unsigned int Znum, double rad);

    double
    getRmaxValenceDensity(unsigned int Znum);

    void
    getRadialCoreDensity(unsigned int         Znum,
                         double               rad,
                         std::vector<double> &Val);

    double
    getRadialCoreDensity(unsigned int Znum, double rad);

    double
    getRmaxCoreDensity(unsigned int Znum);

    double
    getRadialZeroPotential(unsigned int Znum, double rad);

    double
    getRmaxZeroPotential(unsigned int Znum);

    bool
    coreNuclearDensityPresent(unsigned int Znum);
    // Returns the number of Projectors for the given atomID in cooridnates List
    unsigned int
    getTotalNumberOfSphericalFunctionsForAtomId(unsigned int atomId);
    // Returns the Total Number of atoms with support in the processor
    unsigned int
    getTotalNumberOfAtomsInCurrentProcessor();
    // Returns the atomID in coordinates list for the iAtom index.
    unsigned int
    getAtomIdInCurrentProcessor(unsigned int iAtom);


    const dftfe::utils::MemoryStorage<ValueType, memorySpace> &
    getCouplingMatrix(
      CouplingType couplingtype = CouplingType::HamiltonianEntries);

    const dftfe::utils::MemoryStorage<
      typename dftfe::dataTypes::singlePrecType<ValueType>::type,
      memorySpace> &
    getCouplingMatrixSinglePrec(
      CouplingType couplingtype = CouplingType::HamiltonianEntries);


    const std::shared_ptr<
      AtomicCenteredNonLocalOperator<ValueType, memorySpace>>
    getNonLocalOperator();

    const std::shared_ptr<AtomicCenteredNonLocalOperator<
      typename dftfe::dataTypes::singlePrecType<ValueType>::type,
      memorySpace>>
    getNonLocalOperatorSinglePrec();

    void
    evaluateNonLocalHamiltonianElectrostaticsValue(
      const distributedCPUVec<double> &phiTotNodalValues,
      const unsigned int               dofHandlerId);

    void
    evaluateNonLocalHamiltonianElectrostaticsValue(
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &                phiTotQuadValues,
      const unsigned int dofHandlerId);

    void
    initialiseExchangeCorrelationEnergyCorrection(unsigned int s);

    void
    computeNonlocalPseudoPotentialConstants(CouplingType couplingtype,
                                            unsigned int s = 0);



    double
    computeDeltaExchangeCorrelationEnergy(double &DeltaExchangeCorrelationVal);



    std::vector<double>
    getDijWeights();

    std::vector<double>
    DijVectorForMixing(TypeOfField typeOfField);

    double
    densityScalingFactor(const std::vector<std::vector<double>> &atomLocations);

    void
    communicateDijAcrossAllProcessors(
      TypeOfField     typeOfField,
      const MPI_Comm &interpoolcomm,
      const MPI_Comm &interBandGroupComm,
      const bool      communicateDijAcrossAllProcessors = true);

    void
    computeDijFromPSIinitialGuess(
      const dftfe::utils::MemoryStorage<ValueType,
                                        dftfe::utils::MemorySpace::HOST> *X,
      const unsigned int         numberOfElectrons,
      const unsigned int         totalNumWaveFunctions,
      const unsigned int         quadratureIndex,
      const std::vector<double> &kPointWeights,
      const MPI_Comm &           interpoolcomm,
      const MPI_Comm &           interBandGroupComm);

    void
    chargeNeutrality(double      integralRhoValue,
                     TypeOfField typeOfField,
                     bool        computeCompCharge = true);

    void
    fillDijMatrix(TypeOfField                typeOfField,
                  const std::vector<double> &DijVector,
                  const MPI_Comm &           interpoolcomm,
                  const MPI_Comm &           interBandGroupComm);
    std::vector<double>
    getDeltaEnergy();

    void
    computeIntegralCoreDensity(
      const std::map<dealii::CellId, std::vector<double>> &rhoCore);

    double
    computeNormDij(std::vector<double> &DijResidual);

    void
    saveDijEntriesToFile();

    void
    loadDijEntriesFromFile();


  private:
    void
    createAtomTypesList(const std::vector<std::vector<double>> &atomLocations);

    void
    initialiseDataonRadialMesh();

    void
    initialiseColoumbicEnergyCorrection();

    void
    initialiseZeroPotential();



    void
    initialiseKineticEnergyCorrection();

    void
    computeRadialMultipoleData();

    void
    computeMultipoleInverse();

    void
    computeInverseOfMultipoleData();

    void
    computeCompensationChargeL0();

    void
    computeCompensationChargeCoeff();

    void
    computeCompensationChargeCoeffMemoryOpt();

    void
    computeproductOfCGMultipole();

    void
    saveDeltaSinverseEntriesToFile();

    int
    loadDeltaSinverseEntriesFromFile();

    std::map<unsigned int, std::vector<double>> d_KineticEnergyCorrectionTerm;
    std::map<unsigned int, std::vector<double>> d_zeroPotentialij;
    std::map<unsigned int, std::vector<double>>
                                                d_ExchangeCorrelationEnergyCorrectionTerm;
    std::map<unsigned int, std::vector<double>> d_ColoumbicEnergyCorrectionTerm;
    std::map<unsigned int, std::vector<double>> d_DeltaColoumbicEnergyDij;
    std::map<unsigned int, double>              d_coreKE, d_deltaC, d_coreXC,
      d_deltaValenceC;
    std::map<unsigned int, std::vector<double>> d_deltaCij, d_deltaCijkl;
    std::map<unsigned int, std::vector<double>>
                              d_nonLocalHamiltonianElectrostaticValue;
    unsigned int              d_nProjPerTask, d_nProjSqTotal, d_totalProjectors;
    std::vector<unsigned int> d_projectorStartIndex;
    std::vector<unsigned int> d_totalProjectorStartIndex;
    double                    d_TotalCompensationCharge;
    double d_integralCoreDensity, d_integrealCoreDensityRadial;
    std::map<unsigned int, double>                d_integralCoreDensityPerAtom;
    std::map<dealii::CellId, std::vector<double>> d_jxwcompensationCharge;

    /**
     * @brief Converts the periodic image data structure to relevant form for the container class
     * @param[in] atomLocations atomic Coordinates
     * @param[in] imageIds image IDs of periodic cell
     * @param[in] periodicCoords coordinates of image atoms
     * @param[out] imageIdsTemp image IDs of periodic cell
     * @param[out] imageCoordsTemp coordinates of image atoms
     */
    void
    setImageCoordinates(const std::vector<std::vector<double>> &atomLocations,
                        const std::vector<int> &                imageIds,
                        const std::vector<std::vector<double>> &periodicCoords,
                        std::vector<unsigned int> &             imageIdsTemp,
                        std::vector<double> &imageCoordsTemp);
    /**
     * @brief Creating Density splines for all atomTypes
     */
    void
    createAtomCenteredSphericalFunctionsForDensities();

    void
    createAtomCenteredSphericalFunctionsForShapeFunctions();


    void
    createAtomCenteredSphericalFunctionsForProjectors();
    void
    createAtomCenteredSphericalFunctionsForZeroPotential();



    std::complex<double>
    computeTransformationExtries(int l, int mu, int m);
    /**
     * @brief factorial: Recursive method to find factorial
     *
     *
     *  @param[in] n
     *
     */

    double
    factorial(double n)
    {
      if (n < 0)
        {
          // pcout<<"-ve Factorial"<<std::endl;
          return -100.;
        }
      return (n == 1. || n == 0.) ? 1. : factorial(n - 1) * n;
    }
    // Utils Functions
    double
    gaunt(int l_i, int l_j, int l, int m_i, int m_j, int m);
    double
    multipoleIntegrationGrid(double *             f1,
                             double *             f2,
                             std::vector<double> &radial,
                             std::vector<double> &rab,
                             const int            L,
                             const unsigned int   rminIndex,
                             const unsigned int   rmaxIndex);
    double
    simpsonIntegral(
      unsigned int                                 startIndex,
      unsigned int                                 EndIndex,
      std::function<double(const unsigned int &)> &IntegrandValue);

    // COmputes \int{f1(r)*f2(r)*f3(r)*r^2dr*J_r}
    double
    threeTermIntegrationOverAugmentationSphere(double *             f1,
                                               double *             f2,
                                               double *             f3,
                                               std::vector<double> &radial,
                                               std::vector<double> &rab,
                                               const unsigned int   rminIndex,
                                               const unsigned int   rmaxIndex);
    // Computes the potential due to charge fun
    void
    oneTermPoissonPotential(const double *             fun,
                            const unsigned int         l,
                            const unsigned int         rminIndex,
                            const unsigned int         rmaxIndex,
                            const int                  powerofR,
                            const std::vector<double> &radial,
                            const std::vector<double> &rab,
                            std::vector<double> &      Potential);

    void
    twoTermPoissonPotential(const double *             fun1,
                            const double *             fun2,
                            const unsigned int         l,
                            const unsigned int         rminIndex,
                            const unsigned int         rmaxIndex,
                            const int                  powerofR,
                            const std::vector<double> &radial,
                            const std::vector<double> &rab,
                            std::vector<double> &      Potential);

    double
    integralOfProjectorsInAugmentationSphere(const double *       f1,
                                             const double *       f2,
                                             std::vector<double> &radial,
                                             std::vector<double> &rab,
                                             const unsigned int   rminIndex,
                                             const unsigned int   rmaxIndex);

    double
    integralOfDensity(const double *       f1,
                      std::vector<double> &radial,
                      std::vector<double> &rab,
                      const unsigned int   rminIndex,
                      const unsigned int   rmaxIndex);
    void
    getSphericalQuadratureRule(std::vector<double> &             quad_weights,
                               std::vector<std::vector<double>> &quad_points);

    void
    computeCoreDeltaExchangeCorrelationEnergy();


    void
    computeAugmentationOverlap();

    void
    checkOverlapAugmentation();

    std::vector<unsigned int>
    relevantAtomdIdsInCurrentProcs();


    std::vector<double>
    derivativeOfRealSphericalHarmonic(unsigned int lQuantumNo,
                                      int          mQuantumNo,
                                      double       theta,
                                      double       phi);
    std::vector<double>
    radialDerivativeOfMeshData(const std::vector<double> &r,
                               const std::vector<double> &rab,
                               const std::vector<double> &functionValue);

    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
      d_BLASWrapperHostPtr;
#if defined(DFTFE_WITH_DEVICE)
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
      d_BLASWrapperDevicePtr;
#endif
    std::vector<std::vector<double>> d_nonLocalPseudoPotentialConstants;
    std::map<CouplingType, std::map<unsigned int, std::vector<ValueType>>>
      d_atomicNonLocalPseudoPotentialConstants;
    std::map<CouplingType, dftfe::utils::MemoryStorage<ValueType, memorySpace>>
      d_couplingMatrixEntries;
    std::map<CouplingType,
             dftfe::utils::MemoryStorage<
               typename dftfe::dataTypes::singlePrecType<ValueType>::type,
               memorySpace>>
         d_couplingMatrixEntriesSinglePrec;
    bool d_HamiltonianCouplingMatrixEntriesUpdated,
      d_overlapCouplingMatrixEntriesUpdated,
      d_inverseCouplingMatrixEntriesUpdated;
    bool d_HamiltonianCouplingMatrixEntriesUpdatedSinglePrec,
      d_overlapCouplingMatrixEntriesUpdatedSinglePrec,
      d_inverseCouplingMatrixEntriesUpdatedSinglePrec;
    std::vector<std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      d_atomicWaveFnsVector;
    std::shared_ptr<AtomCenteredSphericalFunctionContainer>
      d_atomicProjectorFnsContainer;

    std::shared_ptr<AtomCenteredSphericalFunctionContainer>
      d_atomicShapeFnsContainer;

    std::map<std::pair<unsigned int, unsigned int>,
             std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      d_atomicProjectorFnsMap, d_atomicAEPartialWaveFnsMap,
      d_atomicPSPartialWaveFnsMap;

    std::map<std::pair<unsigned int, unsigned int>,
             std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      d_atomicShapeFnsMap;
    // parallel communication objects
    const MPI_Comm     d_mpiCommParent;
    const unsigned int d_this_mpi_process;
    const unsigned int d_n_mpi_processes;

    // conditional stream object
    dealii::ConditionalOStream pcout;
    bool                       d_useDevice;
    unsigned int               d_densityQuadratureId;
    unsigned int               d_compensationChargeQuadratureIdElectro;
    unsigned int               d_localContributionQuadratureId;
    unsigned int               d_nuclearChargeQuadratureIdElectro;
    unsigned int               d_densityQuadratureIdElectro;
    unsigned int               d_sparsityPatternQuadratureId;
    unsigned int               d_nlpspQuadratureId;
    bool                       d_singlePrecNonLocalOperator;
    // unsigned int                d_dofHandlerID;
    std::shared_ptr<excManager> d_excManagerPtr;
    dftParameters *             d_dftParamsPtr;
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
      d_BasisOperatorHostPtr;
#if defined(DFTFE_WITH_DEVICE)
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::DEVICE>>
      d_BasisOperatorDevicePtr;
#endif
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
      d_BasisOperatorElectroHostPtr;
#if defined(DFTFE_WITH_DEVICE)
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::DEVICE>>
      d_BasisOperatorElectroDevicePtr;
#endif
    std::map<unsigned int, std::vector<double>>
                                                d_ProductOfQijShapeFnAtQuadPoints;
    std::map<unsigned int, std::vector<double>> d_shapeFnAtQuadPoints;

    std::map<TypeOfField, std::map<unsigned int, std::vector<double>>> D_ij;
    std::map<unsigned int, std::vector<double>> d_multipole, d_multipoleInverse;
    std::map<unsigned int, std::vector<double>>
                        d_productOfMultipoleClebshGordon;
    std::vector<double> d_deltaInverseMatrix, d_deltaMatrix;
    std::map<std::pair<unsigned int, unsigned int>, std::vector<double>>
                                         d_gLValuesQuadPoints;
    std::map<unsigned int, double>       d_DeltaL0coeff, d_NtildeCore;
    std::map<unsigned int, double>       d_RmaxAug, d_RminAug, d_RmaxComp;
    std::map<unsigned int, unsigned int> d_RmaxAugIndex;

    // Radial Data on meshGrid
    std::map<unsigned int, std::vector<double>> d_radialMesh,
      d_radialJacobianData;
    std::map<unsigned int, double> d_radialValueCoreSmooth0,
      d_radialValueCoreAE0;
    std::map<unsigned int, std::vector<double>> d_PSWfc0, d_AEWfc0;
    std::map<unsigned int, std::vector<double>> d_productOfAEpartialWfc,
      d_productOfPSpartialWfc, d_atomCoreDensityAE, d_atomCoreDensityPS,
      d_atomicShapeFn;
    std::map<unsigned int, std::vector<double>>
                                                d_productDerCoreDensityWfcDerWfcAE, d_productDerCoreDensityWfcDerWfcPS;
    std::map<unsigned int, std::vector<double>> d_productOfPSpartialWfcDer,
      d_productOfAEpartialWfcDer;
    std::map<unsigned int, std::vector<double>> d_productOfPSpartialWfcValue,
      d_productOfAEpartialWfcValue;
    std::map<unsigned int, std::vector<double>> d_gradCoreSqAE, d_gradCoreSqPS;
    std::map<unsigned int, std::vector<double>> d_radialWfcDerAE,
      d_radialWfcValAE, d_radialWfcDerPS, d_radialWfcValPS, d_radialCoreDerAE,
      d_radialCoreDerPS;
    std::map<unsigned int, std::vector<double>> d_zeroPotentialRadialValues;
    // Total Comepsantion charge field
    std::map<dealii::CellId, std::vector<double>> *d_bQuadValuesAllAtoms;
    // Total Compensation charge field only due to the g_0(r)Delta_0 component
    std::map<dealii::CellId, std::vector<double>>     d_bl0QuadValuesAllAtoms;
    distributedCPUVec<ValueType>                      Pmatrix;
    std::map<unsigned int, bool>                      d_atomTypeCoreFlagMap;
    bool                                              d_floatingNuclearCharges;
    int                                               d_verbosity;
    std::set<unsigned int>                            d_atomTypes;
    std::map<unsigned int, std::vector<unsigned int>> d_atomTypesList;
    std::vector<unsigned int>                         d_LocallyOwnedAtomId;
    std::string                                       d_dftfeScratchFolderName;
    std::vector<int>                                  d_imageIds;
    std::vector<std::vector<double>>                  d_imagePositions;
    unsigned int                                      d_numEigenValues;
    unsigned int                                      d_nOMPThreads;

    std::vector<double> d_kpointWeights;

    // Creating Object for Atom Centerd Nonlocal Operator
    std::shared_ptr<AtomicCenteredNonLocalOperator<ValueType, memorySpace>>
      d_nonLocalOperator;
    // Creating Object for Atom Centerd Nonlocal Operator SinglePrec
    std::shared_ptr<AtomicCenteredNonLocalOperator<
      typename dftfe::dataTypes::singlePrecType<ValueType>::type,
      memorySpace>>
      d_nonLocalOperatorSinglePrec;

    std::vector<std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      d_atomicProjectorFnsVector;
    std::vector<std::map<unsigned int,
                         std::shared_ptr<AtomCenteredSphericalFunctionBase>>>
      d_atomicZeroPotVector;
    std::vector<std::map<unsigned int,
                         std::shared_ptr<AtomCenteredSphericalFunctionBase>>>
      d_atomicValenceDensityVector;
    std::vector<std::map<unsigned int,
                         std::shared_ptr<AtomCenteredSphericalFunctionBase>>>
         d_atomicCoreDensityVector;
    bool d_reproducible_output;
    /// FIXME: eventually it should be a map of atomic number to struct-
    /// {valence number, mesh input etc}
    std::map<unsigned int, unsigned int> d_atomTypeAtributes;



  }; // end of class

} // end of namespace dftfe
//#include "../src/pseudo/paw/pawClassInit.t.cc"
//#include "../src/pseudo/paw/pawClass.t.cc"
//#include "../src/pseudo/paw/pawClassUtils.t.cc"
//#include "../src/pseudo/paw/pawClassEnergy.t.cc"
#endif //  DFTFE_PSEUDOPOTENTIALBASE_H
