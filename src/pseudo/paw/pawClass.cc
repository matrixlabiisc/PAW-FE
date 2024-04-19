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
  pawClass<ValueType, memorySpace>::computeCompensationChargeL0()
  {
    d_BasisOperatorElectroHostPtr->reinit(
      0, 0, d_compensationChargeQuadratureIdElectro);

    const unsigned int numberNodesPerElement =
      d_BasisOperatorElectroHostPtr->nDofsPerCell();
    std::vector<dealii::types::global_dof_index> cellGlobalDofIndices(
      numberNodesPerElement);
    const unsigned int numberAtomsOfInterest =
      d_atomicShapeFnsContainer->getNumAtomCentersSize();
    const unsigned int numberQuadraturePoints =
      d_BasisOperatorElectroHostPtr->nQuadsPerCell();
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                                     quadraturePointsVector = d_BasisOperatorElectroHostPtr->quadPoints();
    const std::vector<unsigned int> &atomicNumber =
      d_atomicShapeFnsContainer->getAtomicNumbers();
    const std::vector<double> &atomCoordinates =
      d_atomicShapeFnsContainer->getAtomCoordinates();
    const std::map<unsigned int, std::vector<double>> &periodicImageCoord =
      d_atomicShapeFnsContainer->getPeriodicImageCoordinatesList();
    const std::map<std::pair<unsigned int, unsigned int>,
                   std::shared_ptr<AtomCenteredSphericalFunctionBase>>
                                    sphericalFunction = d_atomicShapeFnsContainer->getSphericalFunctions();
    const std::vector<unsigned int> atomIdsInCurrentProcess =
      d_atomicShapeFnsContainer->getAtomIdsInCurrentProcess();



    for (unsigned int iAtom = 0; iAtom < atomIdsInCurrentProcess.size();
         iAtom++)
      {
        const unsigned int  atomId = atomIdsInCurrentProcess[iAtom];
        const unsigned int  Znum   = atomicNumber[atomId];
        std::vector<double> imageCoordinates =
          periodicImageCoord.find(atomId)->second;
        const unsigned int imageIdsSize = imageCoordinates.size() / 3;
        std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn =
          sphericalFunction.find(std::make_pair(Znum, 0))->second;
        std::vector<unsigned int> elementIndexesInAtomCompactSupport =
          d_atomicShapeFnsContainer
            ->d_elementIndexesInAtomCompactSupport[atomId];
        int numberElementsInAtomCompactSupport =
          elementIndexesInAtomCompactSupport.size();
        dealii::Point<3> nuclearCoordinates(atomCoordinates[3 * atomId + 0],
                                            atomCoordinates[3 * atomId + 1],
                                            atomCoordinates[3 * atomId + 2]);
        double           dL0     = d_DeltaL0coeff[Znum];
        double           RmaxAug = d_RmaxAug[Znum];

        for (int iElemComp = 0; iElemComp < numberElementsInAtomCompactSupport;
             iElemComp++)
          {
            const unsigned int elementIndex =
              elementIndexesInAtomCompactSupport[iElemComp];

            std::vector<double> &quadvalues =
              d_bl0QuadValuesAllAtoms[d_BasisOperatorElectroHostPtr->cellID(
                elementIndex)];
            if (quadvalues.size() != numberQuadraturePoints)
              quadvalues.resize(numberQuadraturePoints, 0.0);
            for (int iImageAtomCount = 0; iImageAtomCount < imageIdsSize;
                 ++iImageAtomCount)
              {
                dealii::Point<3> chargePoint(0.0, 0.0, 0.0);
                if (iImageAtomCount == 0)
                  {
                    chargePoint = nuclearCoordinates;
                  }
                else
                  {
                    chargePoint[0] = imageCoordinates[3 * iImageAtomCount + 0];
                    chargePoint[1] = imageCoordinates[3 * iImageAtomCount + 1];
                    chargePoint[2] = imageCoordinates[3 * iImageAtomCount + 2];
                  }
                double x[3];
                double sphericalHarmonicVal, radialVal, sphericalFunctionValue;
                double r, theta, phi, angle;

                for (int iQuadPoint = 0; iQuadPoint < numberQuadraturePoints;
                     ++iQuadPoint)
                  {
                    x[0] = quadraturePointsVector[elementIndex *
                                                    numberQuadraturePoints * 3 +
                                                  3 * iQuadPoint] -
                           chargePoint[0];
                    x[1] = quadraturePointsVector[elementIndex *
                                                    numberQuadraturePoints * 3 +
                                                  3 * iQuadPoint + 1] -
                           chargePoint[1];
                    x[2] = quadraturePointsVector[elementIndex *
                                                    numberQuadraturePoints * 3 +
                                                  3 * iQuadPoint + 2] -
                           chargePoint[2];
                    sphericalHarmonicUtils::convertCartesianToSpherical(x,
                                                                        r,
                                                                        theta,
                                                                        phi);
                    if (r <= sphFn->getRadialCutOff())
                      {
                        radialVal = sphFn->getRadialValue(r);
                        quadvalues[iQuadPoint] +=
                          dL0 * radialVal / sqrt(4 * M_PI);



                      } // inside r <= Rmax
                      //pcout<<"q_L0 at qpoint: "<<iElemComp<<" "<<quadvalues[iQuadPoint]<<std::endl; 

                  } // quad loop

              } // image atom loop

          } // iElemComp

      } // iAtom
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeCompensationCharge(
    TypeOfField typeOfField)
  {
    d_BasisOperatorElectroHostPtr->reinit(0,0,d_compensationChargeQuadratureIdElectro);
    const unsigned int numberAtomsOfInterest =
      d_atomicShapeFnsContainer->getNumAtomCentersSize();
    const unsigned int numberQuadraturePoints =
      d_BasisOperatorElectroHostPtr->nQuadsPerCell();
    const std::vector<unsigned int> &atomicNumber =
      d_atomicShapeFnsContainer->getAtomicNumbers();
    const unsigned int one = 1;
    (*d_bQuadValuesAllAtoms).clear();
    for (std::map<dealii::CellId, std::vector<double>>::iterator it =
           d_bl0QuadValuesAllAtoms.begin();
         it != d_bl0QuadValuesAllAtoms.end();
         ++it)
      {
        std::vector<double> &ValueL0 = it->second;
        std::vector<double>  Temp;
        (*d_bQuadValuesAllAtoms).find(it->first)->second.clear();
        // for (unsigned int q_point = 0; q_point < numberQuadraturePoints;
        //      q_point++)
        //   {
        //     Temp.push_back(ValueL0[q_point]);
        //   }
        (*d_bQuadValuesAllAtoms)[it->first] = ValueL0;
      }
    pcout<<"DEBUG: Line 170"<<std::endl;
    const std::vector<unsigned int> atomIdsInCurrentProcess =
      d_atomicShapeFnsContainer->getAtomIdsInCurrentProcess();



    for (unsigned int iAtom = 0; iAtom < atomIdsInCurrentProcess.size();
         iAtom++)
      {
        unsigned int atomId = atomIdsInCurrentProcess[iAtom];
        unsigned int Znum   = atomicNumber[atomId];
        unsigned int numberOfProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        unsigned int        npjsq = numberOfProjectors * numberOfProjectors;
        std::vector<double> Tij   = d_ProductOfQijShapeFnAtQuadPoints[atomId];
        pcout<<"Size of Tij: "<<Tij.size()<<std::endl;        
        std::vector<double> Dij   = D_ij[typeOfField][atomId];
        pcout<<"Size of Dij: "<<Dij.size()<<std::endl;
        std::vector<unsigned int> elementIndexesInAtomCompactSupport =
          d_atomicShapeFnsContainer
            ->d_elementIndexesInAtomCompactSupport[atomId];
        int numberElementsInAtomCompactSupport =
          elementIndexesInAtomCompactSupport.size();
        for (unsigned int iElem = 0; iElem < numberElementsInAtomCompactSupport;
             iElem++)
          {
            const unsigned int elementIndex =
              elementIndexesInAtomCompactSupport[iElem];
            pcout<<"ElementIndex: "<<elementIndex<<" "<<iElem<<std::endl;  
            std::vector<double> &quadvalues =
              (*d_bQuadValuesAllAtoms)[d_BasisOperatorElectroHostPtr->cellID(
                elementIndex)];
            for (unsigned int q_point = 0; q_point < numberQuadraturePoints;
                 ++q_point)
              {
                long unsigned int loc =
                  iElem * npjsq * numberQuadraturePoints + q_point * npjsq;
                quadvalues[q_point] +=
                  ddot_(&npjsq, &Tij[loc], &one, &Dij[0], &one);
              } // q_point

          } // iElem
      }     // iAtom loop
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeCompensationChargeCoeff()
  {
    d_BasisOperatorElectroHostPtr->reinit(
      0, 0, d_compensationChargeQuadratureIdElectro);

    const unsigned int numberNodesPerElement =
      d_BasisOperatorElectroHostPtr->nDofsPerCell();
    std::vector<dealii::types::global_dof_index> cellGlobalDofIndices(
      numberNodesPerElement);
    const unsigned int numberAtomsOfInterest =
      d_atomicShapeFnsContainer->getNumAtomCentersSize();
    const unsigned int numberQuadraturePoints =
      d_BasisOperatorElectroHostPtr->nQuadsPerCell();
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      quadraturePointsVector = d_BasisOperatorElectroHostPtr->quadPoints();
    pcout<<"Size of quadPoints: "<<quadraturePointsVector.size()<<std::endl;
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      JxwVector = d_BasisOperatorElectroHostPtr->JxW();
    pcout<<"Size of JxwVEctor: "<<JxwVector.size()<<std::endl;
    const std::vector<unsigned int> &atomicNumber =
      d_atomicShapeFnsContainer->getAtomicNumbers();
    const std::vector<double> &atomCoordinates =
      d_atomicShapeFnsContainer->getAtomCoordinates();
    const std::map<unsigned int, std::vector<double>> &periodicImageCoord =
      d_atomicShapeFnsContainer->getPeriodicImageCoordinatesList();
    const std::map<std::pair<unsigned int, unsigned int>,
                   std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      sphericalFunction = d_atomicShapeFnsContainer->getSphericalFunctions();
    const std::map<std::pair<unsigned int, unsigned int>,
                   std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      projectorFunction =
        d_atomicProjectorFnsContainer->getSphericalFunctions();
    const std::vector<unsigned int> atomIdsInCurrentProcess =
      d_atomicShapeFnsContainer->getAtomIdsInCurrentProcess();



    for (unsigned int iAtom = 0; iAtom < atomIdsInCurrentProcess.size();
         iAtom++)
      {
        const unsigned int  atomId = atomIdsInCurrentProcess[iAtom];
        const unsigned int  Znum   = atomicNumber[atomId];
        std::vector<double> imageCoordinates =
          periodicImageCoord.find(atomId)->second;
        const unsigned int imageIdsSize = imageCoordinates.size() / 3;
        std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn =
          sphericalFunction.find(std::make_pair(Znum, 0))->second;
        std::vector<unsigned int> elementIndexesInAtomCompactSupport =
          d_atomicShapeFnsContainer
            ->d_elementIndexesInAtomCompactSupport[atomId];
        int numberElementsInAtomCompactSupport =
          elementIndexesInAtomCompactSupport.size();
        dealii::Point<3>    nuclearCoordinates(atomCoordinates[3 * atomId + 0],
                                            atomCoordinates[3 * atomId + 1],
                                            atomCoordinates[3 * atomId + 2]);
        double              RmaxAug   = d_RmaxAug[Znum];
        std::vector<double> multipole = d_multipole[Znum];
        const unsigned int  NumRadialSphericalFunctions =
          d_atomicShapeFnsContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);
        const unsigned int NumTotalSphericalFunctions =
          d_atomicShapeFnsContainer->getTotalNumberOfSphericalFunctionsPerAtom(
            Znum);
        const unsigned int NumProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        const unsigned int NumRadialProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);
        const unsigned int  numProjSq = NumProjectors * NumProjectors;
        std::vector<double> tempCoeff(numberElementsInAtomCompactSupport *
                                        numberQuadraturePoints * numProjSq,
                                      0.0);
        for (int iElemComp = 0; iElemComp < numberElementsInAtomCompactSupport;
             iElemComp++)
          {
            
            const unsigned int elementIndex =
              elementIndexesInAtomCompactSupport[iElemComp];
              //pcout<<"iElemComp: "<<iElemComp<<" elementIndex: "<<elementIndex<<std::endl;
            unsigned int        Lindex = 0;
            std::vector<double> gLValuesQuadPoints(numberQuadraturePoints *
                                                     NumTotalSphericalFunctions,
                                                   0.0);
            //pcout<<"Size of gLValueQuadPoints: "<<numberQuadraturePoints* NumTotalSphericalFunctions<<std::endl;                                      
            for (unsigned int alpha = 0; alpha < NumRadialSphericalFunctions;
                 ++alpha)
              {
                //pcout<<"Lindex: "<<Lindex<<std::endl;
                std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn =
                  sphericalFunction.find(std::make_pair(Znum, alpha))->second;
                int lQuantumNumber = sphFn->getQuantumNumberl();

                for (int mQuantumNumber = -lQuantumNumber;
                     mQuantumNumber <= lQuantumNumber;
                     mQuantumNumber++)
                  {
                    for (int iImageAtomCount = 0;
                         iImageAtomCount < imageIdsSize;
                         ++iImageAtomCount)
                      {
                        dealii::Point<3> chargePoint(0.0, 0.0, 0.0);
                        if (iImageAtomCount == 0)
                          {
                            chargePoint = nuclearCoordinates;
                          }
                        else
                          {
                            chargePoint[0] =
                              imageCoordinates[3 * iImageAtomCount + 0];
                            chargePoint[1] =
                              imageCoordinates[3 * iImageAtomCount + 1];
                            chargePoint[2] =
                              imageCoordinates[3 * iImageAtomCount + 2];
                          }
                        double x[3];
                        double sphericalHarmonicVal, radialVal,
                          sphericalFunctionValue;
                        double r, theta, phi, angle;

                        for (int iQuadPoint = 0;
                             iQuadPoint < numberQuadraturePoints;
                             ++iQuadPoint)
                          {
                            x[0] =
                              quadraturePointsVector[elementIndex *
                                                       numberQuadraturePoints *
                                                       3 +
                                                     3 * iQuadPoint] -
                              chargePoint[0];
                            x[1] =
                              quadraturePointsVector[elementIndex *
                                                       numberQuadraturePoints *
                                                       3 +
                                                     3 * iQuadPoint + 1] -
                              chargePoint[1];
                            x[2] =
                              quadraturePointsVector[elementIndex *
                                                       numberQuadraturePoints *
                                                       3 +
                                                     3 * iQuadPoint + 2] -
                              chargePoint[2];
                            sphericalHarmonicUtils::convertCartesianToSpherical(
                              x, r, theta, phi);
                            sphericalHarmonicUtils::getSphericalHarmonicVal(
                              theta,
                              phi,
                              lQuantumNumber,
                              mQuantumNumber,
                              sphericalHarmonicVal);
                            if (r <= sphFn->getRadialCutOff())
                              {
                                radialVal = sphFn->getRadialValue(r);
                                sphericalFunctionValue =
                                  radialVal * sphericalHarmonicVal;

                                unsigned int alpha_i = 0;
                                for (int i = 0; i < NumRadialProjectors; i++)
                                  {
                                    std::shared_ptr<
                                      AtomCenteredSphericalFunctionBase>
                                      projFnI = projectorFunction
                                                  .find(std::make_pair(Znum, i))
                                                  ->second;
                                    int l_i = projFnI->getQuantumNumberl();
                                    for (int m_i = -l_i; m_i <= l_i; m_i++)
                                      {
                                        unsigned int alpha_j = 0;
                                        for (int j = 0; j < NumRadialProjectors;
                                             j++)
                                          {
                                            std::shared_ptr<
                                              AtomCenteredSphericalFunctionBase>
                                              projFnJ =
                                                projectorFunction
                                                  .find(std::make_pair(Znum, j))
                                                  ->second;
                                            int l_j =
                                              projFnJ->getQuantumNumberl();
                                            for (int m_j = l_j; m_j <= l_j;
                                                 m_j++)
                                              {
                                                double multipolevalue =
                                                  multipole
                                                    [lQuantumNumber *
                                                       NumRadialProjectors *
                                                       NumRadialProjectors +
                                                     i * NumRadialProjectors +
                                                     j];
                                                double Cijl =
                                                  gaunt(l_i,
                                                        l_j,
                                                        lQuantumNumber,
                                                        m_i,
                                                        m_j,
                                                        mQuantumNumber);
                                                if (std::fabs(multipolevalue) <
                                                      1E-16 ||
                                                    std::fabs(Cijl) < 1E-16)
                                                  continue;
 
                                                long unsigned int loc =
                                                  iElemComp *
                                                    (numberQuadraturePoints *
                                                     numProjSq) +
                                                  iQuadPoint * (numProjSq) +
                                                  alpha_i * NumProjectors +
                                                  alpha_j;
                                                //pcout<<loc<<" "<<alpha_i<<" "<<alpha_j<<" "<<iElemComp<<" "<<lQuantumNumber *
                                                    //    NumRadialProjectors *
                                                    //    NumRadialProjectors +
                                                    //  i * NumRadialProjectors +
                                                    //  j<<" "<<iQuadPoint<<std::endl; 
                                                if (r <= RmaxAug)
                                                  tempCoeff[loc] +=
                                                    Cijl * multipolevalue *
                                                    sphericalFunctionValue;
                                                else
                                                  tempCoeff[loc] += 0.0;

                                                alpha_j++;
                                              } // m_j
                                          }     // j

                                        alpha_i++;
                                      } // m_i
                                  }     // i loop
                                gLValuesQuadPoints[Lindex * numberQuadraturePoints+iQuadPoint] +=
                                  JxwVector[elementIndex *
                                              numberQuadraturePoints +
                                            iQuadPoint] *
                                  sphericalFunctionValue;

                              } // inside r <= Rmax


                          } // quad loop

                      } // image atom loop

                    Lindex++;
                  } // mQuantumNumber
              }     // alpha
            d_gLValuesQuadPoints[std::make_pair(atomId, elementIndex)] =
              gLValuesQuadPoints;
          } // iElemComp

        d_ProductOfQijShapeFnAtQuadPoints[iAtom] = tempCoeff;


      } // iAtom
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const dftfe::utils::MemoryStorage<ValueType, memorySpace> &
  pawClass<ValueType, memorySpace>::getCouplingMatrix(CouplingType couplingtype)
  {
    if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
      {
        if (couplingtype == CouplingType::HamiltonianEntries)
          {
            if (!d_HamiltonianCouplingMatrixEntriesUpdated)
              {
                computeNonlocalPseudoPotentialConstants(
                  CouplingType::HamiltonianEntries);
                std::vector<ValueType> Entries;
                dftfe::utils::MemoryStorage<ValueType,
                                            dftfe::utils::MemorySpace::HOST>
                                                couplingEntriesHost;
                const std::vector<unsigned int> atomIdsInProcessor =
                  d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
                std::vector<unsigned int> atomicNumber =
                  d_atomicProjectorFnsContainer->getAtomicNumbers();
                for (int iAtom = 0; iAtom < atomIdsInProcessor.size(); iAtom++)
                  {
                    unsigned int atomId = atomIdsInProcessor[iAtom];
                    unsigned int Zno    = atomicNumber[atomId];
                    unsigned int numberSphericalFunctions =
                      d_atomicProjectorFnsContainer
                        ->getTotalNumberOfSphericalFunctionsPerAtom(Zno);
                    for (unsigned int alpha = 0;
                         alpha < numberSphericalFunctions;
                         alpha++)
                      {
                        double V = d_atomicNonLocalPseudoPotentialConstants
                          [CouplingType::HamiltonianEntries][atomId][alpha];
                        Entries.push_back(V);
                      }
                  }
                couplingEntriesHost.resize(Entries.size());
                couplingEntriesHost.copyFrom(Entries);
                d_couplingMatrixEntries[CouplingType::HamiltonianEntries] =
                  couplingEntriesHost;
                d_HamiltonianCouplingMatrixEntriesUpdated = true;
              }
          }
        else if (couplingtype == CouplingType::pawOverlapEntries)
          {
            if (!d_overlapCouplingMatrixEntriesUpdated)
              {
                std::vector<ValueType> Entries;
                dftfe::utils::MemoryStorage<ValueType,
                                            dftfe::utils::MemorySpace::HOST>
                                                couplingEntriesHost;
                const std::vector<unsigned int> atomIdsInProcessor =
                  d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
                std::vector<unsigned int> atomicNumber =
                  d_atomicProjectorFnsContainer->getAtomicNumbers();
                for (int iAtom = 0; iAtom < atomIdsInProcessor.size(); iAtom++)
                  {
                    unsigned int atomId = atomIdsInProcessor[iAtom];
                    unsigned int Zno    = atomicNumber[atomId];
                    unsigned int numberSphericalFunctions =
                      d_atomicProjectorFnsContainer
                        ->getTotalNumberOfSphericalFunctionsPerAtom(Zno);
                    for (unsigned int alpha = 0;
                         alpha < numberSphericalFunctions;
                         alpha++)
                      {
                        double V = d_atomicNonLocalPseudoPotentialConstants
                          [CouplingType::pawOverlapEntries][Zno][alpha];
                        Entries.push_back(V);
                      }
                  }
                couplingEntriesHost.resize(Entries.size());
                couplingEntriesHost.copyFrom(Entries);
                d_couplingMatrixEntries[CouplingType::pawOverlapEntries] =
                  couplingEntriesHost;
                d_overlapCouplingMatrixEntriesUpdated = true;
              }
          }
        else if (couplingtype == CouplingType::inversePawOverlapEntries)
          {
            if (!d_inverseCouplingMatrixEntriesUpdated)
              {
                std::vector<ValueType> Entries;
                dftfe::utils::MemoryStorage<ValueType,
                                            dftfe::utils::MemorySpace::HOST>
                  couplingEntriesHost;

                const std::vector<unsigned int> atomIdsInProcessor =
                  d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
                std::vector<unsigned int> atomicNumber =
                  d_atomicProjectorFnsContainer->getAtomicNumbers();
                for (int iAtom = 0; iAtom < atomIdsInProcessor.size(); iAtom++)
                  {
                    unsigned int atomId = atomIdsInProcessor[iAtom];
                    unsigned int Zno    = atomicNumber[atomId];
                    unsigned int numberSphericalFunctions =
                      d_atomicProjectorFnsContainer
                        ->getTotalNumberOfSphericalFunctionsPerAtom(Zno);
                    for (unsigned int alpha = 0;
                         alpha < numberSphericalFunctions;
                         alpha++)
                      {
                        double V = d_atomicNonLocalPseudoPotentialConstants
                          [CouplingType::inversePawOverlapEntries][atomId]
                          [alpha];
                        Entries.push_back(V);
                      }
                  }
                couplingEntriesHost.resize(Entries.size());
                couplingEntriesHost.copyFrom(Entries);
                d_couplingMatrixEntries
                  [CouplingType::inversePawOverlapEntries] =
                    couplingEntriesHost;
                d_inverseCouplingMatrixEntriesUpdated = true;
              }
          }
      }
    else
      {
        AssertThrow(dftfe::utils::MemorySpace::HOST == memorySpace,
                    dealii::ExcMessage(
                      "DFT-FE Error: Not yet implemented on GPUs."));
      }

    return d_couplingMatrixEntries[couplingtype];
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeDij(
    const bool         isDijOut,
    const unsigned int startVectorIndex,
    const unsigned int vectorBlockSize,
    const unsigned int spinIndex,
    const unsigned int kpointIndex)
  {
    const std::vector<unsigned int> atomIdsInProcessor =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
    std::vector<unsigned int> atomicNumber =
      d_atomicProjectorFnsContainer->getAtomicNumbers();
    char transB = 'N';
#ifdef USE_COMPLEX
    char transA = 'C';
#else
    char transA = 'T';
#endif
    const ValueType beta  = 0.0;
    const ValueType alpha = 1.0;
    pcout << "COmputing DIJ" << std::endl;
    for (int iAtom = 0; iAtom < atomIdsInProcessor.size(); iAtom++)
      {
        const unsigned int atomId = atomIdsInProcessor[iAtom];
        const unsigned int Znum   = atomicNumber[atomId];
        const unsigned int numberSphericalFunctions =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        if (startVectorIndex == 0)
          D_ij[isDijOut ? TypeOfField::Out : TypeOfField::In][atomId] =
            std::vector<double>(numberSphericalFunctions *
                                  numberSphericalFunctions,
                                0.0);
        std::vector<ValueType> tempDij(numberSphericalFunctions *
                                         numberSphericalFunctions,
                                       0.0);
        pcout << "U Matrix Entries" << std::endl;
        for (int i = 0; i < numberSphericalFunctions * vectorBlockSize; i++)
          pcout << *(d_nonLocalOperator->getCconjtansXLocalDataStructure(
                       atomId) +
                     i)
                << std::endl;
        d_BLASWrapperHostPtr->xgemm(
          transA,
          transB,
          numberSphericalFunctions,
          numberSphericalFunctions,
          vectorBlockSize,
          &alpha,
          d_nonLocalOperator->getCconjtansXLocalDataStructure(atomId),
          vectorBlockSize,
          d_nonLocalOperator->getCconjtansXLocalDataStructure(atomId),
          vectorBlockSize,
          &beta,
          &tempDij[0],
          numberSphericalFunctions);
        std::transform(
          D_ij[isDijOut ? TypeOfField::Out : TypeOfField::In][atomId].data(),
          D_ij[isDijOut ? TypeOfField::Out : TypeOfField::In][atomId].data() +
            numberSphericalFunctions * numberSphericalFunctions,
          tempDij.data(),
          D_ij[isDijOut ? TypeOfField::Out : TypeOfField::In][atomId].data(),
          [](auto &p, auto &q) { return p + dftfe::utils::realPart(q); });
        pcout << "DEBUG: PAW DijIn size: " << D_ij[TypeOfField::In].size()
              << std::endl;
        pcout << "---------------MATRIX METHOD ------------------------"
              << std::endl;


        pcout << "------------------------------------------------------------"
              << std::endl;
        pcout << "D_ij of atom: " << atomId << " with Z:" << Znum << std::endl;
        int                 numberProjectorFunctions = numberSphericalFunctions;
        std::vector<ValueType> tempD_ij                 = tempDij;
        for (int i = 0; i < numberProjectorFunctions; i++)
          {
            for (int j = 0; j < numberProjectorFunctions; j++)
              pcout << tempD_ij[i * numberProjectorFunctions + j] << " ";
            pcout << std::endl;
          }
        pcout << "------------------------------------------------------------"
              << std::endl;
      }
  }
  template class pawClass<dataTypes::number, dftfe::utils::MemorySpace::HOST>;
#if defined(DFTFE_WITH_DEVICE)
  template class pawClass<dataTypes::number,
                           dftfe::utils::MemorySpace::DEVICE>;
#endif
} // namespace dftfe
