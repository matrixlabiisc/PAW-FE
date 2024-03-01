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
  pawClass<ValueType, memorySpace>::computeCompensationchargel0()
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

                  } // quad loop

              } // image atom loop

          } // iElemComp

      } // iAtom
  }

} // namespace dftfe
