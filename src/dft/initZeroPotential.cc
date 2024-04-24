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
// @author Phani Motamarri
//

//
// Initialize rho by reading in single-atom electron-density and fit a spline
//
#include <dftParameters.h>
#include <dft.h>
#include <fileReaders.h>

namespace dftfe
{
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::initZeroPotential()
  {
    // clear existing data
    d_zeroPotential.clear();
    d_zeroPotentialAtoms.clear();

    // Reading single atom rho initial guess
    pcout << std::endl
          << "Computing ZeroPotential at QuadPoints....." << std::endl;

    std::map<unsigned int, double> outerMostPointZeroPotential;
    const double                   truncationTol = 1e-12;
    unsigned int                   fileReadFlag  = 0;

    double maxZeroPotentialTail = 0.0;
    // loop over atom types
    for (std::set<unsigned int>::iterator it = atomTypes.begin();
         it != atomTypes.end();
         it++)
      {
        outerMostPointZeroPotential[*it] =
          d_pawClassPtr->getRmaxZeroPotential(*it);
        if (outerMostPointZeroPotential[*it] > maxZeroPotentialTail)
          maxZeroPotentialTail = outerMostPointZeroPotential[*it];
        if (d_dftParamsPtr->verbosity >= 4)
          pcout << " Atomic number: " << *it
                << " Outermost Point ZeroPotential: "
                << outerMostPointZeroPotential[*it] << std::endl;
      }

    const double cellCenterCutOff = maxZeroPotentialTail + 5.0;
    //
    // Initialize rho
    //
    const dealii::Quadrature<3> &quadrature_formula =
      matrix_free_data.get_quadrature(d_lpspQuadratureId);
    dealii::FEValues<3> fe_values(FE,
                                  quadrature_formula,
                                  dealii::update_quadrature_points);
    const unsigned int  n_q_points = quadrature_formula.size();
    //
    // get number of global charges
    //
    const int numberGlobalCharges = atomLocations.size();

    //
    // get number of image charges used only for periodic
    //
    const int numberImageCharges = d_imageIdsTrunc.size();

    //
    // loop over elements
    //
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = dofHandler.begin_active(),
      endc = dofHandler.end();


    // loop over elements
    //
    cell = dofHandler.begin_active();
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);

            std::vector<double> &zeroPotentialQuadValues =
              d_zeroPotential[cell->id()];
            std::vector<double> zeroPotentialAtom(n_q_points, 0.0);
            zeroPotentialQuadValues.resize(n_q_points, 0.0);



            // loop over atoms
            for (unsigned int iAtom = 0; iAtom < atomLocations.size(); ++iAtom)
              {
                dealii::Point<3> atom(atomLocations[iAtom][2],
                                      atomLocations[iAtom][3],
                                      atomLocations[iAtom][4]);
                bool             isZeroPotentialDataInCell = false;


                if (atom.distance(cell->center()) > cellCenterCutOff)
                  continue;

                // loop over quad points
                for (unsigned int q = 0; q < n_q_points; ++q)
                  {
                    dealii::Point<3> quadPoint = fe_values.quadrature_point(q);
                    dealii::Tensor<1, 3, double> diff = quadPoint - atom;
                    double distanceToAtom = quadPoint.distance(atom);
                    double value          = 0.0;
                    if (distanceToAtom <=
                        outerMostPointZeroPotential[atomLocations[iAtom][0]])
                      {
                        value = d_pawClassPtr->getRadialZeroPotential(
                          atomLocations[iAtom][0], distanceToAtom);

                        // pcout<<distanceToAtom<<" "<<value<<std::endl;
                        isZeroPotentialDataInCell = true;
                      }
                    zeroPotentialQuadValues[q] += value;
                    zeroPotentialAtom[q] = value;


                  } // end loop over quad points
                if (isZeroPotentialDataInCell)
                  {
                    std::vector<double> &ZeroPotentialAtomCell =
                      d_zeroPotentialAtoms[iAtom][cell->id()];
                    ZeroPotentialAtomCell.resize(n_q_points, 0.0);

                    for (unsigned int q = 0; q < n_q_points; ++q)
                      {
                        ZeroPotentialAtomCell[q] = zeroPotentialAtom[q];
                      } // q_point loop
                  }     // if loop
              }         // loop over atoms

            // loop over image charges
            for (unsigned int iImageCharge = 0;
                 iImageCharge < numberImageCharges;
                 ++iImageCharge)
              {
                const int        masterAtomId = d_imageIdsTrunc[iImageCharge];
                dealii::Point<3> imageAtom(
                  d_imagePositionsTrunc[iImageCharge][0],
                  d_imagePositionsTrunc[iImageCharge][1],
                  d_imagePositionsTrunc[iImageCharge][2]);

                if (imageAtom.distance(cell->center()) > cellCenterCutOff)
                  continue;

                bool isZeroPotentialDataInCell = false;

                // loop over quad points
                for (unsigned int q = 0; q < n_q_points; ++q)
                  {
                    dealii::Point<3> quadPoint = fe_values.quadrature_point(q);
                    dealii::Tensor<1, 3, double> diff = quadPoint - imageAtom;
                    double distanceToAtom = quadPoint.distance(imageAtom);



                    double value = 0.0;
                    if (distanceToAtom <=
                        outerMostPointZeroPotential[atomLocations[masterAtomId]
                                                                 [0]])
                      {
                        value = d_pawClassPtr->getRadialZeroPotential(
                          atomLocations[masterAtomId][0], distanceToAtom);
                        // pcout<<distanceToAtom<<" "<<value<<std::endl;
                        isZeroPotentialDataInCell = true;
                      }

                    zeroPotentialQuadValues[q] += value;
                    zeroPotentialAtom[q] = value;

                  } // quad point loop

                if (isZeroPotentialDataInCell)
                  {
                    std::vector<double> &ZeroPotentialAtomCell =
                      d_zeroPotentialAtoms[numberGlobalCharges + iImageCharge]
                                          [cell->id()];
                    ZeroPotentialAtomCell.resize(n_q_points);

                    for (unsigned int q = 0; q < n_q_points; ++q)
                      {
                        ZeroPotentialAtomCell[q] = zeroPotentialAtom[q];
                      } // q_point loop
                  }     // if loop

              } // end of image charges

          } // cell locally owned check

      } // cell loop
  }
#include "dft.inst.cc"
} // namespace dftfe
