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
  double
  pawClass<ValueType, memorySpace>::gaunt(int l_i,
                                          int l_j,
                                          int l,
                                          int m_i,
                                          int m_j,
                                          int m)
  {
    bool flagm = !(m == (m_i + m_j) || m == m_i - m_j || m == -m_i + m_j ||
                   m == -m_i - m_j);
    int  lmax  = l_i + l_j;
    int  k     = std::max(std::abs(l_i - l_j),
                     std::min(std::abs(m_i + m_j), std::abs(m_i - m_j)));
    int  lmin  = (k + lmax) % 2 == 0 ? k : k + 1;
    /*if (flagm || ((l_i + l_j + l) % 2 == 1) ||
        l < lmin || l > lmax)
      {
        return 0.0;
      } */


    int flag1 = m_i == 0 ? 0 : 1;
    int flag2 = m_j == 0 ? 0 : 1;
    int flag3 = m == 0 ? 0 : 1;
    int flag  = flag1 + flag2 + flag3;

    if (flag < 2)
      {
        double gauntvalue = wigner::gaunt<double>(l_i, l_j, l, m_i, m_j, m);
        // double gauntvalue = gauntcomplex(l_i, l_j, l, m_i, m_j, m);
        if (flag == 0)
          {
            return gauntvalue;
          }
        else
          return 0.0;
      }

    if (flag == 3)
      {
        std::complex<double> U1 =
          (computeTransformationExtries(l, m, -(m_i + m_j))) *
          computeTransformationExtries(l_i, m_i, m_i) *
          computeTransformationExtries(l_j, m_j, m_j);
        std::complex<double> U2 =
          (computeTransformationExtries(l, m, -(m_i - m_j))) *
          computeTransformationExtries(l_i, m_i, m_i) *
          computeTransformationExtries(l_j, m_j, -m_j);

        double value =
          2 * U1.real() *
            wigner::gaunt<double>(l_i, l_j, l, m_i, m_j, -(m_i + m_j)) +
          2 * U2.real() *
            wigner::gaunt<double>(l_i, l_j, l, m_i, -m_j, -(m_i - m_j));
        return (value);
      }
    if (flag == 2)
      {
        int l1, l2, l3, m1, m2;
        if (flag1 == 0)
          {
            l3 = l_i;
            l2 = l_j;
            l1 = l;
            m2 = m_j;
            m1 = m;
          }
        else if (flag2 == 0)
          {
            l3 = l_j;
            l2 = l_i;
            l1 = l;
            m2 = m_i;
            m1 = m;
          }
        else
          {
            l3 = l;
            l2 = l_i;
            l1 = l_j;
            m2 = m_i;
            m1 = m_j;
          }
        std::complex<double> U = (computeTransformationExtries(l1, m1, -m2)) *
                                 computeTransformationExtries(l2, m2, m2);
        double value =
          2 * U.real() * wigner::gaunt<double>(l3, l2, l1, 0, m2, -m2);
        return (value);
      }
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  std::complex<double>
  pawClass<ValueType, memorySpace>::computeTransformationExtries(int l,
                                                                 int mu,
                                                                 int m)
  {
    std::complex<double> U(0.0, 0.0);
    int                  delta_mu0deltam0 = (m == 0 && mu == 0) ? 1 : 0;
    U.real(delta_mu0deltam0 +
           1 / sqrt(2) *
             ((mu > 0 ? 1 : 0) * (m == mu ? 1 : 0) +
              (mu > 0 ? 1 : 0) * pow(-1, m) * (m == -mu ? 1 : 0)));
    U.imag(1 / sqrt(2) *
           ((-mu > 0 ? 1 : 0) * pow(-1, m) * (m == mu ? 1 : 0) -
            (-mu > 0 ? 1 : 0) * (m == -mu ? 1 : 0)));
    return (U);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::threeTermIntegrationOverAugmentationSphere(
    double *             f1,
    double *             f2,
    double *             f3,
    std::vector<double> &radial,
    std::vector<double> &rab,
    const unsigned int   rminIndex,
    const unsigned int   rmaxIndex)
  {
    double                                      IntOut = 0.0;
    std::function<double(const unsigned int &)> integrationValue =
      [&](const unsigned int &i) {
        double Value = rab[i] * f3[i] * f2[i] * f1[i] * radial[i] * radial[i];
        return (Value);
      };
    double Q1 = simpsonIntegral(rminIndex, rmaxIndex, integrationValue);
    IntOut += Q1;
    return (IntOut);
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::simpsonIntegral(
    unsigned int                                 startIndex,
    unsigned int                                 EndIndex,
    std::function<double(const unsigned int &)> &IntegrandValue)
  {
    if (startIndex == EndIndex)
      return 0.0;
    else if (startIndex + 1 == EndIndex)
      return 0.5 * (IntegrandValue(startIndex) + IntegrandValue(EndIndex));
    else
      {
        std::vector<double> simfact((EndIndex - startIndex), 0.0);
        simfact[EndIndex - 1] = 1.0 / 3.0;
        simfact[startIndex]   = 0.0;
        unsigned int ir_last  = 0;
        for (unsigned int i = EndIndex - 1; i >= startIndex + 2; i -= 2)
          {
            simfact[i - 1] = 4.0 / 3.0;
            simfact[i - 2] = 2.0 / 3.0;
            ir_last        = i - 2;
          }
        simfact[ir_last] *= 0.5;
        double IntegralResult = 0.0;
        for (unsigned int i = startIndex; i < EndIndex; i++)
          IntegralResult += simfact[i] * IntegrandValue(i);
        double residual = 0.0;
        if ((EndIndex - startIndex) % 2 != 0)
          return (IntegralResult);
        else
          {
            residual = 1.0 / 3.0 *
                       (IntegrandValue(startIndex) * 1.25 +
                        2.0 * IntegrandValue(startIndex + 1) -
                        0.25 * IntegrandValue(startIndex + 2));
            if (std::fabs(residual) > 1E-8)
              pcout << "DEBUG: Residual activated: " << residual << " "
                    << IntegralResult << std::endl;
            return (IntegralResult + residual);
          }
      }
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::oneTermPoissonPotential(
    const double *             fun,
    const unsigned int         l,
    const unsigned int         rminIndex,
    const unsigned int         rmaxIndex,
    const int                  powerofR,
    const std::vector<double> &radial,
    const std::vector<double> &rab,
    std::vector<double> &      Potential)
  {
    Potential.clear();
    Potential.resize(radial.size(), 0.0);
    using namespace boost::math::quadrature;
    const double        rmin = radial[rminIndex];
    const double        rmax = radial[rmaxIndex];
    std::vector<double> Integral1(radial.size(), 0.0);
    std::vector<double> Integral2(radial.size(), 0.0);
    std::vector<double> radL(radial.size(), 0.0);
    std::vector<double> radL1(radial.size(), 0.0);
    std::vector<double> aa(radial.size(), 0.0);
    std::vector<double> bb(radial.size(), 0.0);
    for (int i = 1; i < radial.size(); i++)
      {
        double r   = radial[i];
        double g_y = fun[i];
        radL[i]    = pow(radial[i], l);
        radL1[i]   = radL[i] * radial[i];
        aa[i]      = g_y * pow(r, powerofR) * radL[i] * rab[i];
        bb[i]      = g_y * pow(r, powerofR) / radL1[i] * rab[i];
      }

    for (int i = (int)rminIndex + 2; i <= (int)rmaxIndex; i += 2)
      {
        Integral1[i] = Integral1[i - 2] +
                       (1.0) / (3.0) * (aa[i - 2] + 4.0 * aa[i - 1] + aa[i]);
        Integral1[i - 1] =
          Integral1[i - 2] +
          (1.0) / (3.0) * (1.25 * aa[i - 2] + 2.0 * aa[i - 1] - 0.25 * aa[i]);
      }
    for (int i = (int)rmaxIndex - 2; i >= (int)rminIndex; i -= 2)
      {
        Integral2[i] = Integral2[i + 2] +
                       (1.0) / (3.0) * (bb[i + 2] + 4.0 * bb[i + 1] + bb[i]);
        Integral2[i + 1] =
          Integral2[i + 2] +
          (1.0) / (3.0) * (1.25 * bb[i + 2] + 2.0 * bb[i + 1] - 0.25 * bb[i]);
      }
    if ((rmaxIndex - rminIndex) % 2 != 0)
      {
        Integral1[rmaxIndex] =
          Integral1[rmaxIndex - 1] +
          1.0 / 3.0 *
            (1.25 * aa[rmaxIndex - 2] + 2.0 * aa[rmaxIndex - 1] -
             0.25 * aa[rmaxIndex]);
        Integral2[rminIndex] =
          Integral2[rminIndex + 1] +
          1.0 / 3.0 *
            (1.25 * bb[rminIndex + 2] + 2.0 * bb[rminIndex + 1] -
             0.25 * bb[rminIndex]);
      }



    for (int i = (int)rminIndex; i < (int)radial.size(); i++)
      {
        double r     = radial[i];
        double Value = 0.0;
        if (r > 1E-10)
          {
            double Value1 = Integral1[i];
            double Value2 = Integral2[i];
            Value         = (4 * M_PI / (2 * double(l) + 1.0)) *
                    (Value1 / pow(r, l) + Value2 * pow(r, l + 1));
          }

        Potential[i] = (Value);
        if (i > (int)rmaxIndex)
          Potential[i] = Potential[(int)rmaxIndex];
      }
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::twoTermPoissonPotential(
    const double *             fun1,
    const double *             fun2,
    const unsigned int         l,
    const unsigned int         rminIndex,
    const unsigned int         rmaxIndex,
    const int                  powerofR,
    const std::vector<double> &radial,
    const std::vector<double> &rab,
    std::vector<double> &      Potential)
  {
    Potential.clear();
    Potential.resize(radial.size(), 0.0);
    using namespace boost::math::quadrature;
    const double        rmin = radial[rminIndex];
    const double        rmax = radial[rmaxIndex];
    std::vector<double> Integral1(radial.size(), 0.0);
    std::vector<double> Integral2(radial.size(), 0.0);
    std::vector<double> radL(radial.size(), 0.0);
    std::vector<double> radL1(radial.size(), 0.0);
    std::vector<double> aa(radial.size(), 0.0);
    std::vector<double> bb(radial.size(), 0.0);
    for (int i = 1; i < radial.size(); i++)
      {
        double r = radial[i];
        double g_y;
        radL[i]  = pow(radial[i], l);
        radL1[i] = radL[i] * radial[i];
        g_y      = fun1[i] * fun2[i] * pow(r, powerofR);
        aa[i]    = g_y * radL[i] * rab[i];
        bb[i]    = g_y / radL1[i] * rab[i];
      }


    for (int i = (int)rminIndex + 2; i <= (int)rmaxIndex; i += 2)
      {
        Integral1[i] = Integral1[i - 2] +
                       (1.0) / (3.0) * (aa[i - 2] + 4.0 * aa[i - 1] + aa[i]);
        Integral1[i - 1] =
          Integral1[i - 2] +
          (1.0) / (3.0) * (1.25 * aa[i - 2] + 2.0 * aa[i - 1] - 0.25 * aa[i]);
      }
    for (int i = (int)rmaxIndex - 2; i >= (int)rminIndex; i -= 2)
      {
        Integral2[i] = Integral2[i + 2] +
                       (1.0) / (3.0) * (bb[i + 2] + 4.0 * bb[i + 1] + bb[i]);
        Integral2[i + 1] =
          Integral2[i + 2] +
          (1.0) / (3.0) * (1.25 * bb[i + 2] + 2.0 * bb[i + 1] - 0.25 * bb[i]);
      }
    if ((rmaxIndex - rminIndex) % 2 != 0)
      {
        Integral1[rmaxIndex] =
          Integral1[rmaxIndex - 1] +
          1.0 / 3.0 *
            (1.25 * aa[rmaxIndex - 2] + 2.0 * aa[rmaxIndex - 1] -
             0.25 * aa[rmaxIndex]);
        Integral2[rminIndex] =
          Integral2[rminIndex + 1] +
          1.0 / 3.0 *
            (1.25 * bb[rminIndex + 2] + 2.0 * bb[rminIndex + 1] -
             0.25 * bb[rminIndex]);
      }


    for (int i = (int)rminIndex; i < (int)radial.size(); i++)
      {
        double r     = radial[i];
        double Value = 0.0;
        if (r > 1E-10)
          {
            double Value1 = Integral1[i];
            double Value2 = Integral2[i];
            Value         = (4 * M_PI / (2 * double(l) + 1.0)) *
                    (Value1 / pow(r, l) + Value2 * pow(r, l + 1));
          }

        Potential[i] = (Value);
        if (i > (int)rmaxIndex)
          Potential[i] = Potential[(int)rmaxIndex];
      }
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::integralOfProjectorsInAugmentationSphere(
    const double *       f1,
    const double *       f2,
    std::vector<double> &radial,
    std::vector<double> &rab,
    const unsigned int   rminIndex,
    const unsigned int   rmaxIndex)
  {
    double value = 0.0;

    std::function<double(const unsigned int &)> integral =
      [&](const unsigned int &i) {
        if (radial[i] < 1E-9)
          return 0.0;

        double Value = rab[i] * f2[i] * f1[i] * radial[i];
        return (Value);
      };
    value = simpsonIntegral(rminIndex, rmaxIndex, integral);
    return (value);
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::integralOfDensity(
    const double *       f1,
    std::vector<double> &radial,
    std::vector<double> &rab,
    const unsigned int   rminIndex,
    const unsigned int   rmaxIndex)
  {
    double value = 0.0;

    std::function<double(const unsigned int &)> integral =
      [&](const unsigned int &i) {
        double Value = rab[i] * f1[i] * radial[i];
        return (Value);
      };

    value = simpsonIntegral(rminIndex, rmaxIndex, integral);
    return (value);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::getSphericalQuadratureRule(
    std::vector<double> &             quad_weights,
    std::vector<std::vector<double>> &quad_points)
  {
    std::vector<std::vector<double>> quadratureData;
    char                             quadratureFileName[256];
    if (d_dftParamsPtr->sphericalQuadrature == 0)
      sprintf(quadratureFileName,
              "%s/data/quadratureRules/quadRule86.txt",
              DFTFE_PATH);
    else if (d_dftParamsPtr->sphericalQuadrature == 1)
      sprintf(quadratureFileName,
              "%s/data/quadratureRules/quadRule50.txt",
              DFTFE_PATH);
    else if (d_dftParamsPtr->sphericalQuadrature == 2)
      sprintf(quadratureFileName,
              "%s/data/quadratureRules/quadRule74.txt",
              DFTFE_PATH);
    else if (d_dftParamsPtr->sphericalQuadrature == 3)
      sprintf(quadratureFileName,
              "%s/data/quadratureRules/quadRule110.txt",
              DFTFE_PATH);
    dftUtils::readFile(3, quadratureData, quadratureFileName);
    int numRows = quadratureData.size();
    for (int i = 0; i < numRows; i++)
      {
        quad_weights.push_back(quadratureData[i][2]);
        std::vector<double> temp(2, 0);
        temp[1] = (quadratureData[i][0] + 180) / 180 * M_PI;
        temp[0] = quadratureData[i][1] / 180 * M_PI;
        quad_points.push_back(temp);
      }
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  std::vector<double>
  pawClass<ValueType, memorySpace>::derivativeOfRealSphericalHarmonic(
    unsigned int lQuantumNo,
    int          mQuantumNo,
    double       theta,
    double       phi)
  {
    std::vector<double> RSH(2, 0.0);
    if (lQuantumNo == 0)
      return (RSH);
    double sphericalHarmonicValue, sphericalHarmonicValue1;
    sphericalHarmonicUtils::getSphericalHarmonicVal(
      theta, phi, lQuantumNo, -mQuantumNo, sphericalHarmonicValue);
    // RSH[1] = -std::abs(m) * sphericalHarmonicValue;
    RSH[1] = -mQuantumNo * sphericalHarmonicValue;

    sphericalHarmonicUtils::getSphericalHarmonicVal(
      theta, phi, lQuantumNo, mQuantumNo, sphericalHarmonicValue);
    sphericalHarmonicUtils::getSphericalHarmonicVal(
      theta, phi, lQuantumNo + 1, mQuantumNo, sphericalHarmonicValue1);
    if (std::fabs(std::sin(theta)) > 1E-8)
      RSH[0] = -double(lQuantumNo + 1) * std::cos(theta) / std::sin(theta) *
                 sphericalHarmonicValue +
               sqrt(double(2 * lQuantumNo + 1.0)) /
                 sqrt(double(2 * lQuantumNo + 3.0)) *
                 sqrt(double((lQuantumNo + 1) * (lQuantumNo + 1) -
                             mQuantumNo * mQuantumNo)) *
                 sphericalHarmonicValue1 / std::sin(theta);
    else
      RSH[0] = 0.0;


    return (RSH);
  }



  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  std::vector<double>
  pawClass<ValueType, memorySpace>::DijVectorForMixing(TypeOfField typeOfField)
  {
    std::vector<unsigned int> atomicNumber =
      d_atomicProjectorFnsContainer->getAtomicNumbers();
    const std::vector<unsigned int> ownedAtomIds =
      d_nonLocalOperator->getOwnedAtomIdsInCurrentProcessor();
    std::vector<double> DijVector;
    for (unsigned int iAtom = 0; iAtom < ownedAtomIds.size(); iAtom++)
      {
        unsigned int atomId = ownedAtomIds[iAtom];
        unsigned int Znum   = atomicNumber[atomId];
        unsigned int numProj =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        const std::map<std::pair<unsigned int, unsigned int>,
                       std::shared_ptr<AtomCenteredSphericalFunctionBase>>
          sphericalFunction =
            d_atomicProjectorFnsContainer->getSphericalFunctions();
        unsigned int numRadProj =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);

        std::vector<double> Dij_in  = D_ij[TypeOfField::In][atomId];
        std::vector<double> Dij_out = D_ij[TypeOfField::Out][atomId];
        if (typeOfField == TypeOfField::In)
          {
            for (int iProj = 0; iProj < numProj; iProj++)
              {
                for (int jProj = 0; jProj < numProj; jProj++)
                  {
                    DijVector.push_back(Dij_in[iProj * numProj + jProj]);
                  }
              }
          }
        else if (typeOfField == TypeOfField::Out)
          {
            for (int iProj = 0; iProj < numProj; iProj++)
              {
                for (int jProj = 0; jProj < numProj; jProj++)
                  {
                    DijVector.push_back(Dij_out[iProj * numProj + jProj]);
                  }
              }
          }
        else
          {
            for (int iProj = 0; iProj < numProj; iProj++)
              {
                for (int jProj = 0; jProj < numProj; jProj++)
                  {
                    DijVector.push_back(Dij_out[iProj * numProj + jProj] -
                                        Dij_in[iProj * numProj + jProj]);
                  }
              }
          }
      }
    // std::cout << "Size of Dij vector in processor: " << d_this_mpi_process
    //           << " " << DijVector.size() << std::endl;
    MPI_Barrier(d_mpiCommParent);
    return (DijVector);
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  std::vector<double>
  pawClass<ValueType, memorySpace>::radialDerivativeOfMeshData(
    const std::vector<double> &r,
    const std::vector<double> &rab,
    const std::vector<double> &functionValue)
  {
    alglib::real_1d_array       x, y;
    alglib::spline1dinterpolant p1;
    unsigned int                size = r.size();
    x.setcontent(size, &r[0]);
    y.setcontent(size, &functionValue[0]);
    alglib::ae_int_t natural_bound_type = 1;
    alglib::ae_int_t dir_bound_type     = 0;
    alglib::spline1dbuildcubic(x,
                               y,
                               size,
                               dir_bound_type,
                               functionValue[0],
                               natural_bound_type,
                               0.0,
                               p1);
    std::vector<double> der(size, 0.0);
    std::vector<double> coeff(5, 0.0);
    coeff[0] = -25.0 / 12.0;
    coeff[1] = 4.0;
    coeff[2] = -3.0;
    coeff[3] = 4.0 / 3.0;
    coeff[4] = -1.0 / 4.0;
    MPI_Barrier(d_mpiCommParent);

    for (unsigned int i = 0; i < size - 4; i++)
      {
        double Value, derivativeValue, radialDensitySecondDerivative;
        der[i] =
          (coeff[0] * functionValue[i] + coeff[1] * functionValue[i + 1] +
           coeff[2] * functionValue[i + 2] + coeff[3] * functionValue[i + 3] +
           coeff[4] * functionValue[i + 4]) /
          rab[i];
        alglib::spline1ddiff(
          p1, r[i], Value, derivativeValue, radialDensitySecondDerivative);
      }

    return (der);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::multipoleIntegrationGrid(
    double *             f1,
    double *             f2,
    std::vector<double> &radial,
    std::vector<double> &rab,
    const int            L,
    const unsigned int   rminIndex,
    const unsigned int   rmaxIndex)
  {
    std::function<double(const unsigned int &)> integrationValue =
      [&](const unsigned int &i) {
        double Value = rab[i] * f2[i] * f1[i];
        Value *= pow(radial[i], L + 2);
        return (Value);
      };

    double IntegralResult =
      simpsonIntegral(rminIndex, rmaxIndex, integrationValue);


    return (IntegralResult);
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::densityScalingFactor(
    const std::vector<std::vector<double>> &atomLocations)
  {
    double scaleFactor  = 0.0;
    double numElectrons = 0;

    for (int atomId = 0; atomId < atomLocations.size(); atomId++)
      {
        unsigned int Znum = atomLocations[atomId][0];
        numElectrons += atomLocations[atomId][1];
        const unsigned int numberOfProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        const unsigned int numberOfRadialProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);
        std::vector<double> Dij              = D_ij[TypeOfField::In][atomId];
        std::vector<double> multipoleTable   = d_multipole[Znum];
        unsigned int        projectorIndex_i = 0;
        for (int alpha_i = 0; alpha_i < numberOfRadialProjectors; alpha_i++)
          {
            std::shared_ptr<AtomCenteredSphericalFunctionBase> AEsphFn_i =
              d_atomicAEPartialWaveFnsMap.find(std::make_pair(Znum, alpha_i))
                ->second;
            int lQuantumNo_i = AEsphFn_i->getQuantumNumberl();
            for (int mQuantumNo_i = -lQuantumNo_i; mQuantumNo_i <= lQuantumNo_i;
                 mQuantumNo_i++)
              {
                unsigned int projectorIndex_j = 0;
                for (int alpha_j = 0; alpha_j < numberOfRadialProjectors;
                     alpha_j++)
                  {
                    std::shared_ptr<AtomCenteredSphericalFunctionBase>
                      AEsphFn_j = d_atomicAEPartialWaveFnsMap
                                    .find(std::make_pair(Znum, alpha_j))
                                    ->second;
                    int lQuantumNo_j = AEsphFn_j->getQuantumNumberl();
                    for (int mQuantumNo_j = -lQuantumNo_j;
                         mQuantumNo_j <= lQuantumNo_j;
                         mQuantumNo_j++)
                      {
                        scaleFactor +=
                          gaunt(lQuantumNo_i,
                                lQuantumNo_j,
                                0,
                                mQuantumNo_i,
                                mQuantumNo_j,
                                0) *
                          Dij[projectorIndex_i * numberOfProjectors +
                              projectorIndex_j] *
                          multipoleTable[alpha_i * numberOfRadialProjectors +
                                         alpha_j];
                        // pcout
                        //   << projectorIndex_i << " " << projectorIndex_j << "
                        //   "
                        //   << alpha_i << " " << alpha_j << " "
                        //   << gaunt(lQuantumNo_i,
                        //            lQuantumNo_j,
                        //            0,
                        //            mQuantumNo_i,
                        //            mQuantumNo_j,
                        //            0)
                        //   << " "
                        //   << Dij[projectorIndex_i * numberOfProjectors +
                        //          projectorIndex_j]
                        //   << " "
                        //   << multipoleTable[alpha_i *
                        //   numberOfRadialProjectors +
                        //                     alpha_j]
                        //   << std::endl;
                        projectorIndex_j++;
                      }
                  }

                projectorIndex_i++;
              }
          }
      }
    pcout << "Number of Electrons: " << numElectrons << std::endl;
    pcout << "sqrt(4*M_PI)*DeltaijDij: " << sqrt(4 * M_PI) * scaleFactor
          << std::endl;
    pcout << "Scaling Factor for Init Rho: "
          << numElectrons - sqrt(4 * M_PI) * scaleFactor << std::endl;
    return (numElectrons - sqrt(4 * M_PI) * scaleFactor);
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::communicateDijAcrossAllProcessors(
    TypeOfField     typeOfField,
    const MPI_Comm &interpoolcomm,
    const MPI_Comm &interBandGroupComm)
  {
    const std::vector<unsigned int> ownedAtomIds =
      d_nonLocalOperator->getOwnedAtomIdsInCurrentProcessor();
    std::vector<double>       DijTotalVector(d_nProjSqTotal, 0.0);
    std::vector<unsigned int> atomicNumber =
      d_atomicProjectorFnsContainer->getAtomicNumbers();
    if (ownedAtomIds.size() > 0)
      {
        for (int iAtom = 0; iAtom < ownedAtomIds.size(); iAtom++)
          {
            unsigned int        atomId     = ownedAtomIds[iAtom];
            unsigned int        Znum       = atomicNumber[atomId];
            unsigned int        startIndex = d_projectorStartIndex[atomId];
            std::vector<double> Dij        = D_ij[typeOfField][atomId];
            unsigned int        numberOfProjectors =
              d_atomicProjectorFnsContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
            unsigned int index = 0;
            for (unsigned int i = 0; i < numberOfProjectors; i++)
              {
                for (unsigned int j = 0; j <= i; j++)
                  {
                    DijTotalVector[(startIndex + index)] =
                      Dij[i * numberOfProjectors + j];
                    index++;
                  }
              }
          }
      }
    MPI_Allreduce(MPI_IN_PLACE,
                  &DijTotalVector[0],
                  d_nProjSqTotal,
                  MPI_DOUBLE,
                  MPI_SUM,
                  d_mpiCommParent);
    for (unsigned int atomId = 0; atomId < atomicNumber.size(); atomId++)
      {
        unsigned int Znum = atomicNumber[atomId];
        unsigned int numberOfProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        unsigned int        startIndex = d_projectorStartIndex[atomId];
        std::vector<double> Dij(numberOfProjectors * numberOfProjectors, 0.0);
        unsigned int        index = 0;
        for (int i = 0; i < numberOfProjectors; i++)
          {
            for (int j = 0; j <= i; j++)
              {
                Dij[i * numberOfProjectors + j] =
                  DijTotalVector[(startIndex + index)];
                Dij[j * numberOfProjectors + i] =
                  DijTotalVector[(startIndex + index)];
                index++;
              }
          }
        if (d_dftParamsPtr->verbosity >= 5)
          {
            pcout << "---------------MATRIX METHOD ------------------------"
                  << std::endl;


            pcout
              << "------------------------------------------------------------"
              << std::endl;
            pcout << "D_ij of atom: " << atomId << " with Z:" << Znum
                  << std::endl;
            int numberProjectorFunctions = numberOfProjectors;
            for (int i = 0; i < numberProjectorFunctions; i++)
              {
                for (int j = 0; j < numberProjectorFunctions; j++)
                  pcout << Dij[i * numberProjectorFunctions + j] << " ";
                pcout << std::endl;
              }
            pcout
              << "------------------------------------------------------------"
              << std::endl;
          }


        D_ij[typeOfField][atomId] = Dij;
      }

    d_HamiltonianCouplingMatrixEntriesUpdated = false;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeMultipoleInverse()
  {
    pcout << "PAWClass Init: computing Inverse Multipole Table" << std::endl;
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
        const unsigned int numberOfProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(atomicNumber);
        std::vector<double> fullMultipoleTable(numberOfProjectors *
                                                 numberOfProjectors,
                                               0.0);
        std::vector<double> multipoleTable   = d_multipole[*it];
        int                 projectorIndex_i = 0;
        for (int alpha_i = 0; alpha_i < numberOfRadialProjectors; alpha_i++)
          {
            std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn_i =
              sphericalFunction.find(std::make_pair(atomicNumber, alpha_i))
                ->second;
            int lQuantumNumber_i = sphFn_i->getQuantumNumberl();
            for (int mQuantumNumber_i = -lQuantumNumber_i;
                 mQuantumNumber_i <= lQuantumNumber_i;
                 mQuantumNumber_i++)
              {
                int projectorIndex_j = 0;
                for (int alpha_j = 0; alpha_j < numberOfRadialProjectors;
                     alpha_j++)
                  {
                    std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn_j =
                      sphericalFunction
                        .find(std::make_pair(atomicNumber, alpha_j))
                        ->second;
                    int lQuantumNumber_j = sphFn_j->getQuantumNumberl();
                    for (int mQuantumNumber_j = -lQuantumNumber_j;
                         mQuantumNumber_j <= lQuantumNumber_j;
                         mQuantumNumber_j++)
                      {
                        fullMultipoleTable[projectorIndex_i *
                                             numberOfProjectors +
                                           projectorIndex_j] =
                          sqrt(4 * M_PI) *
                          gaunt(lQuantumNumber_i,
                                lQuantumNumber_j,
                                0,
                                mQuantumNumber_i,
                                mQuantumNumber_j,
                                0) *
                          multipoleTable[alpha_i * numberOfRadialProjectors +
                                         alpha_j];
                        // pcout
                        //   << alpha_i << " " << alpha_j << " "
                        //   << projectorIndex_i << " " << projectorIndex_j << "
                        //   "
                        //   << gaunt(lQuantumNumber_i,
                        //            lQuantumNumber_j,
                        //            0,
                        //            mQuantumNumber_i,
                        //            mQuantumNumber_j,
                        //            0)
                        //   << " "
                        //   << multipoleTable[alpha_i *
                        //   numberOfRadialProjectors +
                        //                     alpha_j]
                        //   << " "
                        //   << fullMultipoleTable[projectorIndex_i *
                        //                           numberOfProjectors +
                        //                         projectorIndex_j]
                        //   << std::endl;
                        projectorIndex_j++;
                      } // mQuantumNumber_j
                  }     // alpha_j
                projectorIndex_i++;
              } // mQuantumNumber_i
          }     // alpha_i
        const char          uplo = 'L';
        const int           N    = numberOfProjectors;
        std::vector<double> A    = fullMultipoleTable;
        pcout << "Multipole Table: " << std::endl;
        for (int i = 0; i < numberOfProjectors; i++)
          {
            for (int j = 0; j < numberOfProjectors; j++)
              pcout << A[i * numberOfProjectors + j] << " ";
            pcout << std::endl;
          }

        dftfe::linearAlgebraOperations::inverse(&A[0], N);
        d_multipoleInverse[atomicNumber] = A;

        pcout << "Multipole Table Inverse: " << std::endl;
        for (int i = 0; i < numberOfProjectors; i++)
          {
            for (int j = 0; j < numberOfProjectors; j++)
              pcout << A[i * numberOfProjectors + j] << " ";
            pcout << std::endl;
          }


      } //*it
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::TotalCompensationCharge()
  {
    double             normValue = 0.0;
    const unsigned int numberAtomsOfInterest =
      d_atomicShapeFnsContainer->getNumAtomCentersSize();
    const unsigned int numberQuadraturePoints =
      (d_jxwcompensationCharge.begin()->second).size();
    const std::vector<unsigned int> &atomicNumber =
      d_atomicShapeFnsContainer->getAtomicNumbers();
    unsigned int iElem = 0;

    for (std::map<dealii::CellId, std::vector<double>>::iterator it =
           (*d_bQuadValuesAllAtoms).begin();
         it != (*d_bQuadValuesAllAtoms).end();
         ++it)
      {
        const dealii::CellId      cellId = it->first;
        const std::vector<double> Temp =
          (*d_bQuadValuesAllAtoms).find(it->first)->second;
        const unsigned int elementIndex =
          d_BasisOperatorElectroHostPtr->cellIndex(cellId);
        const std::vector<double> jxw =
          d_jxwcompensationCharge.find(it->first)->second;
        for (unsigned int q_point = 0; q_point < numberQuadraturePoints;
             q_point++)
          {
            normValue += Temp[q_point] * jxw[q_point];
            // std::cout << "cell Index and Quad: " << elementIndex << " "
            //           << q_point << " " << jxw[q_point] << std::endl;
          }
        iElem++;
      }


    d_TotalCompensationCharge =
      dealii::Utilities::MPI::sum(normValue, d_mpiCommParent);
    pcout << "Total Compensation Charge: " << d_TotalCompensationCharge
          << std::endl;
    return d_TotalCompensationCharge;
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::chargeNeutrality(double integralRhoValue,
                                                     TypeOfField typeOfField,
                                                     bool computeCompCharge)
  {
    if (computeCompCharge)
      computeCompensationCharge(typeOfField);
    double integralCompCharge = TotalCompensationCharge();
    pcout << "----------------------------------------------------"
          << std::endl;
    pcout << "Integral nTilde : " << integralRhoValue << std::endl;
    pcout << "Integral nTilde + nTildeCore: "
          << d_integralCoreDensity + integralRhoValue << std::endl;
    pcout << "Inegral Comp charge: " << integralCompCharge << std::endl;
    pcout << "Charge Neutrality error: "
          << (integralRhoValue + d_integralCoreDensity + integralCompCharge)
          << std::endl;
    pcout << "----------------------------------------------------"
          << std::endl;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::fillDijMatrix(
    TypeOfField                typeOfField,
    const std::vector<double> &DijVector,
    const MPI_Comm &           interpoolcomm,
    const MPI_Comm &           interBandGroupComm)
  {
    std::vector<unsigned int> atomicNumber =
      d_atomicProjectorFnsContainer->getAtomicNumbers();
    for (int iAtom = 0; iAtom < atomicNumber.size(); iAtom++)
      {
        unsigned int Znum   = atomicNumber[iAtom];
        unsigned int atomId = iAtom;
        unsigned int numberOfProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        D_ij[typeOfField][atomId] =
          std::vector<double>(numberOfProjectors * numberOfProjectors, 0.0);
      }

    // std::vector<double> DijTotalVector(d_nProjSqTotal, 0.0);
    const std::vector<unsigned int> ownedAtomIds =
      d_nonLocalOperator->getOwnedAtomIdsInCurrentProcessor();
    if (ownedAtomIds.size() > 0)
      {
        unsigned int index = 0;
        for (int iAtom = 0; iAtom < ownedAtomIds.size(); iAtom++)
          {
            unsigned int atomId = ownedAtomIds[iAtom];
            unsigned int Znum   = atomicNumber[atomId];
            unsigned int numberOfProjectors =
              d_atomicProjectorFnsContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
            std::vector<double> Dij(numberOfProjectors * numberOfProjectors,
                                    0.0);
            for (unsigned int i = 0; i < numberOfProjectors; i++)
              {
                for (unsigned int j = 0; j < numberOfProjectors; j++)
                  {
                    Dij[i * numberOfProjectors + j] = DijVector[index];
                    index++;
                  }
              }
            D_ij[typeOfField][atomId] = Dij;
          }
      }
    communicateDijAcrossAllProcessors(typeOfField,
                                      interpoolcomm,
                                      interBandGroupComm);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  std::vector<double>
  pawClass<ValueType, memorySpace>::getDijWeights()
  {
    std::vector<double>             weights;
    const std::vector<unsigned int> ownedAtomIds =
      d_nonLocalOperator->getOwnedAtomIdsInCurrentProcessor();
    for (unsigned int iAtom = 0; iAtom < ownedAtomIds.size(); iAtom++)
      {
        unsigned int              atomId = ownedAtomIds[iAtom];
        std::vector<unsigned int> atomicNumber =
          d_atomicProjectorFnsContainer->getAtomicNumbers();
        unsigned int Znum = atomicNumber[atomId];
        unsigned int numProj =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        const std::map<std::pair<unsigned int, unsigned int>,
                       std::shared_ptr<AtomCenteredSphericalFunctionBase>>
          sphericalFunction =
            d_atomicProjectorFnsContainer->getSphericalFunctions();
        unsigned int numRadProj =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);
        std::vector<double> multipoleTable = d_multipole[Znum];

        std::vector<double> multipoleTableVal(numProj * numProj, 0.0);
        int                 projectorIndex_i = 0;
        for (int alpha_i = 0; alpha_i < numRadProj; alpha_i++)
          {
            std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn_i =
              sphericalFunction.find(std::make_pair(Znum, alpha_i))->second;
            int lQuantumNo_i = sphFn_i->getQuantumNumberl();
            for (int mQuantumNo_i = -lQuantumNo_i; mQuantumNo_i <= lQuantumNo_i;
                 mQuantumNo_i++)
              {
                int projectorIndex_j = 0;
                for (int alpha_j = 0; alpha_j < numRadProj; alpha_j++)
                  {
                    std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn_j =
                      sphericalFunction.find(std::make_pair(Znum, alpha_j))
                        ->second;
                    int lQuantumNo_j = sphFn_j->getQuantumNumberl();
                    for (int mQuantumNo_j = -lQuantumNo_j;
                         mQuantumNo_j <= lQuantumNo_j;
                         mQuantumNo_j++)
                      {
                        weights.push_back(
                          multipoleTable[alpha_i * numRadProj + alpha_j] *
                          gaunt(lQuantumNo_i,
                                lQuantumNo_j,
                                0,
                                mQuantumNo_i,
                                mQuantumNo_j,
                                0) *
                          sqrt(4 * M_PI));
                        projectorIndex_j++;
                      }
                  }
                projectorIndex_i++;
              }
          }
      }
    // std::cout << "Size of Dij Weights in procs: " << d_this_mpi_process << "
    // "
    //           << weights.size() << std::endl;
    MPI_Barrier(d_mpiCommParent);
    return weights;
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeIntegralCoreDensity(
    const std::map<dealii::CellId, std::vector<double>> &rhoCore)
  {
    d_BasisOperatorHostPtr->reinit(0, 0, d_densityQuadratureId);

    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                 JxwVector     = d_BasisOperatorHostPtr->JxWBasisData();
    unsigned int numQuadPoints = d_BasisOperatorHostPtr->nQuadsPerCell();
    MPI_Barrier(d_mpiCommParent);
    double totalCoreDensity = 0.0;
    if (rhoCore.size() > 0)
      {
        for (std::map<dealii::CellId, std::vector<double>>::const_iterator it =
               rhoCore.begin();
             it != rhoCore.end();
             ++it)
          {
            const std::vector<double> &Value = it->second;
            unsigned int               cellIndex =
              d_BasisOperatorHostPtr->cellIndex(it->first);
            for (int qpoint = 0; qpoint < numQuadPoints; qpoint++)
              totalCoreDensity +=
                Value[qpoint] * JxwVector[cellIndex * numQuadPoints + qpoint];
          }
      }
    d_integralCoreDensity =
      dealii::Utilities::MPI::sum(totalCoreDensity, d_mpiCommParent);
    d_integrealCoreDensityRadial = 0.0;
    std::vector<unsigned int> atomicNumbers =
      d_atomicProjectorFnsContainer->getAtomicNumbers();
    for (int iAtom = 0; iAtom < atomicNumbers.size(); iAtom++)
      {
        unsigned int Znum = atomicNumbers[iAtom];
        d_integrealCoreDensityRadial += d_integralCoreDensityPerAtom[Znum];
      }
    pcout
      << "PAW Class: Error in integralCoreDensity with radial data and FEM: "
      << std::fabs(d_integralCoreDensity - d_integrealCoreDensityRadial)
      << std::endl;
    if (std::fabs(d_integralCoreDensity - d_integrealCoreDensityRadial) > 1E-4)
      pcout << "PAW Class: Warning!! Increase density quadrature rule: "
            << std::endl;
  }
  template class pawClass<dataTypes::number, dftfe::utils::MemorySpace::HOST>;
#if defined(DFTFE_WITH_DEVICE)
  template class pawClass<dataTypes::number, dftfe::utils::MemorySpace::DEVICE>;
#endif
} // namespace dftfe
