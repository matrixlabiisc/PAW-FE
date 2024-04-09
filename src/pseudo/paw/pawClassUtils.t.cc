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

    // pcout<<"Entering Gaunt Values"<<std::endl;
    if (flag < 2)
      {
        double gauntvalue = wigner::gaunt<double>(l_i, l_j, l, m_i, m_j, m);
        // double gauntvalue = gauntcomplex(l_i, l_j, l, m_i, m_j, m);
        if (flag == 0)
          {
            /*if(std::fabs(gauntvalue) < 1E-7)
            pcout<<"Gaunt0: "<<l_i<<" "<<l_j<<" "<<l<<" "<<m_i<<" "<<m_j<<"
            "<<m<<std::endl; */
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
    // pcout << "Integral Error is: " << error << std::endl;
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
        // if(ir_last != startIndex)
        //   pcout<<"DEBUG: Warning there is an issue here: "<<std::endl;
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
    // pcout << "DEBUG: Rmin and Rmax" << rminIndex << " " << rmaxIndex << " "
    //       << rmax << std::endl;
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
        // pcout << "DEBUG Warning: Residual Term added in potential" <<
        // std::endl;
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
        double r = radial[i];
        // pcout << "DEBUG:" << i << " " << r << " " << Integral1[i] << " "
        //       << Integral2[i] << std::endl;
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
        g_y      = fun1[i] * fun2[i];
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
        // pcout << "DEBUG Warning: Residual Term added in potential" <<
        // std::endl;
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
        double r = radial[i];
        // pcout<<i<<" "<<r<<" "<<Integral1[i]<<" "<<Integral2[i]<<std::endl;
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
  double
  pawClass<ValueType, memorySpace>::computeDijResidualNorm()
  {
    std::vector<double> DijResidual = DijVectorForMixing(TypeOfField::Residual);
    double              normsq      = 0.0;
    for (int index = 0; index < DijResidual.size(); index++)
      normsq = DijResidual[index] * DijResidual[index];
    MPI_Allreduce(
      MPI_IN_PLACE, &normsq, 1, MPI_DOUBLE, MPI_SUM, d_mpiCommParent);
    return (std::sqrt(normsq));
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  std::vector<double>
  pawClass<ValueType, memorySpace>::DijVectorForMixing(TypeOfField typeOfField)
  {
    std::vector<double> DijVector(d_nProjPerTask * d_nProjPerTask, 0.0);
    unsigned int        index = 0;
    for (unsigned int i = 0; i < d_LocallyOwnedAtomId.size(); i++)
      {
        unsigned int              atomId = d_LocallyOwnedAtomId[i];
        std::vector<unsigned int> atomicNumber =
          d_atomicProjectorFnsContainer->getAtomicNumbers();
        unsigned int Znum = atomicNumber[atomId];
        unsigned int numProj =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        std::vector<double> multipoleTable = d_multipole[Znum];
        std::vector<double> Dij_in         = D_ij[TypeOfField::In][atomId];
        std::vector<double> Dij_out        = D_ij[TypeOfField::Out][atomId];
        if (typeOfField == TypeOfField::In)
          {
            for (int iProj = 0; iProj < numProj; iProj++)
              {
                for (int jProj = 0; jProj < numProj; jProj++)
                  {
                    DijVector[index] = std::sqrt(4 * M_PI) *
                                       multipoleTable[iProj * numProj + jProj] *
                                       Dij_in[iProj * numProj + jProj];
                    index++;
                  }
              }
          }
        else if (typeOfField == TypeOfField::Out)
          {
            for (int iProj = 0; iProj < numProj; iProj++)
              {
                for (int jProj = 0; jProj < numProj; jProj++)
                  {
                    DijVector[index] = std::sqrt(4 * M_PI) *
                                       multipoleTable[iProj * numProj + jProj] *
                                       Dij_out[iProj * numProj + jProj];
                    index++;
                  }
              }
          }
        else
          {
            for (int iProj = 0; iProj < numProj; iProj++)
              {
                for (int jProj = 0; jProj < numProj; jProj++)
                  {
                    DijVector[index] = std::sqrt(4 * M_PI) *
                                       multipoleTable[iProj * numProj + jProj] *
                                       (Dij_out[iProj * numProj + jProj] -
                                        Dij_in[iProj * numProj + jProj]);
                    index++;
                  }
              }
          }
      }
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
    // pcout << "Checking Radial Derivative Values: " << std::endl;
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
        // pcout << i << " " << r[i] << " " << Value << " " << derivativeValue
        //       << " " << der[i] << std::endl;
        // der[i] = derivativeValue;
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

} // namespace dftfe
