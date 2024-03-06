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


} // namespace dftfe
