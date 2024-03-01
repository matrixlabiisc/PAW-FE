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



} // namespace dftfe
