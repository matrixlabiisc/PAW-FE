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

#include "AtomCenteredSphericalFunctionBessel.h"
#include "vector"
namespace dftfe
{
  AtomCenteredSphericalFunctionBessel::AtomCenteredSphericalFunctionBessel(
    double       RcParameter,
    double       RmaxParameter,
    unsigned int lParameter,
    double       normalizationConstant)
  {
    d_lQuantumNumber = lParameter;
    d_Rc             = RcParameter;
    d_cutOff         = RmaxParameter;
    using namespace boost::math::quadrature;
    AssertThrow(
      d_lQuantumNumber <= 2,
      dealii::ExcMessage(
        "DFT-FE Error:  Bessel functions only till LQuantumNo 2 is defined"));
    std::vector<double> q1 = {3.83171 / d_Rc, 1.84118 / d_Rc, 3.05424 / d_Rc};
    std::vector<double> q2 = {7.01559 / d_Rc, 5.33144 / d_Rc, 6.70613 / d_Rc};
    double              alpha =
      -(std::cyl_bessel_j(d_lQuantumNumber, q1[d_lQuantumNumber] * d_Rc)) /
      (std::cyl_bessel_j(d_lQuantumNumber, q2[d_lQuantumNumber] * d_Rc));

    auto f1 = [&](const double &x) {
      if (std::fabs(x) <= 1E-12)
        return (d_lQuantumNumber == 0 ? 1.0 : 0.0);
      else if (x > std::min(d_Rc, d_cutOff))
        return 0.0;
      else
        {
          double Value =
            x > d_Rc ?
              0.0 :
              (std::cyl_bessel_j(d_lQuantumNumber, q1[d_lQuantumNumber] * x) +
               alpha * (std::cyl_bessel_j(d_lQuantumNumber,
                                          q2[d_lQuantumNumber] * x)));
          return Value * pow(x, d_lQuantumNumber + 2);
        }
    };
    double NormalizationConstant =
      gauss_kronrod<double, 61>::integrate(f1, 0.0, d_cutOff, 15, 1e-12);
    std::cout << "Normalization constatnt: " << NormalizationConstant << " "
              << normalizationConstant << std::endl;
    d_NormalizationConstant = normalizationConstant;
    d_rMinVal               = getRadialValue(0.0);
  }

  double
  AtomCenteredSphericalFunctionBessel::getRadialValue(double r) const
  {
    double              Value = 0.0;
    std::vector<double> q1 = {3.83171 / d_Rc, 1.84118 / d_Rc, 3.05424 / d_Rc};
    std::vector<double> q2 = {7.01559 / d_Rc, 5.33144 / d_Rc, 6.70613 / d_Rc};
    double              alpha =
      -(std::cyl_bessel_j(d_lQuantumNumber, q1[d_lQuantumNumber] * d_Rc)) /
      (std::cyl_bessel_j(d_lQuantumNumber, q2[d_lQuantumNumber] * d_Rc));
    Value = r > d_Rc ?
              0.0 :
              (std::cyl_bessel_j(d_lQuantumNumber, q1[d_lQuantumNumber] * r) +
               alpha * (std::cyl_bessel_j(d_lQuantumNumber,
                                          q2[d_lQuantumNumber] * r)));
    return Value / d_NormalizationConstant;
  }
  std::vector<double>
  AtomCenteredSphericalFunctionBessel::getDerivativeValue(double r) const
  {
    AssertThrow(false, dftUtils::ExcNotImplementedYet());
  }

  double
  AtomCenteredSphericalFunctionBessel::getrMinVal() const
  {
    return d_rMinVal;
  }
} // end of namespace dftfe
