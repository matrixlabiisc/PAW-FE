#ifdef DFTFE_MINIMAL_COMPILE
template class poissonSolverProblem<2, 2>;
template class poissonSolverProblem<3, 3>;
template class poissonSolverProblem<4, 4>;
template class poissonSolverProblem<5, 5>;
template class poissonSolverProblem<6, 6>;
template class poissonSolverProblem<6, 7>;
template class poissonSolverProblem<6, 8>;
template class poissonSolverProblem<6, 9>;
template class poissonSolverProblem<7, 7>;
#else
template class poissonSolverProblem<1, 1>;
template class poissonSolverProblem<1, 2>;
template class poissonSolverProblem<2, 2>;
template class poissonSolverProblem<2, 3>;
template class poissonSolverProblem<2, 4>;
template class poissonSolverProblem<3, 3>;
template class poissonSolverProblem<3, 4>;
template class poissonSolverProblem<3, 5>;
template class poissonSolverProblem<3, 6>;
template class poissonSolverProblem<4, 4>;
template class poissonSolverProblem<4, 5>;
template class poissonSolverProblem<4, 6>;
template class poissonSolverProblem<4, 7>;
template class poissonSolverProblem<4, 8>;
template class poissonSolverProblem<5, 5>;
template class poissonSolverProblem<5, 6>;
template class poissonSolverProblem<5, 7>;
template class poissonSolverProblem<5, 8>;
template class poissonSolverProblem<5, 9>;
template class poissonSolverProblem<5, 10>;
template class poissonSolverProblem<5, 11>;
template class poissonSolverProblem<5, 12>;
template class poissonSolverProblem<5, 13>;
template class poissonSolverProblem<5, 14>;
template class poissonSolverProblem<6, 6>;
template class poissonSolverProblem<6, 7>;
template class poissonSolverProblem<6, 8>;
template class poissonSolverProblem<6, 9>;
template class poissonSolverProblem<6, 10>;
template class poissonSolverProblem<6, 11>;
template class poissonSolverProblem<6, 12>;
template class poissonSolverProblem<6, 13>;
template class poissonSolverProblem<6, 14>;
template class poissonSolverProblem<7, 7>;
template class poissonSolverProblem<7, 8>;
template class poissonSolverProblem<7, 9>;
template class poissonSolverProblem<7, 10>;
template class poissonSolverProblem<7, 11>;
template class poissonSolverProblem<7, 12>;
template class poissonSolverProblem<7, 13>;
template class poissonSolverProblem<7, 14>;
template class poissonSolverProblem<8, 8>;
template class poissonSolverProblem<8, 9>;
template class poissonSolverProblem<8, 10>;
template class poissonSolverProblem<8, 11>;
template class poissonSolverProblem<8, 12>;
template class poissonSolverProblem<8, 13>;
template class poissonSolverProblem<8, 14>;
template class poissonSolverProblem<8, 15>;
template class poissonSolverProblem<8, 16>;
#endif
