#ifdef DFTFE_MINIMAL_COMPILE
template class forceClass<2, 2, dftfe::utils::MemorySpace::HOST>;
template class forceClass<3, 3, dftfe::utils::MemorySpace::HOST>;
template class forceClass<4, 4, dftfe::utils::MemorySpace::HOST>;
template class forceClass<5, 5, dftfe::utils::MemorySpace::HOST>;
template class forceClass<6, 6, dftfe::utils::MemorySpace::HOST>;
template class forceClass<6, 7, dftfe::utils::MemorySpace::HOST>;
template class forceClass<6, 8, dftfe::utils::MemorySpace::HOST>;
template class forceClass<6, 9, dftfe::utils::MemorySpace::HOST>;
template class forceClass<7, 7, dftfe::utils::MemorySpace::HOST>;
#  ifdef DFTFE_WITH_DEVICE
template class forceClass<2, 2, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<3, 3, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<4, 4, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<5, 5, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<6, 6, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<6, 7, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<6, 8, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<6, 9, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<7, 7, dftfe::utils::MemorySpace::DEVICE>;
#  endif
#else
template class forceClass<1, 1, dftfe::utils::MemorySpace::HOST>;
template class forceClass<1, 2, dftfe::utils::MemorySpace::HOST>;
template class forceClass<2, 2, dftfe::utils::MemorySpace::HOST>;
template class forceClass<2, 3, dftfe::utils::MemorySpace::HOST>;
template class forceClass<2, 4, dftfe::utils::MemorySpace::HOST>;
template class forceClass<3, 3, dftfe::utils::MemorySpace::HOST>;
template class forceClass<3, 4, dftfe::utils::MemorySpace::HOST>;
template class forceClass<3, 5, dftfe::utils::MemorySpace::HOST>;
template class forceClass<3, 6, dftfe::utils::MemorySpace::HOST>;
template class forceClass<4, 4, dftfe::utils::MemorySpace::HOST>;
template class forceClass<4, 5, dftfe::utils::MemorySpace::HOST>;
template class forceClass<4, 6, dftfe::utils::MemorySpace::HOST>;
template class forceClass<4, 7, dftfe::utils::MemorySpace::HOST>;
template class forceClass<4, 8, dftfe::utils::MemorySpace::HOST>;
template class forceClass<5, 5, dftfe::utils::MemorySpace::HOST>;
template class forceClass<5, 6, dftfe::utils::MemorySpace::HOST>;
template class forceClass<5, 7, dftfe::utils::MemorySpace::HOST>;
template class forceClass<5, 8, dftfe::utils::MemorySpace::HOST>;
template class forceClass<5, 9, dftfe::utils::MemorySpace::HOST>;
template class forceClass<5, 10, dftfe::utils::MemorySpace::HOST>;
template class forceClass<5, 11, dftfe::utils::MemorySpace::HOST>;
template class forceClass<5, 12, dftfe::utils::MemorySpace::HOST>;
template class forceClass<5, 13, dftfe::utils::MemorySpace::HOST>;
template class forceClass<5, 14, dftfe::utils::MemorySpace::HOST>;
template class forceClass<6, 6, dftfe::utils::MemorySpace::HOST>;
template class forceClass<6, 7, dftfe::utils::MemorySpace::HOST>;
template class forceClass<6, 8, dftfe::utils::MemorySpace::HOST>;
template class forceClass<6, 9, dftfe::utils::MemorySpace::HOST>;
template class forceClass<6, 10, dftfe::utils::MemorySpace::HOST>;
template class forceClass<6, 11, dftfe::utils::MemorySpace::HOST>;
template class forceClass<6, 12, dftfe::utils::MemorySpace::HOST>;
template class forceClass<6, 13, dftfe::utils::MemorySpace::HOST>;
template class forceClass<6, 14, dftfe::utils::MemorySpace::HOST>;
template class forceClass<7, 7, dftfe::utils::MemorySpace::HOST>;
template class forceClass<7, 8, dftfe::utils::MemorySpace::HOST>;
template class forceClass<7, 9, dftfe::utils::MemorySpace::HOST>;
template class forceClass<7, 10, dftfe::utils::MemorySpace::HOST>;
template class forceClass<7, 11, dftfe::utils::MemorySpace::HOST>;
template class forceClass<7, 12, dftfe::utils::MemorySpace::HOST>;
template class forceClass<7, 13, dftfe::utils::MemorySpace::HOST>;
template class forceClass<7, 14, dftfe::utils::MemorySpace::HOST>;
template class forceClass<8, 8, dftfe::utils::MemorySpace::HOST>;
template class forceClass<8, 9, dftfe::utils::MemorySpace::HOST>;
template class forceClass<8, 10, dftfe::utils::MemorySpace::HOST>;
template class forceClass<8, 11, dftfe::utils::MemorySpace::HOST>;
template class forceClass<8, 12, dftfe::utils::MemorySpace::HOST>;
template class forceClass<8, 13, dftfe::utils::MemorySpace::HOST>;
template class forceClass<8, 14, dftfe::utils::MemorySpace::HOST>;
template class forceClass<8, 15, dftfe::utils::MemorySpace::HOST>;
template class forceClass<8, 16, dftfe::utils::MemorySpace::HOST>;
#  ifdef DFTFE_WITH_DEVICE
template class forceClass<1, 1, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<1, 2, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<2, 2, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<2, 3, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<2, 4, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<3, 3, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<3, 4, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<3, 5, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<3, 6, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<4, 4, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<4, 5, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<4, 6, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<4, 7, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<4, 8, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<5, 5, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<5, 6, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<5, 7, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<5, 8, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<5, 9, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<5, 10, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<5, 11, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<5, 12, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<5, 13, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<5, 14, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<6, 6, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<6, 7, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<6, 8, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<6, 9, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<6, 10, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<6, 11, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<6, 12, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<6, 13, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<6, 14, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<7, 7, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<7, 8, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<7, 9, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<7, 10, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<7, 11, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<7, 12, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<7, 13, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<7, 14, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<8, 8, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<8, 9, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<8, 10, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<8, 11, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<8, 12, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<8, 13, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<8, 14, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<8, 15, dftfe::utils::MemorySpace::DEVICE>;
template class forceClass<8, 16, dftfe::utils::MemorySpace::DEVICE>;
#  endif
#endif
