#ifdef DFTFE_MINIMAL_COMPILE
template class symmetryClass<2, 2, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<3, 3, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<4, 4, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<5, 5, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<6, 6, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<6, 7, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<6, 8, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<6, 9, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<7, 7, dftfe::utils::MemorySpace::HOST>;
#  ifdef DFTFE_WITH_DEVICE
template class symmetryClass<2, 2, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<3, 3, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<4, 4, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<5, 5, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<6, 6, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<6, 7, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<6, 8, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<6, 9, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<7, 7, dftfe::utils::MemorySpace::DEVICE>;
#  endif
#else
template class symmetryClass<1, 1, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<1, 2, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<2, 2, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<2, 3, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<2, 4, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<3, 3, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<3, 4, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<3, 5, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<3, 6, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<4, 4, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<4, 5, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<4, 6, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<4, 7, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<4, 8, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<4, 10, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<4, 14, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<5, 5, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<5, 6, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<5, 7, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<5, 8, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<5, 9, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<5, 10, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<5, 14, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<6, 6, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<6, 7, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<6, 8, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<6, 9, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<6, 10, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<6, 11, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<6, 12, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<6, 14, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<7, 7, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<7, 8, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<7, 9, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<7, 10, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<7, 11, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<7, 12, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<7, 13, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<7, 14, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<8, 8, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<8, 9, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<8, 10, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<8, 11, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<8, 12, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<8, 13, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<8, 14, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<8, 15, dftfe::utils::MemorySpace::HOST>;
template class symmetryClass<8, 16, dftfe::utils::MemorySpace::HOST>;
#  ifdef DFTFE_WITH_DEVICE
template class symmetryClass<1, 1, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<1, 2, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<2, 2, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<2, 3, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<2, 4, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<3, 3, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<3, 4, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<3, 5, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<3, 6, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<4, 4, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<4, 5, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<4, 6, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<4, 7, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<4, 8, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<5, 5, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<5, 6, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<5, 7, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<5, 8, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<5, 9, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<5, 10, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<6, 6, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<6, 7, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<6, 8, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<6, 9, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<6, 10, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<6, 11, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<6, 12, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<7, 7, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<7, 8, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<7, 9, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<7, 10, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<7, 11, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<7, 12, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<7, 13, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<7, 14, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<8, 8, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<8, 9, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<8, 10, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<8, 11, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<8, 12, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<8, 13, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<8, 14, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<8, 15, dftfe::utils::MemorySpace::DEVICE>;
template class symmetryClass<8, 16, dftfe::utils::MemorySpace::DEVICE>;
#  endif
#endif
