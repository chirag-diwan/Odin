#pragma once

#ifndef ODIN_INLINE
#if defined (__clang__)
#define ODIN_INLINE [[clang::always_inline]] inline
#elif defined(__GNUC__)
#define ODIN_INLINE [[gnu::always_inline]] inline
#else
#warning ("Compiler specific inline for this compiler isn't supported yet")
#define ODIN_INLINE inline
#endif
#endif
