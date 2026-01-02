#pragma once

#ifdef ENABLE_OVERLAP
#define PCG_ENABLE_OVERLAP 1
#else
#define PCG_ENABLE_OVERLAP 0
#endif

#ifdef USE_GHOST_EXCHANGE
#define PCG_USE_GHOST_EXCHANGE 1
#else
#define PCG_USE_GHOST_EXCHANGE 0
#endif