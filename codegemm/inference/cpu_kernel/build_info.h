// A backend label must describe the binary actually imported, not merely the
// Python name requested. This also catches an old installed cpu-1 -O2 binary.
#pragma once

#include <string>
#include <pybind11/pybind11.h>

#if !defined(CODEGEMM_CPU_VARIANT) || !defined(CODEGEMM_CPU_FLAGS)
#error "Build CPU extensions using the provided setup.py so flags are recorded."
#endif

// setuptools may supply -O3 earlier in a compile command. setup.py puts its
// -O0 last; this compile-time check makes accidental optimized baselines fail.
#if CODEGEMM_CPU_VARIANT < 4 && defined(__OPTIMIZE__)
#error "cpu-1/cpu-2/cpu-3 must be compiled with optimization disabled (-O0)."
#endif

#if CODEGEMM_CPU_VARIANT == 4 && !defined(__OPTIMIZE__)
#error "cpu-4 must be compiled with optimization enabled (-O3)."
#endif

// Reject inherited -Ofast/-ffast-math settings instead of silently comparing
// different numerical rules. Explicit NEON FMA is the same in variants 2-4.
#if defined(__FAST_MATH__)
#error "Fast-math is not part of the CPU variant comparison; remove that flag."
#endif

namespace codegemm_cpu {

inline pybind11::dict build_info() {
    pybind11::dict info;
    constexpr bool neon = CODEGEMM_CPU_VARIANT > 1;
    constexpr bool threaded = CODEGEMM_CPU_VARIANT >= 3;
    constexpr bool optimized = CODEGEMM_CPU_VARIANT == 4;
    info["backend"] = std::string("cpu-") + std::to_string(CODEGEMM_CPU_VARIANT);
    info["neon"] = neon;
    info["threaded"] = threaded;
    info["multithreaded"] = threaded;
    info["optimized"] = optimized;
    info["optimization"] = optimized ? "O3" : "O0";
    info["optimization_level"] = optimized ? "O3" : "O0";
    info["max_threads"] = threaded ? 4 : 1;
    info["compile_flags"] = CODEGEMM_CPU_FLAGS;
    info["compiler_flags"] = CODEGEMM_CPU_FLAGS;
    info["parallel_backend"] = CODEGEMM_CPU_PARALLEL_BACKEND;
    return info;
}

} // namespace codegemm_cpu
