// CPU variants share one scheduling implementation so scalar-versus-NEON
// comparisons use the same partitioning, barriers, and four-worker limit.
#pragma once

#include <ATen/Parallel.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>

#ifndef CODEGEMM_CPU_SCALAR_MT
#define CODEGEMM_CPU_SCALAR_MT 0
#endif

namespace codegemm_cpu {

// Call function(begin, end) on disjoint pieces of [0, count), then join every
// worker before returning. Callbacks operate on raw pointers, not Tensor APIs.
// A caller may request fewer threads through torch.set_num_threads().
template <class Function>
void for_chunks(int64_t count, const Function& function) {
#if CODEGEMM_CPU_VARIANT >= 3 || CODEGEMM_CPU_SCALAR_MT
    const int64_t workers = std::min<int64_t>({4, at::get_num_threads(), count});
#if AT_PARALLEL_OPENMP && defined(_OPENMP)
    // Use the OpenMP runtime linked by PyTorch, but explicitly limit the team:
    // ATen parallel_for alone could wake a globally configured larger team.
    // Existing parallel regions execute serially here to avoid nested teams.
    if (workers > 1 && !at::in_parallel_region() && !omp_in_parallel()) {
        std::exception_ptr error;
        std::atomic_flag caught = ATOMIC_FLAG_INIT;
#pragma omp parallel num_threads(workers)
        {
            // OMP_DYNAMIC can provide fewer workers than requested. Using
            // the actual team size ensures that no output range is missed.
            const int64_t team = omp_get_num_threads();
            const int64_t worker = omp_get_thread_num();
            try {
                function(count * worker / team, count * (worker + 1) / team);
            } catch (...) {
                // OpenMP cannot propagate exceptions across its region.
                // Keep the first exception and rethrow after the join barrier.
                if (!caught.test_and_set(std::memory_order_relaxed)) {
                    error = std::current_exception();
                }
            }
        }
        if (error) {
            std::rethrow_exception(error);
        }
        return;
    }
#else
    // Native-pool PyTorch builds already own worker threads. Submit at most
    // four useful work chunks, without changing the process's global settings.
    if (workers > 1) {
        at::parallel_for(0, workers, 1, [&](int64_t begin, int64_t end) {
            for (int64_t worker = begin; worker < end; ++worker) {
                function(count * worker / workers, count * (worker + 1) / workers);
            }
        });
        return;
    }
#endif
#endif
    function(0, count);
}

} // namespace codegemm_cpu
