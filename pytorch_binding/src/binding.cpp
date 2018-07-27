#include <iostream>
#include <vector>

#include <numeric>

#include "ctc.h"

#ifdef WARPCTC_ENABLE_GPU
    #include "THC.h"
    #include "THCTensor.h"
    #include "detail/reduce.h"
    extern THCState* state;
#else
    #include "TH.h"
#endif

extern "C" int cpu_ctc(THFloatTensor *probs,
                       THFloatTensor *grads,
                       THIntTensor *labels,
                       THIntTensor *label_sizes,
                       THIntTensor *sizes,
                       int minibatch_size,
                       THFloatTensor *costs,
                       int blank_label)
{
    float *probs_ptr = THFloatTensor_data(probs);
    float *grads_ptr;
    if (THFloatTensor_storage(grads)) {
        grads_ptr = THFloatTensor_data(grads);
    } else {
        grads_ptr = NULL; // this will trigger the score forward code path
    }

    int *sizes_ptr = THIntTensor_data(sizes);
    int *labels_ptr = THIntTensor_data(labels);
    int *label_sizes_ptr = THIntTensor_data(label_sizes);
    float *costs_ptr = THFloatTensor_data(costs);

    int probs_size = THFloatTensor_size(probs, 2);

    ctcOptions options;
    memset(&options, 0, sizeof(options));
    options.loc = CTC_CPU;
    options.num_threads = 0; // will use default number of threads
    options.blank_label = blank_label;

#if defined(CTC_DISABLE_OMP) || defined(APPLE)
    // have to use at least one
    options.num_threads = std::max(options.num_threads, (unsigned int) 1);
#endif

    size_t cpu_size_bytes;
    get_workspace_size(label_sizes_ptr, sizes_ptr,
                       probs_size, minibatch_size,
                       options, &cpu_size_bytes);

    float* cpu_workspace = new float[cpu_size_bytes / sizeof(float)];

    compute_ctc_loss(probs_ptr, grads_ptr,
                     labels_ptr, label_sizes_ptr,
                     sizes_ptr, probs_size,
                     minibatch_size, costs_ptr,
                     cpu_workspace, options);

    delete[] cpu_workspace;
    return 1;
}

#ifdef WARPCTC_ENABLE_GPU
extern "C" int gpu_ctc(THCudaTensor *probs,
                       THCudaTensor *grads,
                       THIntTensor *labels,
                       THIntTensor *label_sizes,
                       THIntTensor *sizes,
                       int minibatch_size,
                       THFloatTensor *costs,
                       int blank_label)
{
    float *probs_ptr = THCudaTensor_data(state, probs);
    float *grads_ptr;
    if (THCudaTensor_storage(state, grads)) {
        grads_ptr = THCudaTensor_data(state, grads);
    } else {
        grads_ptr = NULL; // this will trigger the score forward code path
    }

    int *sizes_ptr = THIntTensor_data(sizes);
    int *labels_ptr = THIntTensor_data(labels);
    int *label_sizes_ptr = THIntTensor_data(label_sizes);
    float *costs_ptr = THFloatTensor_data(costs);

    int probs_size = THFloatTensor_size(probs, 2);

    ctcOptions options;
    memset(&options, 0, sizeof(options));
    options.loc = CTC_GPU;
    options.blank_label = blank_label;
    options.stream = THCState_getCurrentStream(state);

    size_t gpu_size_bytes;
    get_workspace_size(label_sizes_ptr, sizes_ptr,
                       probs_size, minibatch_size,
                       options, &gpu_size_bytes);

    void* gpu_workspace = THCudaMalloc(state, gpu_size_bytes);

    compute_ctc_loss(probs_ptr, grads_ptr,
                     labels_ptr, label_sizes_ptr,
                     sizes_ptr, probs_size,
                     minibatch_size, costs_ptr,
                     gpu_workspace, options);

    THCudaFree(state, (void *) gpu_workspace);
    return 1;
}
#endif
