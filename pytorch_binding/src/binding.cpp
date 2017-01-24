#include <iostream>
#include <vector>

#include <numeric>

#include "ctc.h"

//#ifdef TORCH_NOGPU
    #include "TH.h"
//#else
//    #include "THC.h"
//    #include "THCTensor.h"
//    #include "detail/reduce.h"
//#endif

extern "C" int cpu_ctc(THFloatTensor *probs,
                        THFloatTensor *grads,
                        THFloatTensor *labels_ptr,
                        THFloatTensor *label_sizes_ptr,
                        THFloatTensor *sizes,
                        int *alphabet_size,
                        int minibatch_size,
                        THFloatTensor *costs) {

    float *probs_ptr;
    probs_ptr = probs->storage->data + probs->storageOffset;
    float *grads_ptr;

    if (grads->storage) {
            grads_ptr = grads->storage->data + grads->storageOffset;
    } else {
            grads_ptr = NULL; // this will trigger the score forward code path
    }

    float* out_costs = new float[minibatch_size];

    int* sizes_ptr;
    sizes_ptr = sizes

    ctcOptions options;
    memset(&options, 0, sizeof(options));
    options.loc = CTC_CPU;
    options.num_threads = 0; // will use default number of threads

#if defined(CTC_DISABLE_OMP) || defined(APPLE)
    // have to use at least one
    options.num_threads = std::max(options.num_threads, (unsigned int) 1);
#endif

    size_t cpu_size_bytes;
    get_workspace_size(label_sizes_ptr, sizes,
                       (int) probs->size[1], minibatch_size,
                       options, &cpu_size_bytes);

    float* cpu_workspace = (float*) new unsigned char[cpu_size_bytes];

    compute_ctc_loss(probs_ptr, grads_ptr,
                     labels_ptr, label_sizes_ptr,
                     sizes, probs->size[1],
                     minibatch_size, out_costs,
                     cpu_workspace, options);

    delete cpu_workspace;
    delete sizes;
    delete labels_ptr;
    delete label_sizes_ptr;

    return 1;
}
