int cpu_ctc(THFloatTensor *probs,
                        THFloatTensor *grads,
                        THFloatTensor *labels_ptr,
                        THFloatTensor *label_sizes_ptr,
                        THFloatTensor *sizes,
                        int *alphabet_size,
                        int minibatch_size,
                        THFloatTensor *costs);