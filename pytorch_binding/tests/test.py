import torch
import torch.cuda

import warpctc_pytorch as warp_ctc

probs = torch.FloatTensor([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]]).transpose(0, 1).contiguous()

grads = torch.zeros(probs.size())
labels = torch.IntTensor([3, 3])
label_sizes = torch.IntTensor([2])

sizes = torch.IntTensor(probs.size(1)).fill_(probs.size(0))
minibatch_size = probs.size(1)
costs = torch.zeros(1)
warp_ctc.cpu_ctc(probs,
                 grads,
                 labels,
                 label_sizes,
                 sizes,
                 minibatch_size,
                 costs)

print(costs.sum())

probs = torch.FloatTensor([[[-5, -4, -3, -2, -1], [-10, -9, -8, -7, -6], [-15, -14, -13, -12, -11]]]) \
    .transpose(0, 1).contiguous()

grads = torch.zeros(probs.size())
labels = torch.IntTensor([2, 3])
label_sizes = torch.IntTensor([2])

sizes = torch.IntTensor(probs.size(1)).fill_(probs.size(0))
minibatch_size = probs.size(1)
costs = torch.zeros(1)
warp_ctc.cpu_ctc(probs,
                 grads,
                 labels,
                 label_sizes,
                 sizes,
                 minibatch_size,
                 costs)

print(costs.sum())

probs = torch.FloatTensor([
    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
    [[-5, -4, -3, -2, -1], [-10, -9, -8, -7, -6], [-15, -14, -13, -12, -11]]
]).transpose(0, 1).contiguous()

grads = torch.zeros(probs.size())
labels = torch.IntTensor([1, 3, 3, 2, 3])
label_sizes = torch.IntTensor([1, 2, 2])

sizes = torch.IntTensor([1, 3, 3])
minibatch_size = probs.size(1)
costs = torch.zeros(3)
warp_ctc.cpu_ctc(probs,
                 grads,
                 labels,
                 label_sizes,
                 sizes,
                 minibatch_size,
                 costs)

print(costs.sum())

probs = torch.FloatTensor([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]]).transpose(0,
                                                                                                 1).contiguous().cuda()

grads = torch.zeros(probs.size()).cuda()
labels = torch.IntTensor([3, 3])
label_sizes = torch.IntTensor([2])

sizes = torch.IntTensor(probs.size(1)).fill_(probs.size(0))
minibatch_size = probs.size(1)
costs = torch.zeros(1)
warp_ctc.gpu_ctc(probs,
                 grads,
                 labels,
                 label_sizes,
                 sizes,
                 minibatch_size,
                 costs)

print(costs.sum())
