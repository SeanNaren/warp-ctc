import torch
import warpctc_pytorch as warp_ctc
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn import Module
from torch.nn.modules.loss import _assert_no_grad

from ._warp_ctc import *


class _CTC(Function):
    @staticmethod
    def forward(ctx, acts, labels, act_lens, label_lens, size_average=False,
                length_average=False):
        with torch.no_grad():

            is_cuda = True if acts.is_cuda else False
            acts = acts.contiguous()
            loss_func = warp_ctc.gpu_ctc if is_cuda else warp_ctc.cpu_ctc
            grads = torch.zeros(acts.size()).type_as(acts)
            minibatch_size = acts.size(1)

            # Output for debugging
            print("_CTC forward function: minibatch size: " + str(minibatch_size))

            costs = torch.zeros(minibatch_size).cpu()
            loss_func(acts,
                      grads,
                      labels,
                      label_lens,
                      act_lens,
                      minibatch_size,
                      costs)

            # Output for debugging
            print("_CTC forward function: costs before summation: " + str(costs))

            costs = torch.FloatTensor([costs.sum()])

            if length_average:
                # Compute the avg. log-probability per batch sample and frame.
                total_length = torch.sum(act_lens)
                grads = grads / total_length
                costs = costs / total_length
            elif size_average:
                # Compute the avg. log-probability per batch sample.
                grads = grads / minibatch_size
                costs = costs / minibatch_size

            # Replaced with " torch.no_grad()" around the whole function body
            # The goal was to not track the computation with autograd appearantly.
            # https://stackoverflow.com/questions/49837638/what-is-volatile-variable-in-pytorch
            # This is now done differently
            # https://pytorch.org/2018/04/22/0_4_0-migration-guide.html
            # See: https://discuss.pytorch.org/t/torch-no-grad/12296
            # https://github.com/SeanNaren/deepspeech.pytorch/issues/277
            # https://discuss.pytorch.org/t/why-volatility-changed-after-operation-parameter-parameter-0-01-parameter-grad/11639
            # ctx.grads = Variable(grads, volatile=True)
            ctx.grads = grads

            # Debugging output
            # print(">>>  ctx.grads: " + str(ctx.grads))
            # print(">>>  ctx.grads.requires_grad: " + str(ctx.grads.requires_grad))

        return costs

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.grads, None, None, None, None, None


class CTCLoss(Module):
    """
    Parameters:
        size_average (bool): normalize the loss by the batch size
            (default: `False`)
        length_average (bool): normalize the loss by the total number of frames
            in the batch. If `True`, supersedes `size_average`
            (default: `False`)
    """
    def __init__(self, size_average=False, length_average=False):
        super(CTCLoss, self).__init__()
        self.ctc = _CTC.apply
        self.size_average = size_average
        self.length_average = length_average

    def forward(self, acts, labels, act_lens, label_lens):
        """
        acts: Tensor of (seqLength x batch x outputDim) containing output from network
        labels: 1 dimensional Tensor containing all the targets of the batch in one sequence
        act_lens: Tensor of size (batch) containing size of each output sequence from the network
        label_lens: Tensor of (batch) containing label length of each example
        """

        print("labels.size()" + str(labels.size()))
        if not len(labels.size()) == 1:
            raise RuntimeError("Error: the labels must be one dimensional and contain\n" +
                               "all label sequences in the examples of the batch concatenated." +
                               "\nBut got however labels.size(): " + str(labels.size()) + "\n" +
                               "which is not the right usage.")
        #assert len(labels.size()) == 1  # labels must be 1 dimensional

        sum_label_lens = torch.sum(label_lens)
        # label_lens contains the lengths of the label sequences (sub-ranges)
        # that are concatenated together in labels. Hence the sum of label_lens
        # should be the length of labels
        if not sum_label_lens == len(labels):
            raise RuntimeError("Error: length of labels is: " + str(len(labels)) +
                               " but sum of label lengths in " + str(labels) +
                               " is " + str(sum_label_lens) + "." +
                               "\nPlease make sure the label lengths" +
                               " add up to the number of labels")

        _assert_no_grad(labels)
        _assert_no_grad(act_lens)
        _assert_no_grad(label_lens)
        return self.ctc(acts, labels, act_lens, label_lens, self.size_average,
                        self.length_average)
