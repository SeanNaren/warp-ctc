import torch
import pytest

criterion = torch.nn.CTCLoss(reduction='sum')
gpu_criterion = torch.nn.CTCLoss(reduction='sum').to('cuda')


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_simple():
    probs = torch.FloatTensor([[[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]]]).transpose(0, 1).contiguous()
    probs = probs.log_softmax(2).detach().requires_grad_()
    labels = torch.IntTensor([1, 2])
    label_sizes = torch.IntTensor([2])
    sizes = torch.IntTensor(probs.size(1)).fill_(probs.size(0))
    costs = criterion(probs,
                      labels,
                      sizes,
                      label_sizes,
                      )
    print('CPU_cost: %f' % costs.sum())
    probs = probs.clone().cuda().detach().requires_grad_()
    costs = gpu_criterion(probs,
                          labels,
                          sizes,
                          label_sizes,
                          )
    costs.backward()
    grads = probs.grad
    print('GPU_cost: %f' % costs.sum())
    print(grads.view(grads.size(0) * grads.size(1), grads.size(2)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("multiplier", [1.0, 200.0])
def test_medium(multiplier):
    probs = torch.FloatTensor([
        [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
        [[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.5, 0.2, 0.1]]
    ]).contiguous() * multiplier
    probs = probs.log_softmax(2).detach().requires_grad_()

    labels = torch.IntTensor([1, 2, 1, 2])
    label_sizes = torch.IntTensor([2, 2])
    sizes = torch.IntTensor([2, 2])
    costs = criterion(probs,
                      labels,
                      sizes,
                      label_sizes,
                      )
    print('CPU_cost: %f' % costs.sum())
    probs = probs.clone().cuda().detach().requires_grad_()
    costs = gpu_criterion(probs,
                          labels,
                          sizes,
                          label_sizes,
                          )
    costs.backward()
    grads = probs.grad
    print('GPU_cost: %f' % costs.sum())
    print(grads.view(grads.size(0) * grads.size(1), grads.size(2)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_empty_label():
    probs = torch.FloatTensor([
        [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
        [[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.5, 0.2, 0.1]]
    ]).contiguous()
    probs = probs.log_softmax(2).detach().requires_grad_()

    labels = torch.IntTensor([1, 2])
    label_sizes = torch.IntTensor([2, 0])
    sizes = torch.IntTensor([2, 2])
    costs = criterion(probs,
                      labels,
                      sizes,
                      label_sizes,
                      )
    print('CPU_cost: %f' % costs.sum())
    probs = probs.clone().cuda().detach().requires_grad_()
    costs = gpu_criterion(probs,
                          labels,
                          sizes,
                          label_sizes,
                          )
    costs.backward()
    grads = probs.grad
    print('GPU_cost: %f' % costs.sum())
    print(grads.view(grads.size(0) * grads.size(1), grads.size(2)))


if __name__ == '__main__':
    pytest.main([__file__])
