"""
Tests are based on the torch7 bindings for warp-ctc. Reference numbers are also obtained from the tests.
"""
import unittest

import numpy as np
import torch
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss

places = 5

use_cuda = torch.cuda.is_available()


class TestCases(unittest.TestCase):
    def _run_grads(self, label_sizes, labels, probs, sizes, size_average=False,
                   length_average=False):
        ctc_loss = CTCLoss(size_average=size_average,
                           length_average=length_average)
        probs = Variable(probs, requires_grad=True)
        cost = ctc_loss(probs, labels, sizes, label_sizes)
        cost.backward()
        cpu_cost = cost.data[0]
        cpu_grad = probs.grad
        if use_cuda:
            probs = Variable(probs.data.cuda(), requires_grad=True)
            cost = ctc_loss(probs, labels, sizes, label_sizes)
            cost.backward()
            gpu_cost = cost.data[0]
            gpu_grad = probs.grad
        else:
            gpu_cost = cpu_cost
            gpu_grad = cpu_grad
        grads = probs.grad
        print(grads.view(grads.size(0) * grads.size(1), grads.size(2)))
        self.assertEqual(cpu_cost, gpu_cost)
        np.testing.assert_almost_equal(
            cpu_grad.numpy(), gpu_grad.cpu().numpy(), decimal=places)
        return cpu_cost, cpu_grad.numpy()

    def _test_normalization(self, label_sizes, labels, probs, sizes,
                            expected_cost, expected_grad, size_average=False,
                            length_average=False):
        cpu_cost, cpu_grad = self._run_grads(
            label_sizes, labels, probs, sizes, size_average=size_average,
            length_average=length_average)
        self.assertAlmostEqual(cpu_cost, expected_cost, places)
        np.testing.assert_almost_equal(expected_grad, cpu_grad, places)

    def test_simple(self):
        probs = torch.FloatTensor(
            [[[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]]]).transpose(
            0, 1).contiguous()
        labels = Variable(torch.IntTensor([1, 2]))
        label_sizes = Variable(torch.IntTensor([2]))
        sizes = Variable(torch.IntTensor([2]))
        # Basic checks
        cpu_cost, cpu_grad = self._run_grads(label_sizes, labels, probs, sizes)
        expected_cost = 2.4628584384918
        self.assertAlmostEqual(cpu_cost, expected_cost, places)
        # Check normalization by batch size
        self._test_normalization(
            label_sizes, labels, probs, sizes, cpu_cost, cpu_grad,
            size_average=True)
        # Check normalization by total frames
        self._test_normalization(
            label_sizes, labels, probs, sizes, cpu_cost / 2, cpu_grad / 2,
            length_average=True)

    def test_medium(self):
        probs = torch.FloatTensor([
            [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
            [[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.5, 0.2, 0.1]]
        ]).contiguous()

        labels = Variable(torch.IntTensor([1, 2, 1, 2]))
        label_sizes = Variable(torch.IntTensor([2, 2]))
        sizes = Variable(torch.IntTensor([2, 2]))
        # Basic checks
        cpu_cost, cpu_grad = self._run_grads(label_sizes, labels, probs, sizes)
        expected_cost = 6.0165174007416
        self.assertAlmostEqual(cpu_cost, expected_cost, places)
        # Check normalization by batch size
        self._test_normalization(
            label_sizes, labels, probs, sizes, cpu_cost / 2, cpu_grad / 2,
            size_average=True)
        # Check normalization by total frames
        self._test_normalization(
            label_sizes, labels, probs, sizes, cpu_cost / 4, cpu_grad / 4,
            length_average=True)

    def test_medium_stability(self):
        multiplier = 200
        probs = torch.FloatTensor([
            [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
            [[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.5, 0.2, 0.1]]
        ]).contiguous() * multiplier

        labels = Variable(torch.IntTensor([1, 2, 1, 2]))
        label_sizes = Variable(torch.IntTensor([2, 2]))
        sizes = Variable(torch.IntTensor([2, 2]))
        # Basic checks
        cpu_cost, cpu_grad = self._run_grads(label_sizes, labels, probs, sizes)
        expected_cost = 199.96618652344
        self.assertAlmostEqual(cpu_cost, expected_cost, places)
        # Check normalization by batch size
        self._test_normalization(
            label_sizes, labels, probs, sizes, cpu_cost / 2, cpu_grad / 2,
            size_average=True)
        # Check normalization by total frames
        self._test_normalization(
            label_sizes, labels, probs, sizes, cpu_cost / 4, cpu_grad / 4,
            length_average=True)

    def test_empty_label(self):
        probs = torch.FloatTensor([
            [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
            [[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.5, 0.2, 0.1]]
        ]).contiguous()

        labels = Variable(torch.IntTensor([1, 2]))
        label_sizes = Variable(torch.IntTensor([2, 0]))
        sizes = Variable(torch.IntTensor([2, 2]))
        # Basic checks
        cpu_cost, cpu_grad = self._run_grads(label_sizes, labels, probs, sizes)
        expected_cost = 6.416517496109
        self.assertAlmostEqual(cpu_cost, expected_cost, places)
        # Check normalization by batch size
        self._test_normalization(
            label_sizes, labels, probs, sizes, cpu_cost / 2, cpu_grad / 2,
            size_average=True)
        # Check normalization by total frames
        self._test_normalization(
            label_sizes, labels, probs, sizes, cpu_cost / 4, cpu_grad / 4,
            length_average=True)


if __name__ == '__main__':
    unittest.main()
