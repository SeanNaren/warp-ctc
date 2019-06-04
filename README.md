# PyTorch bindings for Warp-ctc built with VS2017

This is an extension onto the original repo found [here](https://github.com/baidu-research/warp-ctc).
This is a branch for Visual Studio 2017 building, with the errors fixed while installing from pytorch_bindings branch. The installation has been successfully performed in the following environments:
>Win10 cuda10.1 anaconda3 python3.7 pytorch1.0 vs2017

>Win7 cuda9.0 anaconda3 pytorch3.7 pytorch1.0 vs2015

Most of our modifications follow [amberblade's](https://github.com/amberblade/warp-ctc) (It is a Windows compatible version of [SeanNaren's work](https://github.com/SeanNaren/warp-ctc), but cannot work with pytorch1.0). In case you should modify it yourself, all the necessary modifications can be viewed in the commit [fix errors in vs2017 building](https://github.com/hzli-ucas/warp-ctc/commit/a19202c399b8e40adc96739df946d3bdb26eefac). You can also refer to "Specific Modifications" for all the modifications and the error messages that would have be encountered during the installation without the modifications.

## Installation

Install [PyTorch](https://github.com/pytorch/pytorch#installation) v1.0 or higher.

From within a new warp-ctc clone you could build WarpCTC like this:
```cmd
git clone -b vs2017 https://github.com/hzli-ucas/warp-ctc.git
cd warp-ctc
mkdir build
cd build
cmake -G "Visual Studio 15 2017 Win64" ..
```
If using vs2015 instead of vs2017 for building, replace the `cmake -G "Visual Studio 15 2017 Win64" ..` with `cmake -G "Visual Studio 14 2015 Win64" ..`.

Open the `warp-ctc/build/ctc_release.sln` with vs2017, and build the project `ALL_BUILD` (Release x64). Then install the bindings:
```cmd
cd ../pytorch_binding
python setup.py install
```

At last, copy `warp-ctc/build/Release/warpctc.dll` to the path where you install the warpctc. For example, for the installation performed with Anaconda3 within an environment named "pytorch", the path oughts to be: `Anaconda3/envs/pytorch/Lib/site-packages/warpctc_pytorch-0.1-py3.7-win-amd64.egg/warpctc_pytorch/`

You could use the following commands to check that warpctc_pytorch has been successfully installed:
```cmd
python tests/test_cpu.py
python tests/test_gpu.py
```

Example to use the bindings below.

```python
import torch
from warpctc_pytorch import CTCLoss
ctc_loss = CTCLoss()
# expected shape of seqLength x batchSize x alphabet_size
probs = torch.FloatTensor([[[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]]]).transpose(0, 1).contiguous()
labels = torch.IntTensor([1, 2])
label_sizes = torch.IntTensor([2])
probs_sizes = torch.IntTensor([2])
probs.requires_grad_(True)  # tells autograd to compute gradients for probs
cost = ctc_loss(probs, labels, probs_sizes, label_sizes)
cost.backward()
```

## Documentation

```
CTCLoss(size_average=False, length_average=False)
    # size_average (bool): normalize the loss by the batch size (default: False)
    # length_average (bool): normalize the loss by the total number of frames in the batch. If True, supersedes size_average (default: False)

forward(acts, labels, act_lens, label_lens)
    # acts: Tensor of (seqLength x batch x outputDim) containing output activations from network (before softmax)
    # labels: 1 dimensional Tensor containing all the targets of the batch in one large sequence
    # act_lens: Tensor of size (batch) containing size of each output sequence from the network
    # label_lens: Tensor of (batch) containing label length of each example
```

## Specific Modifications

***When build the project with vs2017, there might be***

***
```
2>F:/warp-ctc/src/ctc_entrypoint.cu(1): error : this declaration has no storage class or type specifier
2>F:/warp-ctc/src/ctc_entrypoint.cu(1): error : expected a ";"
```
`warp-ctc/src/ctc_entrypoint.cu`: 
change `ctc_entrypoint.cpp` to `#include "ctc_entrypoint.cpp"`
***

```
2>f:\warp-ctc\include\contrib\moderngpu\include\device\intrinsics.cuh(115): error : identifier "__shfl_up" is undefined
2>f:\warp-ctc\include\contrib\moderngpu\include\device\intrinsics.cuh(125): error : identifier "__shfl_up" is undefined
```
`warp-ctc/CMakeLists.txt`: 
comment out `set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70")`
***

```
3>f:\warp-ctc\tests\test.h(57): error : namespace "std" has no member "max"
```
`warp-ctc\tests\test.h`: 
add `#include <algorithm>`
***

```
3>test_gpu_generated_test_gpu.cu.obj : error LNK2019: 无法解析的外部符号 get_warpctc_version，该符号在函数 main 中被引用
3>test_gpu_generated_test_gpu.cu.obj : error LNK2019: 无法解析的外部符号 ctcGetStatusString，该符号在函数 "float __cdecl grad_check(int,int,class std::vector<float,class std::allocator<float> > &,class std::vector<class std::vector<int,class std::allocator<int> >,class std::allocator<class std::vector<int,class std::allocator<int> > > > const &,class std::vector<int,class std::allocator<int> > const &)" (?grad_check@@YAMHHAEAV?$vector@MV?$allocator@M@std@@@std@@AEBV?$vector@V?$vector@HV?$allocator@H@std@@@std@@V?$allocator@V?$vector@HV?$allocator@H@std@@@std@@@2@@2@AEBV?$vector@HV?$allocator@H@std@@@2@@Z) 中被引用
3>test_gpu_generated_test_gpu.cu.obj : error LNK2019: 无法解析的外部符号 compute_ctc_loss，该符号在函数 "float __cdecl grad_check(int,int,class std::vector<float,class std::allocator<float> > &,class std::vector<class std::vector<int,class std::allocator<int> >,class std::allocator<class std::vector<int,class std::allocator<int> > > > const &,class std::vector<int,class std::allocator<int> > const &)" (?grad_check@@YAMHHAEAV?$vector@MV?$allocator@M@std@@@std@@AEBV?$vector@V?$vector@HV?$allocator@H@std@@@std@@V?$allocator@V?$vector@HV?$allocator@H@std@@@std@@@2@@2@AEBV?$vector@HV?$allocator@H@std@@@2@@Z) 中被引用
3>test_gpu_generated_test_gpu.cu.obj : error LNK2019: 无法解析的外部符号 get_workspace_size，该符号在函数 "float __cdecl grad_check(int,int,class std::vector<float,class std::allocator<float> > &,class std::vector<class std::vector<int,class std::allocator<int> >,class std::allocator<class std::vector<int,class std::allocator<int> > > > const &,class std::vector<int,class std::allocator<int> > const &)" (?grad_check@@YAMHHAEAV?$vector@MV?$allocator@M@std@@@std@@AEBV?$vector@V?$vector@HV?$allocator@H@std@@@std@@V?$allocator@V?$vector@HV?$allocator@H@std@@@std@@@2@@2@AEBV?$vector@HV?$allocator@H@std@@@2@@Z) 中被引用
3>F:\warp-ctc\build\Release\test_gpu.exe : fatal error LNK1120: 4 个无法解析的外部命令
```
`warp-ctc\include\ctc.h`:
put `__declspec(dllexport)` in front of the four variables/functions
***

***When run the following command,***
```cmd
cd ../pytorch_binding
python setup.py install
```
***there might be***

***
```
(pytorch) F:\warp-ctc\pytorch_binding>python setup.py install
Could not find libwarpctc.so in ../build.
Build warp-ctc and set WARP_CTC_PATH to the location of libwarpctc.so (default is '../build')
```
`warp-ctc/pytorch_binding/steup.py`:
change `.so` to `.dll`
change `../build` to `../build/Release`
change `libwarpctc` to `warpctc`
***

```
binding.obj : error LNK2001: 无法解析的外部符号 "struct THCState * state" (?state@@3PEAUTHCState@@EA)
build\lib.win-amd64-3.7\warpctc_pytorch\_warp_ctc.cp37-win_amd64.pyd : fatal error LNK1120: 1 个无法解析的外部命令
error: command 'C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\bin\\HostX86\\x64\\link.exe' failed with exit status 1120
```
`warp-ctc/pyrotch_binding/src/binding.cpp`:
change `extern THCState* state;` to `THCState* state = at::globalContext().lazyInitCUDA();`
***

***The error might occur using the bindings***

***
```
>>> import warpctc_pytorch
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Users\dell\Anaconda3\envs\pytorch\lib\site-packages\warpctc_pytorch-0.1-py3.7-win-amd64.egg\warpctc_pytorch\__init__.py", line 6, in <module>
    from ._warp_ctc import *
ImportError: DLL load failed: 找不到指定的模块。
```
copy `warp-ctc/build/Release/warpctc.dll` to `where/you/install/the/warpctc_pytorch/`
