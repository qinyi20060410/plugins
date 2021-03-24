'''
Author: your name
Date: 2021-03-16 17:04:25
LastEditTime: 2021-03-16 19:31:15
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Demo/extension/mish/mish.py
'''

import torch  # Must import torch before C extension
from Mish import mish_forward, mish_backward
from torch.onnx import register_custom_op_symbolic


class MishCudaFunction(torch.autograd.Function):
    @staticmethod
    def symbolic(g, inp):
        return g.op("Mish::Mish", inp, name_s="Mish")

    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        return mish_forward(inp)

    @staticmethod
    def backward(ctx, grad_out):
        inp, = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return (None, )
        return mish_backward(inp, grad_out)


register_custom_op_symbolic("plugin::Mish", MishCudaFunction.symbolic, 11)


class MishCuda(torch.nn.Module):
    def forward(self, inp):
        return MishCudaFunction.apply(inp)
