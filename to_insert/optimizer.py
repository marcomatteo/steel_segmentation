import torch.nn as nn
from fastcore.foundation import L

#from: https://github.com/muellerzr/fastai_minima/blob/master/fastai_minima/optimizer.py#L159
def params(m):
    "Return all parameters of `m`"
    return [p for p in m.parameters()]

def convert_params(o:list) -> list:
    """
    Converts `o` into Pytorch-compatable param groups
    `o` should be a set of layer-groups that should be split in the optimizer
    Example:
    ```python
    def splitter(m): return convert_params([[m.a], [m.b]])
    ```
    Where `m` is a model defined as:
    ```python
    class RegModel(Module):
      def __init__(self): self.a,self.b = nn.Parameter(torch.randn(1)),nn.Parameter(torch.randn(1))
      def forward(self, x): return x*self.a + self.b
    ```
    """
    if not isinstance(o[0], dict):
        splitter = []
        for group in o:
            if not isinstance(group[0], nn.parameter.Parameter):
                group = L(group).map(params)[0]
            splitter.append({'params':group})
        return splitter
    return o