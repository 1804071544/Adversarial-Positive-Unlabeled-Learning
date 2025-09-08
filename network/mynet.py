import torch
import torch.nn as nn


class MyNET(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('this is mynet')

