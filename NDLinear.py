"""
ND-Linear Layer with Configurable Dimensional Processing Order.

Extends Algorithm 1 from "NdLinear Is All You Need for Representation Learning"
(https://arxiv.org/abs/2503.17353) to support arbitrary dimension processing orders.
"""

import torch

from torch import nn as nn


class NDLinear(nn.Module):
    """
    NdLinear Layer with configurable per-dimension processing order.
    See also: NdLinear Is All You Need for Representation Learning (https://arxiv.org/abs/2503.17353)

    Parameters
    ----------
    input_dimensions : list[int]
        List of sizes D_1, ..., D_n for each input dimension (excluding batch)
    output_dimensions: list[int]
        List of sizes H_1, ..., H_n for each output dimension (excluding batch)
    bias : bool, optional
        Whether to use additive bias terms. Default: True
    dimensionality_order : list[int], optional
        Custom processing order for dimensions. Default: natural order
    """

    def __init__(self, input_dimensions: list[int],
                 output_dimensions: list[int],
                 bias: bool = True,
                 dimensionality_order: list[int] | None = None) -> None:
        super().__init__()
        assert len(input_dimensions) == len(
            output_dimensions), 'input_dimensions and output_dimensions must have the same length'

        self.input_dimension = input_dimensions
        self.output_dimensions = output_dimensions
        self.n = len(input_dimensions)

        self.dimensionality_order = dimensionality_order or list(range(self.n))

        assert min(dimensionality_order) >= 0 and max(dimensionality_order) < len(
            input_dimensions), 'dimensionality_order indices must be within [0, n-1]'

        self.layers = nn.ModuleList([
            nn.Linear(
                in_features=in_dim,
                out_features=out_dim,
                bias=bias
            )
            for in_dim, out_dim in zip(input_dimensions, output_dimensions)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transforms input tensor through sequential dimension-wise linear operations.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, D_1, ..., D_n)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, H_1, ..., H_n)
        """
        for current_dim in self.dimensionality_order:
            x = x.transpose(current_dim + 1, self.n)
            old_shape = x.shape
            x = x.reshape(-1, old_shape[-1])
            x = self.layers[current_dim](x)
            x = x.reshape(*old_shape[:-1], x.shape[-1])
            x = x.transpose(current_dim + 1, self.n)
        return x
