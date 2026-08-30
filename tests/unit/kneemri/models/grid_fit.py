"""A trunk is asked whether it resamples its position grid, and answers by its own signature."""

from typing import override

from torch import Tensor, nn

from kneemri.models.grid_fit import INTERPOLATE, GridFit


class FixedGrid(nn.Module):
    """A trunk whose forward accepts the resampling keyword — the AugReg ViT shape."""

    @override
    def forward(self, pixel_values: Tensor, interpolate_pos_encoding: bool = False) -> Tensor:
        return pixel_values if interpolate_pos_encoding else pixel_values


class OwnGrid(nn.Module):
    """A trunk that resamples inside its own forward and takes no keyword — the DINOv2 shape."""

    @override
    def forward(self, pixel_values: Tensor) -> Tensor:
        return pixel_values


def test_of() -> None:
    assert GridFit.of(FixedGrid()).interpolates
    assert not GridFit.of(OwnGrid()).interpolates


def test_extra() -> None:
    assert GridFit.of(FixedGrid()).extra() == {INTERPOLATE: True}
    assert GridFit.of(OwnGrid()).extra() == {}
