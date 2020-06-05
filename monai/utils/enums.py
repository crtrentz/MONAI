# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union, Sequence


class InterpolationOrder(IntEnum):
    SPLINE0 = 0
    SPLINE1 = 1
    SPLINE2 = 2
    SPLINE3 = 3
    SPLINE4 = 4
    SPLINE5 = 5


"""
interp_order: the order of the spline interpolation. Default: ``FIXME``

    - ``"0``"
    - ``"1``"
    - ``"2``"
    - ``"3``"
    - ``"4``"
    - ``"5``"
"""

InterpolationOrderType = Union[int, InterpolationOrder]
InterpolationOrderSequenceType = Union[InterpolationOrder, Sequence[InterpolationOrder]]


class ExtendMode(Enum):
    REFLECT = "reflect"
    CONSTANT = "constant"
    NEAREST = "nearest"
    MIRROR = "mirror"
    WRAP = "wrap"


"""
extend_mode: how the input array is extended beyond its boundaries. Default: ``FIXME``

    - ``"reflect``": extends by reflecting about the edge of the last pixel.
    - ``"constant``": extends by filling all values beyond the edge with the same constant value, defined by the cval parameter.
    - ``"nearest``": extends by replicating the last pixel.
    - ``"mirror``": extends by reflecting about the center of the last pixel.
    - ``"wrap``": extends by wrapping around to the opposite edge.
"""

ExtendModeType = Union[str, ExtendMode]
ExtendModeSequenceType = Union[ExtendMode, Sequence[ExtendMode]]


class ResizeMode(Enum):
    CONSTANT = "constant"
    EDGE = "edge"
    SYMMETRIC = "symmetric"
    REFLECT = "reflect"
    WRAP = "wrap"


"""
resize_mode: how points outside the boundaries of the input are filled. Default: ``FIXME``

    - ``"constant``": pads with a constant value.
    - ``"edge``": pads with the edge values of array.
    - ``"symmetric``": pads with the reflection of the vector mirrored along the edge of the array.
    - ``"reflect``": pads with the reflection of the vector mirrored on the first and last values of the vector along each axis.
    - ``"wrap``": pads with the wrap of the vector along the axis. The first values are used to pad the end and the end values are used to pad the beginning.
"""

ResizeModeType = Union[str, ResizeMode]
ResizeModeSequenceType = Union[ResizeMode, Sequence[ResizeMode]]


class InterpolationMode(Enum):
    BILINEAR = "bilinear"
    NEAREST = "nearest"


"""
interp_mode: interpolation mode to calculate output values. Default: ``FIXME``

    - ``"bilinear``"
    - ``"nearest``"
"""

InterpolationModeType = Union[str, InterpolationMode]
InterpolationModeSequenceType = Union[InterpolationMode, Sequence[InterpolationMode]]


class PaddingMode(Enum):
    ZEROS = "zeros"
    BORDER = "border"
    REFLECTION = "reflection"


"""
padding_mode: padding mode for outside grid values. Default: ``FIXME``

    - ``"zeros``": uses ``0`` for out-of-bound grid locations.
    - ``"border``": uses border values for out-of-bound grid locations.
    - ``"reflection``": uses values at locations reflected by the border for out-of-bound grid locations. For location far away from the border, it will keep being reflected until becoming in bound.
"""

PaddingModeType = Union[str, PaddingMode]
PaddingModeSequenceType = Union[PaddingMode, Sequence[PaddingMode]]


class PadMode(Enum):
    CONSTANT = "constant"
    EDGE = "edge"
    LINEAR_RAMP = "linear_ramp"
    MAXIMUM = "maximum"
    MEAN = "mean"
    MEDIAN = "median"
    MINIMUM = "minimum"
    REFLECT = "reflect"
    SYMMETRIC = "symmetric"
    WRAP = "wrap"
    EMPTY = "empty"


"""
pad_mode: one of the following string values or a user supplied function. Default: ``FIXME``

    - ``"constant``": pads with a constant value.
    - ``"edge``": pads with the edge values of array.
    - ``"linear_ramp``": pads with the linear ramp between end_value and the array edge value.
    - ``"maximum``": pads with the maximum value of all or part of the vector along each axis.
    - ``"mean``": pads with the mean value of all or part of the vector along each axis.
    - ``"median``": pads with the median value of all or part of the vector along each axis.
    - ``"minimum``": pads with the minimum value of all or part of the vector along each axis.
    - ``"reflect``": Pads with the reflection of the vector mirrored on the first and last values of the vector along each axis.
    - ``"symmetric``": pads with the reflection of the vector mirrored along the edge of the array.
    - ``"wrap``": pads with the wrap of the vector along the axis. The first values are used to pad the end and the end values are used to pad the beginning.
    - ``"empty``": pads with undefined values.
"""

PadModeType = Union[str, PadMode]
PadModeSequenceType = Union[PadMode, Sequence[PadMode]]


class BlendMode(Enum):
    CONSTANT = "constant"
    GAUSSIAN = "gaussian"


"""
blend_mode: how to blend output of overlapping windows.

    - ``"constant``": gives equal weight to all predictions.
    - ``"gaussian``": gives less weight to predictions on edges of windows.
"""

BlendModeType = Union[str, BlendMode]
BlendModeSequenceType = Union[BlendMode, Sequence[BlendMode]]
