# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class DepthwiseConv2DOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsDepthwiseConv2DOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = DepthwiseConv2DOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def DepthwiseConv2DOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # DepthwiseConv2DOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # DepthwiseConv2DOptions
    def Padding(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # DepthwiseConv2DOptions
    def StrideW(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # DepthwiseConv2DOptions
    def StrideH(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # DepthwiseConv2DOptions
    def DepthMultiplier(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # DepthwiseConv2DOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # DepthwiseConv2DOptions
    def DilationWFactor(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 1

    # DepthwiseConv2DOptions
    def DilationHFactor(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 1

def DepthwiseConv2DOptionsStart(builder): builder.StartObject(7)
def DepthwiseConv2DOptionsAddPadding(builder, padding): builder.PrependInt8Slot(0, padding, 0)
def DepthwiseConv2DOptionsAddStrideW(builder, strideW): builder.PrependInt32Slot(1, strideW, 0)
def DepthwiseConv2DOptionsAddStrideH(builder, strideH): builder.PrependInt32Slot(2, strideH, 0)
def DepthwiseConv2DOptionsAddDepthMultiplier(builder, depthMultiplier): builder.PrependInt32Slot(3, depthMultiplier, 0)
def DepthwiseConv2DOptionsAddFusedActivationFunction(builder, fusedActivationFunction): builder.PrependInt8Slot(4, fusedActivationFunction, 0)
def DepthwiseConv2DOptionsAddDilationWFactor(builder, dilationWFactor): builder.PrependInt32Slot(5, dilationWFactor, 1)
def DepthwiseConv2DOptionsAddDilationHFactor(builder, dilationHFactor): builder.PrependInt32Slot(6, dilationHFactor, 1)
def DepthwiseConv2DOptionsEnd(builder): return builder.EndObject()


class DepthwiseConv2DOptionsT(object):

    # DepthwiseConv2DOptionsT
    def __init__(self):
        self.padding = 0  # type: int
        self.strideW = 0  # type: int
        self.strideH = 0  # type: int
        self.depthMultiplier = 0  # type: int
        self.fusedActivationFunction = 0  # type: int
        self.dilationWFactor = 1  # type: int
        self.dilationHFactor = 1  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        depthwiseConv2DOptions = DepthwiseConv2DOptions()
        depthwiseConv2DOptions.Init(buf, pos)
        return cls.InitFromObj(depthwiseConv2DOptions)

    @classmethod
    def InitFromObj(cls, depthwiseConv2DOptions):
        x = DepthwiseConv2DOptionsT()
        x._UnPack(depthwiseConv2DOptions)
        return x

    # DepthwiseConv2DOptionsT
    def _UnPack(self, depthwiseConv2DOptions):
        if depthwiseConv2DOptions is None:
            return
        self.padding = depthwiseConv2DOptions.Padding()
        self.strideW = depthwiseConv2DOptions.StrideW()
        self.strideH = depthwiseConv2DOptions.StrideH()
        self.depthMultiplier = depthwiseConv2DOptions.DepthMultiplier()
        self.fusedActivationFunction = depthwiseConv2DOptions.FusedActivationFunction()
        self.dilationWFactor = depthwiseConv2DOptions.DilationWFactor()
        self.dilationHFactor = depthwiseConv2DOptions.DilationHFactor()

    # DepthwiseConv2DOptionsT
    def Pack(self, builder):
        DepthwiseConv2DOptionsStart(builder)
        DepthwiseConv2DOptionsAddPadding(builder, self.padding)
        DepthwiseConv2DOptionsAddStrideW(builder, self.strideW)
        DepthwiseConv2DOptionsAddStrideH(builder, self.strideH)
        DepthwiseConv2DOptionsAddDepthMultiplier(builder, self.depthMultiplier)
        DepthwiseConv2DOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        DepthwiseConv2DOptionsAddDilationWFactor(builder, self.dilationWFactor)
        DepthwiseConv2DOptionsAddDilationHFactor(builder, self.dilationHFactor)
        depthwiseConv2DOptions = DepthwiseConv2DOptionsEnd(builder)
        return depthwiseConv2DOptions