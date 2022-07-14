# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class UnidirectionalSequenceLSTMOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsUnidirectionalSequenceLSTMOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = UnidirectionalSequenceLSTMOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def UnidirectionalSequenceLSTMOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # UnidirectionalSequenceLSTMOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # UnidirectionalSequenceLSTMOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # UnidirectionalSequenceLSTMOptions
    def CellClip(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

    # UnidirectionalSequenceLSTMOptions
    def ProjClip(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

    # UnidirectionalSequenceLSTMOptions
    def TimeMajor(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

def UnidirectionalSequenceLSTMOptionsStart(builder): builder.StartObject(4)
def UnidirectionalSequenceLSTMOptionsAddFusedActivationFunction(builder, fusedActivationFunction): builder.PrependInt8Slot(0, fusedActivationFunction, 0)
def UnidirectionalSequenceLSTMOptionsAddCellClip(builder, cellClip): builder.PrependFloat32Slot(1, cellClip, 0.0)
def UnidirectionalSequenceLSTMOptionsAddProjClip(builder, projClip): builder.PrependFloat32Slot(2, projClip, 0.0)
def UnidirectionalSequenceLSTMOptionsAddTimeMajor(builder, timeMajor): builder.PrependBoolSlot(3, timeMajor, 0)
def UnidirectionalSequenceLSTMOptionsEnd(builder): return builder.EndObject()


class UnidirectionalSequenceLSTMOptionsT(object):

    # UnidirectionalSequenceLSTMOptionsT
    def __init__(self):
        self.fusedActivationFunction = 0  # type: int
        self.cellClip = 0.0  # type: float
        self.projClip = 0.0  # type: float
        self.timeMajor = False  # type: bool

    @classmethod
    def InitFromBuf(cls, buf, pos):
        unidirectionalSequenceLSTMOptions = UnidirectionalSequenceLSTMOptions()
        unidirectionalSequenceLSTMOptions.Init(buf, pos)
        return cls.InitFromObj(unidirectionalSequenceLSTMOptions)

    @classmethod
    def InitFromObj(cls, unidirectionalSequenceLSTMOptions):
        x = UnidirectionalSequenceLSTMOptionsT()
        x._UnPack(unidirectionalSequenceLSTMOptions)
        return x

    # UnidirectionalSequenceLSTMOptionsT
    def _UnPack(self, unidirectionalSequenceLSTMOptions):
        if unidirectionalSequenceLSTMOptions is None:
            return
        self.fusedActivationFunction = unidirectionalSequenceLSTMOptions.FusedActivationFunction()
        self.cellClip = unidirectionalSequenceLSTMOptions.CellClip()
        self.projClip = unidirectionalSequenceLSTMOptions.ProjClip()
        self.timeMajor = unidirectionalSequenceLSTMOptions.TimeMajor()

    # UnidirectionalSequenceLSTMOptionsT
    def Pack(self, builder):
        UnidirectionalSequenceLSTMOptionsStart(builder)
        UnidirectionalSequenceLSTMOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        UnidirectionalSequenceLSTMOptionsAddCellClip(builder, self.cellClip)
        UnidirectionalSequenceLSTMOptionsAddProjClip(builder, self.projClip)
        UnidirectionalSequenceLSTMOptionsAddTimeMajor(builder, self.timeMajor)
        unidirectionalSequenceLSTMOptions = UnidirectionalSequenceLSTMOptionsEnd(builder)
        return unidirectionalSequenceLSTMOptions
