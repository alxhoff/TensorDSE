# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class SVDFOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsSVDFOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SVDFOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def SVDFOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # SVDFOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SVDFOptions
    def Rank(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # SVDFOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

def SVDFOptionsStart(builder): builder.StartObject(2)
def SVDFOptionsAddRank(builder, rank): builder.PrependInt32Slot(0, rank, 0)
def SVDFOptionsAddFusedActivationFunction(builder, fusedActivationFunction): builder.PrependInt8Slot(1, fusedActivationFunction, 0)
def SVDFOptionsEnd(builder): return builder.EndObject()


class SVDFOptionsT(object):

    # SVDFOptionsT
    def __init__(self):
        self.rank = 0  # type: int
        self.fusedActivationFunction = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        sVDFOptions = SVDFOptions()
        sVDFOptions.Init(buf, pos)
        return cls.InitFromObj(sVDFOptions)

    @classmethod
    def InitFromObj(cls, sVDFOptions):
        x = SVDFOptionsT()
        x._UnPack(sVDFOptions)
        return x

    # SVDFOptionsT
    def _UnPack(self, sVDFOptions):
        if sVDFOptions is None:
            return
        self.rank = sVDFOptions.Rank()
        self.fusedActivationFunction = sVDFOptions.FusedActivationFunction()

    # SVDFOptionsT
    def Pack(self, builder):
        SVDFOptionsStart(builder)
        SVDFOptionsAddRank(builder, self.rank)
        SVDFOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        sVDFOptions = SVDFOptionsEnd(builder)
        return sVDFOptions