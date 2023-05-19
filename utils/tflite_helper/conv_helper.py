import numpy as np


def GetKernelShape(input_tensors) -> np.ndarray:

    """Description
    :type input_tensors:
    :param input_tensors:

    :raises:

    :rtype:
    """
    return input_tensors[1][0][1:]


def GetFilterCount(input_tensors) -> int:

    """Description
    :type input_tensors:
    :param input_tensors:

    :raises:

    :rtype:
    """
    return input_tensors[2][0][0]


def GetInputShape(input_tensors) -> np.ndarray:

    """Description
    :type input_tensors:
    :param input_tensors:

    :raises:

    :rtype:
    """
    return input_tensors[0][0]

def GetInputType(input_tensors) -> type:

    """ Description
    :type input_tensors:
    :param input_tensors:

    :raises:

    :rtype:
    """
    import tensorflow as tf
    from utils.tflite_helper import GetTensorType

    return GetTensorType(input_tensors[0])

def GetFilterType(input_tensors) -> type:

    """ Description
    :type input_tensors:
    :param input_tensors:

    :raises:

    :rtype:
    """
    import tensorflow as tf
    from utils.tflite_helper import GetTensorType

    return GetTensorType(input_tensors[2])
