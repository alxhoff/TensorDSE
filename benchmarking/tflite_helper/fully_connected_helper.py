def GetUnits(output_tensors) -> int:
    
    """ Description
    :type output_tensors:
    :param output_tensors:

    :raises:

    :rtype:
    """
    return output_tensors[0][0][-1]


def GetNumDims(options):

    """Description
    :type options:
    :param options:

    :raises:

    :rtype:
    """
    return options["KeepNumDims"]