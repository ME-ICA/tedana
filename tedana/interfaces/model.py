from tedana.interfaces.data import MultiEchoData


class TEDModelFitter():
    """
    Object to fit TE-dependent models to input data and select valid components

    ****************************************************************************
    This could be built like an sklearn class where you call .fit() and even
    .select() to fit models and select components
    ****************************************************************************

    Parameters
    ----------
    medata : tedana.interfaces.data.MultiEchoData
    """

    def __init__(self, medata):
        if not isinstance(medata, MultiEchoData):
            raise TypeError("Input must be an instance of MultiEchoData.")
        self.medata = medata
