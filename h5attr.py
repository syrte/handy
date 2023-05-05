"""
H5Attr: quick access to hdf5 data through attributes,
allowing `group.key` instead of `group['key']`.

It is more efficient than the previous H5File class,
as it does not create `f['a']` when calling `f['a/b']`.
"""


import h5py
import numpy as np
import pathlib
from collections.abc import Mapping

__all__ = ['H5Attr']


class H5Attr():
    '''quick access to hdf5 data through attributes,
    allowing `group.key` instead of `group['key']`.

    Examples
    --------
    # create example HDF5 file
    import h5py, io
    file = io.BytesIO()
    with h5py.File(file, 'w') as fp:
        fp['0'] = [1, 2]
        fp['a'] = [3, 4]
        fp['b/c'] = 5
        fp.attrs['d'] = 's'

    # open file
    f = H5Attr(file)

    # easy access to members
    f.a, f['a']

    # also work for subgroups, but note that f['b/c'] is more efficient
    # because it does not create f['b']
    f.b.c, f['b'].c, f['b/c']

    # convert integer keys to strings automatically
    f[0], f['0']

    # allow dict operations
    list(f), [key for key in f], 'a' in f

    # access to HDF5 attrs via a H5Attr wrapper
    f._attrs.d, f._attrs['d']

    # show summary of the data
    f._show()

    # lazy (default) and non-lazy mode
    f = H5Attr(file)
    f.a  # <HDF5 dataset "a": shape (2,), type "<i8">

    f = H5Attr(file, lazy=False)
    f.a  # array([3, 4])
    '''

    def __init__(self, path, lazy=True, **args):
        """
        Parameters
        ----------
        path: h5py Group, file path, or file-like object.
        lazy: bool, if true, dataset[()] will be returned.
        args: arguments used for opening HDF5 file.
        """
        if isinstance(path, Mapping):
            self.__data = path
        else:
            if isinstance(path, (str, pathlib.Path)):
                path = pathlib.Path(path).expanduser()
            self.__data = h5py.File(path, mode='r', **args)

        self.__lazy = lazy

    def __repr__(self):
        if isinstance(self.__data, h5py.Group):
            return "H5Attr\n file: {file}\n name: {name}\n keys: {keys}".format(
                file=self.__data.file.filename,
                name=self.__data.name,
                keys=", ".join(self.__data)
            )
        else:
            return "H5Attr\n keys: {keys}".format(
                keys=", ".join(self.__data)
            )

    def __dir__(self):
        props = list(self.__data) + super().__dir__()
        return props

    def __iter__(self):
        return self.__data.__iter__()

    def __len__(self):
        return self.__data.__len__()

    def __contains__(self, key):
        return self.__data.__contains__(key)

    def __getitem__(self, key):
        if isinstance(key, int):
            key = str(key)
        value = self.__data[key]

        if isinstance(value, h5py.Group):
            value = H5Attr(value, lazy=self.__lazy)
        elif not self.__lazy and isinstance(value, h5py.Dataset):
            value = value[()]

        return value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)
            # important for auto completing, see
            # https://github.com/ipython/ipython/issues/12828#issuecomment-902991224

    @property
    def _attrs(self):
        if hasattr(self.__data, 'attrs'):
            return H5Attr(self.__data.attrs)
        else:
            return H5Attr({})

    def _show(self):
        for key, value in self.__data.items():
            if isinstance(value, h5py.Group):
                print("{}/\t{} members".format(key, len(value)))
            elif isinstance(value, (h5py.Dataset, np.ndarray)):
                print("{}\t{} {}".format(key, value.dtype.name, value.shape))
            elif np.isscalar(value):
                print("{}\t{}".format(key, value))
            else:
                print("{}\t{} object".format(key, type(value)))
