"""
Easier access to hdf5 subgroups/datasets
by using `group.key` instead of `group['key']`.

Usage:
    hg = H5File('data.h5')
    a = hg.a
    a = hg['a']
    a_b = hg.a.b
    a_b = hg['a/b']
    a_b_arr = hg.a.b.value  # using .value to load the array

    # list available properties
    dir(hg)

    # or print
    print(hg)

    # attrs
    a = hg.attrs.a

    # non-lazy mode
    hg = H5File('data.h5', lazy=False)
    a_b_arr = hg.a.b  # no need of .value

    # add new property (original file will not be changed)
    hg.b = 1

    # access properties starting with non-alphabetic
    a_1 = hg.a.1  # SyntaxError: invalid syntax
    a_1 = hg.a['1']
    a_1 = hg['a/1']

    # slicing
    sl = hg[slice]
    sl.x == hg.x[slice]
    sl.y == hg.y[slice]

    # slicing only takes effect on direct dataset
    sl.dataset == hg.dataset[slice]
    sl.group.dataset == hg.group.dataset

    # slice of slice
    hg[slice1][slice2].x == hg.x[slice1][slice2]
    # slice of slice is not efficient, don't use it too much.
"""

from __future__ import print_function
import h5py
import numpy as np
from six import string_types


__all__ = ['H5File']


class H5Group(object):
    '''Wrap of hdf5 group for quick access.
    '''

    def __init__(self, file, lazy=True):
        """
        Parameters
        ----------
        file : h5py.Group or file path.
        lazy : bool
        """
        if isinstance(file, string_types):
            file = h5py.File(file, 'r')

        self.__dict__['_file_'] = file
        self.__dict__['_lazy_'] = lazy
        self.__dict__['_keys_'] = list(file.keys())

        if hasattr(file, 'attrs') and file.attrs:
            self.__dict__['_keys_'] += ['attrs']

    def __dir__(self):
        return self._keys_

    def __str__(self):
        return "file:\t{file}\nname:\t{name}\nkeys:\t{keys}".format(
            file=self._file_.file.filename,
            name=self._file_.name,
            keys="\n\t".join(self._keys_)
        )

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]

    def __getitem__(self, key):
        # slice
        if not isinstance(key, string_types):
            return H5Slice(self, key)

        # hierarchical key
        elif '/' in key:
            keys = key.strip('/').split('/')
            value = self
            for key in keys:
                value = value[key]
            return value

        # simple key
        else:
            if key not in self._keys_:
                raise AttributeError("no attribute: '%s'" % key)
            elif key in self.__dict__:
                return self.__dict__[key]
            else:
                return self._load_(key)

    def __setitem__(self, key, value):
        if not isinstance(key, string_types):
            raise TypeError("key must be a string")
        elif '/' in key:
            raise ValueError("key with '/' is not supported")
        else:
            self.__dict__[key] = value
            if key not in self._keys_:
                self._keys_.append(key)

    def __delitem__(self, key):
        if not isinstance(key, string_types):
            raise TypeError("key must be a string")
        elif '/' in key:
            raise ValueError("key with '/' is not supported")
        else:
            if key not in self._keys_:
                raise AttributeError("No attribute: '%s'" % key)
            elif key in self.__dict__:
                del self.__dict__[key]

    def _load_(self, key):
        if key == 'attrs':
            value = H5Attrs(self._file_.attrs)
        else:
            value = self._file_[key]
            if isinstance(value, h5py.Group):
                value = H5Group(value, lazy=self._lazy_)
            elif not self._lazy_ and isinstance(value, h5py.Dataset):
                value = value.value
        self.__dict__[key] = value
        return value

    def _show_(self):
        for key in self._keys_:
            value = self[key]
            if isinstance(value, (h5py.Dataset, np.ndarray)):
                print("{}:\n\t{:>5s} {}".format(
                    key, value.dtype.str.strip(">|<"), value.shape)
                )
            else:
                print("{}:\n\t{}".format(key, value))


class H5Slice(H5Group):
    '''Slice of H5Group
    '''

    def __init__(self, group, slice):
        self.__dict__['_group_'] = group
        self.__dict__['_slice_'] = slice
        self.__dict__['_keys_'] = dir(group)

    def __str__(self):
        return "{original}\nslice:\t{slice}".format(
            original=str(self._group_),
            slice=self._slice_
        )

    def _load_(self, key):
        value = self._group_[key]
        if not isinstance(value, H5Group) and hasattr(self, '__getitem__'):
            value = value[self._slice_]
            self.__dict__[key] = value  # only cache sliced dataset
        return value


class H5Attrs(H5Group):
    '''Wrap of hdf5 attrs for quick access.
    '''

    def __str__(self):
        return "\n".join(
            "%s:\t%s" % (key, getattr(self, key)) for key in dir(self)
        )


class H5File(H5Group):
    '''Wrap of hdf5 file for quick access.
    '''
    pass
