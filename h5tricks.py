"""
h5py tricks
"""
import h5py
import pickle
import numpy as np
from sklearn.neighbors import KDTree
from itertools import product

__all__ = ["save_vlen_array", "KDTreeH5"]


def save_vlen_array(group, name, array_list):
    """Equivalent to group[name] = array_list

    Parameters
    ----------
    group : h5py.Group
    name : str
    array_list : array/list of arrays
        Elements in array_list must have the same dtype!

    Examples
    --------
    import h5py
    from numpy import array

    # a can be list of array
    a = [array([0]), array([0, 1]), array([0, 1, 2])]

    # or array of array
    a = array([[array([0]), array([0, 1]), array([0, 1, 2])],
               [array([0, 1, 2]), array([0, 1]), array([0])]], dtype=object)

    # or created as below
    a = np.empty((2, 3), 'O')
    for i in range(2):
        for j in range(3):
            a[i, j] = np.arange(i * 3 + j)

    with h5py.File('test_tmp.h5') as f:
        save_vlen_array(f, 'a', a)
    """
    array_list = np.asarray(array_list)
    if array_list.dtype.kind == 'O':
        shape = array_list.shape
    else:
        shape = array_list.shape[:-1]
    ix_0 = tuple(0 for _ in shape)  # the index of the first array element in array_list

    dtype = h5py.special_dtype(vlen=array_list[ix_0].dtype)
    # dtype = h5py.vlen_dtype(array_list[ix_0].dtype)
    dset = group.create_dataset(name, shape, dtype=dtype)
    try:
        for ix in product(*map(range, shape)):
            dset[ix] = array_list[ix]
    except Exception:
        del group[name]
        raise


# length of KDTree.__getstate__()
# Check the source code of KDTree at
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/neighbors/binary_tree.pxi
KDTREE_STATE_LEN = 12


class KDTreeH5(KDTree):
    def dump(self, file):
        """
        file: str or HDF group

        Examples
        --------
        # dump KDTree object
        KDTreeH5.dump(tree, filepath)
        """
        if not isinstance(file, h5py.Group):
            file = h5py.File(file)

        state = list(self.__getstate__())
        assert len(state) == KDTREE_STATE_LEN

        # convert dist_metric to string for hdf5 storage
        state[-1] = pickle.dumps(state[-1])
        for i, v in enumerate(state):
            file[str(i)] = v

    @classmethod
    def load(cls, file):
        """
        file: str or HDF group
        """
        if not isinstance(file, h5py.Group):
            file = h5py.File(file, 'r')

        state = [None] * len(file)
        assert len(state) == KDTREE_STATE_LEN

        for i, _ in enumerate(state):
            state[i] = file[str(i)].value
        # recover dist_metric from string
        state[-1] = pickle.loads(state[-1])

        obj = cls.__new__(cls)
        obj.__setstate__(state)
        return obj
