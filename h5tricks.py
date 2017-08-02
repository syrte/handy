"""
h5py tricks
"""
import h5py
import pickle
import numpy as np
from sklearn.neighbors import KDTree


__all__ = ["save_vlen_array", "KDTreeH5"]


def save_vlen_array(group, name, array_list):
    """Equivalent to group[name] = array_list
    group : h5py.Group
    name : str
    array_list : array(list) of array
        Elements in array_list must have the same dtype!
    """
    array_list = np.asarray(array_list)
    shape = array_list.shape
    dtype = h5py.special_dtype(vlen=array_list[0].dtype)

    dset = group.create_dataset(name, shape, dtype=dtype)
    for i, v in np.ndenumerate(array_list):
        dset[i] = v


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
