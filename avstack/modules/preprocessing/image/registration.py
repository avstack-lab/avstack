from typing import Tuple, Union

import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors


class ImageRegistration:
    def __init__(self, max_size: int) -> None:
        """Base class for image registration
        
        Uses a python list for the buffers
        """
        self.max_size = max_size
        self._init_buffers()

    def _init_buffers(self):
        """Buffers store the data
        
        Transforms store the 3x3 homogeneous affine transformations
        """
        self.filled = False
        self._i_next = 0
        self._populated = [False] * self.max_size
        self._buffer_raw = [None] * self.max_size
        self._buffer_aligned = [None] * self.max_size
        self._transforms = np.dstack([np.eye(3, dtype=float)] * self.max_size)

    def append_and_register(self, data: np.ndarray) -> None:
        """Add and register frames
        
        Follows the full registration procedure:
            1. adds data to the buffer
            2. gets the transform from last to current
            3. applies transform to all prior frames
        """
        self.append(data)
        if self.filled or (self._i_next > 2):
            T_last_to_current = self._register_pair(
                fixed=data, moving=self._buffer_raw[self._i_next-1 % self.max_size]
            )
            self.apply(T_last_to_current)

    def append(self, data: np.ndarray) -> None:
        """Adds new data to the buffers and resets transforms"""
        if not self.filled:
            self._buffer_raw.append(data)
        else:
            self._buffer_raw[self._i_next] = data
        if not self.filled:
            self._populated[self._i_next] = True
        self._transforms[:,:,self._i_next] = np.zeros((3,3), dtype=float)
        self.filled = self.filled or (self._i_next == self.max_size-1)
        self._i_next = (self._i_next + 1) % self.max_size

    def apply(self, transform: np.ndarray) -> None:
        """Applies transformation matrix to all elements in the aligned"""
        for i, (populated, data_aligned) in enumerate(zip(self._populated, self._buffer_aligned)):
            if not populated:
                break
            else:
                self._buffer_aligned[i] = self._apply_single(data_aligned, transform)
                self._transforms[:,:,i] = transform @ self._transforms[:,:,i]  # integrate the transform


class IterativeClosestPointBasedRegistration(ImageRegistration):
    def __init__(self, max_size: int=10) -> None:
        """Initialize a circular buffer for image registration"""
        super().__init__(max_size=max_size)


    @staticmethod
    def _apply_single(data, transform):
        """Apply transformation to matrix of points"""
        pass

    @staticmethod
    def _register_pair(fixed, moving, n_iterations=10):
        """Get the transform from moving to fixed"""
        T = icp(moving, fixed, no_iterations=n_iterations)
        return T


# class IntensityBasedRegistration(ImageRegistration):
#     def __init__(self, max_size: int=10) -> None:
#         super().__init__(max_size=max_size)

#     @staticmethod
#     def _apply_single(data, transform):
#         """Apply transformation to an image"""
#         pass

#     @staticmethod
#     def _register_pair(fixed, moving):
#         """Get the transform from moving to fixed"""
#         pass


def icp(a, b, init_pose=(0,0,0), no_iterations = 13):
    """The Iterative Closest Point estimator.
    
    derived from: https://stackoverflow.com/questions/20120384/iterative-closest-point-icp-implementation-on-python

    Takes two cloudpoints a[x,y], b[x,y], an initial estimation of
    their relative pose and the number of iterations
    Returns the affine transform that transforms
    the cloudpoint a to the cloudpoint b.
    Note:
        (1) This method works for cloudpoints with minor
        transformations. Thus, the result depents greatly on
        the initial pose estimation.
        (2) A large number of iterations does not necessarily
        ensure convergence. Contrarily, most of the time it
        produces worse results.
    """


    src = np.array([a.T], copy=True).astype(np.float32)
    dst = np.array([b.T], copy=True).astype(np.float32)

    #Initialise with the initial pose estimation
    Tr = np.array([[np.cos(init_pose[2]),-np.sin(init_pose[2]),init_pose[0]],
                   [np.sin(init_pose[2]), np.cos(init_pose[2]),init_pose[1]],
                   [0,                    0,                   1          ]])

    src = cv2.transform(src, Tr[0:2])

    for i in range(no_iterations):
        #Find the nearest neighbours between the current source and the
        #destination cloudpoint
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto',
                                warn_on_equidistant=False).fit(dst[0])
        distances, indices = nbrs.kneighbors(src[0])

        #Compute the transformation between the current source
        #and destination cloudpoint
        T = cv2.estimateRigidTransform(src, dst[0, indices.T], False)
        #Transform the previous source and update the
        #current source cloudpoint
        src = cv2.transform(src, T)
        #Save the transformation from the actual source cloudpoint
        #to the destination
        Tr = np.dot(Tr, np.vstack((T,[0,0,1])))
    return Tr[0:2]


def main():
    pass


if __name__ == "__main__":
    main()