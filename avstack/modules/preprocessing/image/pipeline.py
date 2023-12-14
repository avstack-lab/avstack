import numpy as np

import estimation
import registration


class Pipeline:
    def __init__(self, register, estimator):
        self.register = register
        self.estimator = estimator

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Call the pipeline
        
        Input is new image or array of points
        Output is filtered image or points
        """
        buffer = self.register(data)
        image_out = self.estimator(buffer)
        return image_out
        

class PointBasedPipeline(Pipeline):
    def __init__(self, max_size: int=10):
        super().__init__(
            register=registration.IterativeClosestPointBasedRegistration(
                max_size=max_size
            ),
            estimator=estimation.PointBasedMeanFilter(),
        )
        

class ImageBasedPipeline(Pipeline):
    def __call__(self, data: np.ndarray) -> np.ndarray:
        pass