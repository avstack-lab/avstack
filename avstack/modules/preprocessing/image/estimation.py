import numpy as np


class PointBasedMeanFilter:
    def __call__(self, buffer, image_size):
        """Run the background estimation
        
        Each element in the bufer is a list of points.
        Convert the points to images and take the mean.
        """
        images = np.zeros((image_size[0], image_size[1], len(buffer)))
        for i, points in enumerate(buffer):
            image = np.zeros(image_size, dtype=float)
            image[points] = 1.0
            images[i].append(image)
        return ImageBasedMeanFilter.filter(images)


class ImageBasedMeanFilter:
    def __call__(self, buffer):
        return self.filter(buffer)

    @staticmethod
    def filter(buffer):
        return np.mean(np.concatenate(buffer, axis=2), axis=2)