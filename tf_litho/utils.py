"""
Utility functions for TensorFlow lithography simulation.
Replaces OpenCV dependencies with scikit-image and TensorFlow built-ins.
"""
import tensorflow as tf
import numpy as np
from skimage.transform import resize


def get_mask_fft(mask):
    """Compute FFT of mask tensor."""
    return tf.signal.fftshift(tf.signal.fft2d(tf.cast(mask, tf.complex64)))


def interpolate_aerial_image(image, pixel, mode="bilinear"):
    """Interpolate aerial image using scikit-image instead of OpenCV."""
    target_shape = (image.shape[0] * pixel, image.shape[1] * pixel)
    # Convert to numpy for scikit-image processing
    image_np = image.numpy() if hasattr(image, 'numpy') else np.array(image)
    interpolated = resize(image_np, target_shape, order=1 if mode == "bilinear" else 0, 
                         anti_aliasing=True, preserve_range=True)
    return tf.constant(interpolated, dtype=image.dtype)


def interpolate_aerial_image_batch(images, pixel, mode="bilinear"):
    """Batch interpolation for aerial images."""
    batch_size = images.shape[0]
    results = []
    for i in range(batch_size):
        interpolated = interpolate_aerial_image(images[i], pixel, mode)
        results.append(interpolated)
    return tf.stack(results)


class BBox:
    """Bounding box utility class."""
    def __init__(self, min_point, max_point):
        self.min_point = min_point
        self.max_point = max_point
    
    def width(self):
        return self.max_point.x - self.min_point.x
    
    def height(self):
        return self.max_point.y - self.min_point.y


class Point:
    """Point utility class."""
    def __init__(self, x, y):
        self.x = x
        self.y = y