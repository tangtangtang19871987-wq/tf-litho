"""
TCC (Transmission Cross Coefficient) generation for Hopkins model.
Ported from TorchLitho-Lite with OpenCV and sklearn dependencies removed.
"""
import pickle
from typing import Union, List, Tuple
import numpy as np
from scipy.sparse.linalg import svds
from skimage.transform import resize


SINMAX = 0.9375
MAX_TCC_SIZE = 64


def get_freqs(pixel: int, canvas: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate frequency grids."""
    size = round(canvas / pixel)
    basic = np.fft.fftshift(np.fft.fftfreq(size, d=pixel))
    freq_x = np.tile(basic.reshape(1, -1), (size, 1))
    freq_y = np.tile(basic.reshape(-1, 1), (1, size))
    assert freq_x.shape[0] == size and freq_x.shape[1] == size
    assert freq_y.shape[0] == size and freq_y.shape[1] == size
    return freq_x, freq_y


def src_point(pixel: int, canvas: int) -> np.ndarray:
    """Generate source point (delta function at origin)."""
    freq_x, freq_y = get_freqs(pixel, canvas)
    result = (freq_x == 0) * (freq_y == 0)
    return result.astype(np.float64)


def func_pupil(pixel: int, canvas: int, na: float, lam: int, 
               defocus: Union[None, int] = None, refract: Union[None, float] = None) -> np.ndarray:
    """Generate pupil function with optional defocus."""
    limit = na / lam
    freq_x, freq_y = get_freqs(pixel, canvas)
    result = np.sqrt(freq_x**2 + freq_y**2) < limit
    result = result.astype(np.float64)
    
    if defocus is not None:
        assert refract is not None
        mask = result > 0
        opd = defocus * (refract - np.sqrt(refract**2 - lam**2 * ((freq_x * mask)**2 + (freq_y * mask)**2)))
        shift = np.exp(1j * (2 * np.pi / lam) * opd)
        result = result * shift
    
    return result


def tcc(src: np.ndarray, pupil: np.ndarray, pixel: int, canvas: int, thresh: float = 1.0e-6) -> Tuple[List[np.ndarray], List[float]]:
    """Compute Transmission Cross Coefficient using SVD."""
    size = round(canvas / pixel)
    pupil_fft = np.fft.fftshift(np.fft.fft2(pupil))  # h
    pupil_star = np.conj(pupil_fft)  # h*
    src_fft = np.fft.fftshift(np.fft.fft2(src / np.sum(src)))  # J
    
    # Create TCC matrix
    w = np.zeros(pupil_star.shape + pupil_star.shape, dtype=np.complex64)
    for idx in range(pupil_star.shape[0]):
        for jdx in range(pupil_star.shape[1]):
            src_shifted = np.roll(src_fft, shift=(idx, jdx), axis=(0, 1))
            src_shifted = np.flip(src_shifted, axis=(0, 1))
            w[idx, jdx] = src_shifted * pupil_fft[idx, jdx] * pupil_star / (np.prod(pupil.shape) * np.prod(src.shape))
    
    # Reshape for SVD
    size_all = np.prod(pupil_star.shape)
    w_reshaped = w.reshape(size_all, size_all)
    
    # Use SciPy SVD instead of sklearn randomized_svd
    n_components = min(64, size_all - 1)
    try:
        mat_u, mat_s, mat_vt = svds(w_reshaped, k=n_components)
        # svds returns singular values in ascending order, need to reverse
        mat_u = mat_u[:, ::-1]
        mat_s = mat_s[::-1]
        mat_vt = mat_vt[::-1, :]
    except Exception as e:
        print(f"SVD failed, using full SVD: {e}")
        mat_u, mat_s, mat_vt = np.linalg.svd(w_reshaped)
        mat_u = mat_u[:, :n_components]
        mat_s = mat_s[:n_components]
        mat_vt = mat_vt[:n_components, :]
    
    # Extract significant components
    phis = []
    weights = []
    for idx, weight in enumerate(mat_s):
        if thresh is not None and weight >= thresh:
            phi = mat_u[:, idx].reshape(size, size) * (size * size)
            phis.append(phi)
            weights.append(weight)
    
    return phis, weights


def gen_tcc(pixel: int, 
            canvas: int, 
            na: float, 
            wavelength: int, 
            defocus: Union[None, List[int]] = None, 
            thresh: float = 1.0e-6) -> Union[Tuple[List[np.ndarray], List[float]], Tuple[List[List[np.ndarray]], List[List[float]]]]:
    """Generate TCC for Hopkins model."""
    refract = na / SINMAX
    size = canvas // pixel
    
    if defocus is not None:
        phis, weights = [], []
        for d in defocus:
            if size <= MAX_TCC_SIZE:
                pupil = func_pupil(pixel, canvas, na, wavelength, defocus=d, refract=refract)
                circ = src_point(pixel, canvas)
                _phis, _weights = tcc(circ, pupil, pixel, canvas, thresh=thresh)
            else:
                # Handle large canvas by downsampling
                tcc_pixel = pixel
                tcc_canvas = canvas
                resize_factor = 1
                padding_factor = 1
                
                while tcc_canvas // tcc_pixel > MAX_TCC_SIZE:
                    if tcc_canvas > 2048:
                        tcc_canvas //= 2
                        resize_factor *= 2
                    else:
                        tcc_pixel *= 2
                        padding_factor *= 2
                
                pupil = func_pupil(tcc_pixel, tcc_canvas, na, wavelength, refract=refract)
                circ = src_point(tcc_pixel, tcc_canvas)
                _phis, _weights = tcc(circ, pupil, tcc_pixel, tcc_canvas, thresh=thresh)
                tcc_size = tcc_canvas // tcc_pixel
                
                # Upsample results back to original size
                for idx in range(len(_phis)):
                    padded_size = tcc_size * padding_factor
                    ffted = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(_phis[idx])))
                    
                    # Use scikit-image for resizing instead of OpenCV
                    real_part = np.zeros((padded_size, padded_size))
                    imag_part = np.zeros((padded_size, padded_size))
                    begin = (padded_size - tcc_size) // 2
                    end = begin + tcc_size
                    real_part[begin:end, begin:end] = ffted.real
                    imag_part[begin:end, begin:end] = ffted.imag
                    ffted_combined = real_part + 1j * imag_part
                    
                    # Resize using scikit-image
                    real_resized = resize(ffted_combined.real, (size, size), 
                                        order=1, anti_aliasing=True, preserve_range=True)
                    imag_resized = resize(ffted_combined.imag, (size, size), 
                                        order=1, anti_aliasing=True, preserve_range=True)
                    ffted_resized = real_resized + 1j * imag_resized
                    
                    _phis[idx] = (padding_factor**2 * resize_factor**2 * 
                                np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(ffted_resized))))
            
            phis.append(_phis)
            weights.append(_weights)
        return phis, weights
    else:
        if size <= MAX_TCC_SIZE:
            pupil = func_pupil(pixel, canvas, na, wavelength, refract=refract)
            circ = src_point(pixel, canvas)
            phis, weights = tcc(circ, pupil, pixel, canvas, thresh=thresh)
        else:
            # Handle large canvas by downsampling
            tcc_pixel = pixel
            tcc_canvas = canvas
            resize_factor = 1
            padding_factor = 1
            
            while tcc_canvas // tcc_pixel > MAX_TCC_SIZE:
                if tcc_canvas > 2048:
                    tcc_canvas //= 2
                    resize_factor *= 2
                else:
                    tcc_pixel *= 2
                    padding_factor *= 2
            
            pupil = func_pupil(tcc_pixel, tcc_canvas, na, wavelength, refract=refract)
            circ = src_point(tcc_pixel, tcc_canvas)
            phis, weights = tcc(circ, pupil, tcc_pixel, tcc_canvas, thresh=thresh)
            tcc_size = tcc_canvas // tcc_pixel
            
            # Upsample results back to original size
            for idx in range(len(phis)):
                padded_size = tcc_size * padding_factor
                ffted = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(phis[idx])))
                
                real_part = np.zeros((padded_size, padded_size))
                imag_part = np.zeros((padded_size, padded_size))
                begin = (padded_size - tcc_size) // 2
                end = begin + tcc_size
                real_part[begin:end, begin:end] = ffted.real
                imag_part[begin:end, begin:end] = ffted.imag
                ffted_combined = real_part + 1j * imag_part
                
                real_resized = resize(ffted_combined.real, (size, size), 
                                    order=1, anti_aliasing=True, preserve_range=True)
                imag_resized = resize(ffted_combined.imag, (size, size), 
                                    order=1, anti_aliasing=True, preserve_range=True)
                ffted_resized = real_resized + 1j * imag_resized
                
                phis[idx] = (padding_factor**2 * resize_factor**2 * 
                            np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(ffted_resized))))
        
        return phis, weights


def read_tcc_from_disc(path: str) -> Tuple[List[np.ndarray], List[float]]:
    """Read precomputed TCC from disk."""
    with open(path, "rb") as fin:
        phis, weights = pickle.load(fin)
    return phis, weights


def write_tcc_to_disc(phis: List[np.ndarray], weights: List[float], path: str):
    """Write TCC to disk for reuse."""
    with open(path, "wb") as fout:
        pickle.dump((phis, weights), fout)