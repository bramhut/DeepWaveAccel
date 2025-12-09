import numpy as np
import Preprocess as pp
from Goertzel import goertzel, goertzel_multi_bin_with_logging
import CrossCor as cc
import Backproj as bp
import Laplacian as lap
import Deblur as deblur
import Plotting as pl
from Logger import SignalLogger

# Setup Inputs: 
# N_ch: number of channels
# N_px: number of pixels
# R: pixel coordinates (3, N_px)
# Pf_file: path to the parameter file

# Functions:
# One function to run inference (argument: Wav_file: path to the wav file)
# One function to run inference and compare to the reference implementation (argument: DeepWave reference images, saved in npz file). This function should return the PSNR


import os
import scipy.io.wavfile as wav

class DeepWaveAccel:
    def __init__(self, N_ch, N_px, ff, R, Pf_file):
        """
        Args:
            N_ch (int): Number of channels (microphones)
            N_px (int): Number of pixels (directions)
            ff (float): Frequency of interest (Hz)
            R (np.ndarray): Pixel coordinates, shape (3, N_px)
            Pf_file (str): Path to the parameter file (.npz)
        """
        self.N_ch = N_ch
        self.N_px = N_px
        self.ff = ff  # Frequency of interest
        self.R = R
        self.Pf_file = Pf_file
        

        # Load model parameters
        Pf = np.load(Pf_file)
        self.K = int(Pf['K'])
        self.N_layer = Pf['N_layer']
        p_opt = Pf['p_opt'][np.argmin(Pf['v_loss'])]
        param = pp.Parameter(N_ch, N_px, self.K)
        self.theta, self.B, self.tau = param.decode(p_opt)
        
        # Deblurring
        self.laplacian, rho = lap.laplacian_scipy(self.R)
        self.laplacian = self.laplacian.todia()
        self.laplacian_banded = lap.sparsify_band_symmetric(self.laplacian, threshold=1e-4)

    def run_inference(self, wav_file, num_iter_power=10, norm_floor=1, energy_norm=False, logger=None):
        """
        Run inference on a wave file.

        Args:
            wav_file (str): Path to the wave file.
            num_iter_power (int): Number of iterations for power iteration in cross-correlation.
            logger (SignalLogger or None): Logger to collect min/max/mean/std info.

        Returns:
            deblurred_images (np.ndarray): Deblurred intensity maps [frames, pixels]
            psnr (np.ndarray): PSNR values per frame (if reference available)
        """
             
        
        fs, Draw = wav.read(os.path.expanduser(wav_file))
        Draw = Draw / 2048.0  # Normalize 16-bit PCM
        
        # Time-frequency conversion
        nffloat = 10 * fs / self.ff
        nf = 200
        fr = fs / nf
        bin = round(self.ff / fr)
        factual = bin * fr
        
        bins = [bin, bin - 1] # DeepWave reference bins
        dft_per_bin, _ = goertzel_multi_bin_with_logging(Draw, bins, nf, 0.0, True, False, logger=logger)
        dft = np.sum(dft_per_bin, axis=2)
        # dft, step = goertzel(Draw, bin, nf, 0.0, True, False)
        # dft += goertzel(Draw, bin-1, nf, 0.0, True, False)[0]
        
        dft_abs = np.abs(dft)

        # Cross-correlation
        R_all, norms = cc.cross_correlation(dft, group_size=9, floor_value=norm_floor, num_iter_power=num_iter_power, energy_norm=energy_norm)
        
        N_frames = R_all.shape[0]

        # Backprojection
        bpp = np.zeros((N_frames, self.B.shape[1]), dtype=np.float64)
        for i in range(N_frames):
            bpp[i] = bp.backproject_py_opt(R_all[i], self.B, self.tau)

        deblurred_images = np.zeros((N_frames, self.N_px))
        for i in range(N_frames):
            x = np.zeros((self.N_px,))
            for layer in range(self.N_layer):
                y = deblur.chebyshev_conv(self.laplacian, x, self.theta)
                deblurred_images[i] = deblur.retanh_activation(y + bpp[i])
                x = deblurred_images[i]
                
        # Save signal logger data
        if logger is not None:
            logger.log('Draw', Draw)
            logger.log('dft', dft)
            logger.log('R_all', R_all)
            logger.log('bpp', bpp)
            logger.log('deblurred_images', deblurred_images)

        return deblurred_images, bpp, norms

    def compare_to_reference_global_norm(self, deblurred_images, reference_images):
        """
        Compare deblurred images to DeepWave reference and compute PSNR.

        Args:
            deblurred_images (np.ndarray): Output from run_inference
            reference_images (np.ndarray): Reference images from DeepWave (shape: [frames, pixels])

        Returns:
            psnr (np.ndarray): PSNR values per frame
        """
        peak = np.max(reference_images)
        mse = np.mean((deblurred_images - reference_images) ** 2, axis=1)
        psnr = 10 * np.log10((peak ** 2) / mse)
        return psnr
    
    def compare_to_reference_indep_norm(self, deblurred_images, reference_images, normalize=True):
        """
        Compare deblurred images to DeepWave reference and compute PSNR.

        Args:
            deblurred_images (np.ndarray): Output from run_inference (shape: [frames, pixels])
            reference_images (np.ndarray): Reference images from DeepWave (shape: [frames, pixels])
            normalize (bool): Whether to normalize each frame to have max=1

        Returns:
            psnr (np.ndarray): PSNR values per frame (NaN for invalid/zero frames)
        """
        deblurred_max = np.max(deblurred_images, axis=1, keepdims=True)
        reference_max = np.max(reference_images, axis=1, keepdims=True)
        nonzero_mask = (deblurred_max.squeeze() > 0) & (reference_max.squeeze() > 0)

        deblurred_norm = np.copy(deblurred_images)
        reference_norm = np.copy(reference_images)

        if normalize:
            # Only normalize nonzero frames
            deblurred_norm[nonzero_mask] = deblurred_images[nonzero_mask] / deblurred_max[nonzero_mask]
            reference_norm[nonzero_mask] = reference_images[nonzero_mask] / reference_max[nonzero_mask]
            peak = 1.0
        else:
            peak = np.max(reference_images)

        mse = np.mean((deblurred_norm - reference_norm) ** 2, axis=1)
        psnr = np.full_like(mse, np.nan)
        # Only compute PSNR for nonzero frames
        psnr[nonzero_mask] = 10 * np.log10((peak ** 2) / mse[nonzero_mask])
        # For zero frames, PSNR is NaN
        return psnr


    def plot_image_3D(self, deblurred_images, frame_idx=0, source_lons=None, source_labels=None):
        """
        Plot a single frame using the spherical mesh.

        Args:
            deblurred_images (np.ndarray): Output from run_inference
            frame_idx (int): Frame index to plot
        """
        source_lats = [0] * len(source_lons) if source_lons is not None else None
        fig = pl.draw_spherical_mesh(deblurred_images[frame_idx], self.R, source_lons, source_lats, source_labels)
        fig.show()
        
    def save_images(self, output_file, images, psnr):
        """
        Save deblurred images to a file.

        Args:
            deblurred_images (np.ndarray): Output from run_inference
            psnr (np.ndarray): PSNR values per frame
            output_file (str): Path to save the images
        """
        np.savez(output_file, intensity=images, psnr=psnr)
        
    def analyse_signal_range(self):
        """ 
        Analyze the range of all signals and parameters, and return a summary.
        Returns:
            summary (dict): Summary of signal ranges and parameters.
        """
        