import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import wfdb.processing
from scipy import signal
import pywt
import os


def high_pass_filter(signals, fs):
    # design FIR filter
    fc = 0.5  # cuttoff frequency in Hz
    desired = (0, 0, 1, 1)
    bands = (0, fc, fc, fs / 2)
    filter_fir = signal.firls(
        numtaps=171, bands=bands, desired=desired, fs=fs
    ) * signal.kaiser(
        171, 5
    )  # multiply with a Kaiser window for less ripples in the response
    signals = signals - np.mean(signals)
    filtered_signals = signal.filtfilt(b=filter_fir, a=1, x=signals)
    return filtered_signals


def morlet_wavelet(signals, fs):
    # Morlet wavelet

    dt = 1 / fs  # sampling period
    scales = np.logspace(
        np.log10(10), np.log10(500), num=10
    )  # this will give frequencies 1Hz to 50Hz in log scale
    frequencies = pywt.scale2frequency("cmor1.5-1.0", scales) / dt
    # print(frequencies)

    cwtmatr, freqs = pywt.cwt(
        signals, scales, "cmor1.5-1.0", sampling_period=dt, method="fft",
    )
    return cwtmatr

