import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, sosfreqz, lfilter, iirnotch, filtfilt


def get_time_domain_plot(data):
    plt.clf()
    plt.title("Time Domain")
    plt.plot(data)


def get_frequency_domain_plot(data, fs):
    fft = np.fft.rfft(data)
    freq = np.fft.rfftfreq(len(data), 1 / fs)
    plt.clf()
    plt.title("Frequency Domain")
    plt.plot(freq, abs(fft))
    plt.xlabel("Frequency")
    plt.ylabel("Number of samples")


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='bandpass', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut=5, highcut=35, fs=250, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y


def get_spectogram(data, fs, nfft, overlap):
    plt.title('Spectogram')
    plt.specgram(data, Fs=fs, NFFT=nfft, noverlap=overlap)
    plt.xlabel("Time Sec")
    plt.ylabel("Frequency Hz")


def remove_dc_offset(data, fs=250, offset=.5, order=3):
    nyq = .5 * fs
    normal_cutoff = offset / nyq
    b, a = butter(N=order, Wn=normal_cutoff, btype='highpass', analog=False)
    fil_data = lfilter(b, a, data)
    return fil_data


def lowpass_filter(data, fs=250, offset=.5, order=3):
    nyq = .5 * fs
    normal_cutoff = offset / nyq
    b, a = butter(N=order, Wn=normal_cutoff, btype='lowpass', analog=False)
    fil_data = lfilter(b, a, data)
    return fil_data


def mains_removal(data, fs=250, notch_freq=60.0, quality_factor=30.0):
    b, a = iirnotch(notch_freq, quality_factor, fs)
    fil_data = filtfilt(b, a, data, padlen=len(data) -1)
    return fil_data

# %%
