from scipy import signal

def artifact_removal(data, cutoff=2, sampling_frequency=200, order=3):
    
    nyq = .5 * sampling_frequency
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = signal.lfilter(b, a, data)

    return filtered_data