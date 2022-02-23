import numpy as np

def normalize(data):
    
    # This is not the write way to do it.
    # I could have used tensorflow Transform or Apache beam 
    # But for some werid reasons i couldnt make it to work.
    # So I am doing it the crude way
    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)
    norm = ((data.transpose() - mean) / std).transpose()

    return norm
