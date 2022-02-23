"""
// LAB: Biomedical Information and Signal Lab 
// Engineer: Dere Mustapha Deji
// Created on Moday Feb 21 2022 15:49:00 
// Data: MyoData 
// Source: https://github.com/UlysseCoteAllard/MyoArmbandDataset 
// Design Name: Gesture Classification in MyoUP_dataset
// Project Name: Realtime gesture classfication for Rehabilitation Purposes
// Target Devices: Ultra96-v2, Myoband
// Credit: Ulysse Cote-Allard et.al
// 
"""

import numpy as np
from scipy import signal
import math
from sklearn.utils import shuffle


def vec_to_matrix(vector_data, no_channel=8):
    matrix_shape = int(len(vector_data) / no_channel)
    matrix = np.zeros([matrix_shape, no_channel])
    data_range = [list(range(0, int(len(vector_data)), 8))]

    for idx, data_id in enumerate(data_range[0]):
        start = data_id
        end = data_id + no_channel
        matrix[idx] = vector_data[start:end]

    return matrix


def window_without_overlap(data, sampling_frequency=200, window_time=260, no_channel=8):
    samples = int(sampling_frequency * (window_time / 1000))
    size = math.floor(len(data) / samples)
    matrix = np.zeros([size, no_channel, samples])
    data = data.transpose()
    data_range = [list(range(0, data.shape[1], samples))]

    for idx, data_id in enumerate(data_range[0]):

        start = data_id
        end = data_id + samples

        if end <= data.shape[1]:
            matrix[idx] = data[0:no_channel, start:end]

    return matrix


def window_with_overlap(data, sampling_frequency=200, window_time=260, overlap=235, no_channel=8):
    samples = int(sampling_frequency * (window_time / 1000))
    num_overlap_samples = int(sampling_frequency * (overlap / 1000))
    data = data.transpose()
    idx = [i for i in range(samples, data.shape[1], num_overlap_samples)]

    matrix = np.zeros([len(idx), no_channel, samples])
    filtered_matrix = np.zeros([len(idx), no_channel, samples])

    for i, end in enumerate(idx):
        start = end - samples

        if end <= data.shape[1]:
            matrix[i] = data[0:no_channel, start:end]
            filtered_matrix[i] = artifact_removal(data[0:no_channel, start:end])

    return matrix, filtered_matrix


def artifact_removal(data, cutoff=2, sampling_frequency=200, order=3):
    nyq = .5 * sampling_frequency
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = signal.lfilter(b, a, data)

    return filtered_data


def data_shuffle(data, label):
    X, y = shuffle(data, label, random_state=42)

    return X, y


def data_normalization(data):
    # This is not the write way to do it.
    # I could have used tensorflow Transform or Apache beam 
    # But for some werid reasons i couldnt make it to work.
    # So I am doing it the crude way

    for idx, info in enumerate(data):
        mean = np.mean(info, axis=1)
        std = np.std(info, axis=1)
        norm = ((info.transpose() - mean) / std).transpose()

        data[idx] = norm

    return data


def get_data(abs_path, n_male, n_female, n_exercise):
    total_male_subject = n_male
    total_female_subject = n_female
    total_exercise = n_exercise

    full_buffer = []

    for subject in range(total_male_subject):

        buffer = []
        label_counter = 0

        for exercise in range(total_exercise):

            per_subject = abs_path + "/Male" + str(subject) + "/training0" + "/classe_" + str(exercise) + ".dat"
            read_data = np.fromfile(per_subject, dtype=np.int16)
            data_array = np.array(read_data, dtype=np.float32)

            data_matrix = vec_to_matrix(data_array)
            unfiltred_data, filtered_data = window_with_overlap(data_matrix)

            if label_counter == 7:
                label_counter = 0

            label = np.zeros(filtered_data.shape[0]) + label_counter
            label = label.reshape(-1, 1)

            if not buffer:
                data_stack = filtered_data
                label_stack = label
                buffer = 1

            else:
                data_stack = np.row_stack((data_stack, filtered_data))
                label_stack = np.vstack((label_stack, label))

            label_counter += 1

        if not full_buffer:
            full_data_stack = data_stack
            full_label_stack = label_stack
            full_buffer = 1

        else:
            full_data_stack = np.row_stack((full_data_stack, data_stack))
            full_label_stack = np.vstack((full_label_stack, label_stack))

    X, y = data_shuffle(full_data_stack, full_label_stack)
    return X, y