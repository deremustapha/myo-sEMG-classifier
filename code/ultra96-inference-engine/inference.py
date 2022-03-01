# Importing basic python libraries

import os 
import time
import numpy as np
import math
import threading
import sys
import time
# Importing Myoware libraries 
from queue import Queue
from serial import Serial
from acquire_data import *
from filter import *
from normalization import *
from pyomyo import Myo, emg_mode
from pynq_dpu import DpuOverlay

com_port = '/dev/ttyUSB0'  # Myoband dongle was connected to Ultra96-v2 on this port
arduino = Serial(com_port, 115200, timeout=2)

# Importing DPU Overlay blocks and .xmodel
overlay = DpuOverlay("dpu.bit")  # load the DPU
overlay.load_model("model.xmodel") # load the complied model

def write_prediction(x):  
    
    X = str(x)
    arduino.write(bytes(X, 'utf-8'))
    
def calculate_softmax(data):
    result = np.exp(data)
    return result


while True:
    
    data = get_data()
    data = data[0:8,1:53] # Get the recent 52 samples
    
    data = artifact_removal(data)
    # data = normalize(data)
    data = np.float32(data) # Convert to float 
    data = np.expand_dims(data, axis=2) # Expand dimensions to (8, 52, 1)
        
    dpu = overlay.runner

    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()

    shapeIn = tuple(inputTensors[0].dims)
    shapeOut = tuple(outputTensors[0].dims)
    outputSize = int(outputTensors[0].get_data_size() / shapeIn[0])

    softmax = np.empty(outputSize)
    
    # Buffer defination to store input and output data. They will be reused during multiple runs.

    output_data = [np.empty(shapeOut, dtype=np.float32, order="C")]
    input_data = [np.empty(shapeIn, dtype=np.float32, order="C")]
    emg = input_data[0]
    
    # Store data in buffer
    emg[0, ...] = data
    
    job_id = dpu.execute_async(input_data, output_data)
    dpu.wait(job_id)
    temp = [j.reshape(1, outputSize) for j in output_data]
    softmax = calculate_softmax(temp[0][0])
    prediction = softmax.argmax()
    
    write_prediction(prediction)
    print(prediction)
    
    
del overlay
del dpu
