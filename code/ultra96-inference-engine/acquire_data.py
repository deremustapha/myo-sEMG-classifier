'''
The MIT License (MIT)
Copyright (c) 2020 PerlinWarp
Copyright (c) 2014 Danny Zhu


// Lab: Biomedical Information and Signal Lab (BMIS)
// Engineer: Dere Mustapha Deji
// Create Date: 2022/01/18  4:37:00 PM
// Design Name: Modified data acquisition
// Module Name: Myoband
// Tool Versions: Python 3.8
// Description: The aim is to continuously acquire data vai the myoband for inference on ultra96
// Dependencies: Pyomyo https://github.com/PerlinWarp/pyomyo
// Revision: 1
// Additional Comments: Credit to Peter Walkington
//
//////////////////////////////////////////////////////////////////////////////////
'''

import time
import numpy as np
from pyomyo import Myo, emg_mode


def get_data(mode=emg_mode.RAW, seconds=0.297):
    collect = True

    # ------------ Myo Setup ---------------
    m = Myo(mode=mode)
    m.connect()

    myo_data = []

    def add_to_queue(emg, movement):
        myo_data.append(emg)

    m.add_emg_handler(add_to_queue)

    # print("Data Worker started to collect")
    # Start collecing data.
    start_time = time.time()

    while collect:
        if time.time() - start_time < seconds:
            m.run()  # receive packet via bluetooth
        else:
            collect = False
            myo_data_array = np.array(myo_data).transpose()

    return myo_data_array




