import tkinter as tk
from experiment_one import *
from playsound import playsound
import os
from PIL import ImageTk, Image


def acquire_data():
    playsound("/home/darcula-venom/Documents/ExperimentalDesign/MyoBand_sEMG_Acquisition/sound.wav")

    seconds = float(acquisition_time.get()) + 0.2
    subject_index = subject_no.get()
    experiment_index = clicked.get().split()[1]
    repetition_index = rep_no.get()

    csv_file_path = "/home/darcula-venom/Documents/ExperimentalDesign/BMIS_EMG_DATA/data/csv_data/subject_" + subject_index
    mat_file_path = "/home/darcula-venom/Documents/ExperimentalDesign/BMIS_EMG_DATA/data/mat_data/subject_" + subject_index

    print(csv_file_path)
    print(mat_file_path)
    if not os.path.exists(csv_file_path):
        os.makedirs(csv_file_path)
        os.makedirs(mat_file_path)

    csv_file = csv_file_path + '/S{}_R{}_G{}.csv'.format(subject_index, repetition_index, experiment_index)
    mat_file = mat_file_path + '/S{}_R{}_G{}'.format(subject_index, repetition_index, experiment_index)

    mode = emg_mode.RAW
    p = multiprocessing.Process(target=data_worker(mode, seconds, csv_file, mat_file))
    p.start()


window = tk.Tk()

experiment_message = tk.Label(text="EMG EXPERIMENT")
experiment_message.pack()

acquisition_time = tk.Entry(window, width=50, borderwidth=5)
acquisition_time.pack()

subject_no = tk.Entry(window, width=50, borderwidth=10)
subject_no.pack()

rep_no = tk.Entry(window, width=50, borderwidth=15)
rep_no.pack()

clicked = tk.StringVar()
clicked.set("Gesture 1")
drop = tk.OptionMenu(window, clicked, "Gesture 1", "Gesture 2", "Gesture 3", "Gesture 4", "Gesture 5",
                     "Gesture 6", "Gesture 7")
drop.pack()

image_list = ["exp{}.png".format(i) for i in range(1, 8)]
picture_path = "/home/darcula-venom/Documents/ExperimentalDesign/MyoBand_sEMG_Acquisition/pictures/{}".format(image_list[1])
image = ImageTk.PhotoImage(Image.open(picture_path))
image_label = tk.Label(image=image)
image_label.pack()

start_button = tk.Button(window, text="Start Experiment", command=acquire_data, bg="green")
start_button.pack()

window.mainloop()
