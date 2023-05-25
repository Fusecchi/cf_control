'''
Can plot EMG data in 2 different ways
change DRAW_LINES to try each.
Press Ctrl + C in the terminal to exit 
'''
import numpy
import pygame
import time
from pygame.locals import *
import multiprocessing
import keyboard
import numpy as np
import scipy.signal as sig
import tensorflow as tf

from pyomyo import Myo, emg_mode

# ------------ Myo Setup ---------------
q = multiprocessing.Queue()


# def signal_filter(emg_signal):
#     # Step 1: Filter the signal
#     # High-pass filter to remove baseline noise
#     fs = 1000  # Sampling frequency in Hz
#     cutoff_freq = 20  # Cutoff frequency for high-pass filter in Hz
#     b, a = sig.butter(4, cutoff_freq / (fs / 2), 'highpass')
#     emg_signal_filtered = sig.filtfilt(b, a, emg_signal)
#
#     # Low-pass filter to remove high-frequency noise
#     cutoff_freq = 500  # Cutoff frequency for low-pass filter in Hz
#     b, a = sig.butter(4, cutoff_freq / (fs / 2), 'lowpass')
#     emg_signal_filtered = sig.filtfilt(b, a, emg_signal_filtered)
#
#     # Step 2: Rectify the signal
#     emg_signal_rectified = np.abs(emg_signal_filtered)
#
#     # Step 3: Smooth the signal
#     window_size = 0.1  # Window size for smoothing in seconds
#     window_length = int(window_size * fs)
#     emg_signal_smoothed = sig.savgol_filter(emg_signal_rectified, window_length, 2)
#     return emg_signal_smoothed


def worker(q):
    m = Myo(mode=emg_mode.PREPROCESSED)
    # m = Myo(mode=emg_mode.RAW)
    # m = Myo(mode=emg_mode.FILTERED)

    m.connect()

    def add_to_queue(emg, movement):
        q.put(emg)

    m.add_emg_handler(add_to_queue)

    def print_battery(bat):
        print("Battery level:", bat)

    m.add_battery_handler(print_battery)

    # Orange logo and bar LEDs
    m.set_leds([128, 0, 0], [128, 0, 0])
    # Vibrate to know we connected okay
    m.vibrate(1)

    """worker function"""
    while True:
        m.run()
    print("Worker Stopped")


last_vals = None


def plot(scr, vals):
    DRAW_LINES = True
    # font = pygame.font.Font(None, 36)
    global last_vals
    if last_vals is None:
        last_vals = vals
        return

    D = 5
    scr.scroll(-D)
    scr.fill((0, 0, 0), (w - D, 0, w, h))
    for i, (u, v) in enumerate(zip(last_vals, vals)):
        if DRAW_LINES:
            pygame.draw.line(scr, (0, 255, 0),
                             (w - D, int(h / 9 * (i + 1 - u))),
                             (w, int(h / 9 * (i + 1 - v))))
            pygame.draw.line(scr, (255, 255, 255),
                             (w - D, int(h / 9 * (i + 1))),
                             (w, int(h / 9 * (i + 1))))

            # Get current time
            current_time = time.strftime("%H:%M:%S")

        # # Render time text
        # time_text = font.render(current_time, True, (0, 0, 0))
        # time_rect = time_text.get_rect(center=(screen_width / 2, screen_height / 2))
        #
        # # Draw time text on screen
        # screen.blit(time_text, time_rect)
        # else:
        #     c = int(255 * max(0, min(1, v)))
        #     scr.fill((c, c, c), (w - D, i * h / 8, D, (i + 1) * h / 8 - i * h / 8))

    pygame.display.flip()
    last_vals = vals


# -------- Main Program Loop -----------
if __name__ == "__main__":
    p = multiprocessing.Process(target=worker, args=(q,))
    p.start()

    w, h = 800, 600
    scr = pygame.display.set_mode((w, h))

    running = time.time()
    model = tf.keras.models.load_model("/home/pc/Downloads/Ovass30.h5", compile=True)
    x = 0
    buffer = list()
    try:
        while True:
            # Handle pygame events to keep the window responding
            pygame.event.pump()
            # Get the emg data and plot it
            while not (q.empty()):
                end_time = time.time()
                emgnew = list()
                rec = list(q.get())
                # make an empty list

                for row in range(25):  # wait until 8 data
                    emg = list(q.get())  # subscribe data
                    plot(scr, [e / 500. for e in emg])  # plot it
                    # A for loop for column entries
                    emgnew.append(emg)  # add to our empty list
                new = tf.convert_to_tensor([emgnew]) # convert to tensor
                shape = tf.shape(new) # see the shape of our matrix
                # print(shape)
                prediction = model.predict_on_batch(new) #predict the list
                # result_index = np.argmax(prediction)
                result = np.around(prediction, decimals=2)
                # print(str(result_index) + "Prediction :"+ str(result))
                print(buffer)
                if(np.average(emg) <= 40):
                    print("idle")
                    buffer.clear()
                else:
                    if x <= 5:
                        result_index = np.argmax(prediction)
                        buffer.append(result_index)
                        x+=1
                        if x == 6:
                            highest = max(set(buffer), key = buffer.count)
                            print(highest)
                            x = 0
                        
                            if(highest == 1):
                                print("Maju")
                                # mc.forward(0.1)
                            if(highest == 2):
                                print("Kanan")
                                # mc.right(0.1)
                            if(highest == 3):
                                print("Kiri")
                                # mc.left(0.1)
                            if(highest == 4):
                                print("Belok Kanan")
                                # mc.turn_right(0.1)
                            if(highest == 5):
                                print("Belok Kiri")
                                # mc.turn_left(0.1)
                            buffer.clear()
                                
                # if(np.average(emg) <= 40):
                #     print("idle")
                # if(np.average(emg) >= 50 ):
                #     if(result_index == 2):
                #         print("kontol")
                #     if(result_index == 3):
                #         print("tolkon")
                #         # mc.back(0.01)
                #     if(result_index == 4):
                #         print("tonyol")
                #         # mc.right(0.01)
                #     if(result_index == 5):
                #         print("nyolton")
                #         # mc.left(0.01)
                                
                # current_time = time.time() - running
                # conv = current_time * 1000
                # conv_to_int = int(conv)
                # int_to_float = float(conv_to_int)
                # real_float = int_to_float / 1000
                # print(end_time - running)
                # with open('data/NewData/RotateLeft_25.txt', "a") as f:
                # # 3_Pre.txt
                # # 0_Raw.txt
                # # 3_Filtered.txt
                # #   # Use this for recording the data
                # #     # 0 for rest
                # #     # 1 for fist
                # #     # 2 for top
                # #     # 3 for bottom
                # #       4 for Rotate Right
                # #       5 for Rotate Left
                #         f.writelines("%s\n" % str(emgnew)+str(real_float))
                # #   # insert command below to writelines to print the timestamp
    except KeyboardInterrupt:
        print("Quitting")
        pygame.quit()
        quit()
