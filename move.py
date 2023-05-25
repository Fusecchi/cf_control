
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
import logging
import sys
import time
from threading import Event

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper

q = multiprocessing.Queue()

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

DEFAULT_HEIGHT = 0.5

deck_attached_event = Event()

logging.basicConfig(level=logging.ERROR)


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
    pygame.display.flip()
    last_vals = vals



if __name__ == '__main__':

    cflib.crtp.init_drivers()
    p = multiprocessing.Process(target=worker, args=(q,))
    p.start()

    w, h = 800, 600
    scr = pygame.display.set_mode((w, h))
    running = time.time()
    model = tf.keras.models.load_model("/home/pc/Downloads/Ovass30.h5", compile=True)
    cflib.crtp.init_drivers()
    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        with MotionCommander(scf) as mc: 
            
            mc.up(0.1)
            try:
                x = 0
                buffer = list()
                while True:
                    mc.stop()
                    pygame.event.pump()
                    end_time = time.time()
                    emgnew = list()


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
                    result_index = np.argmax(prediction)
                    result = np.around(prediction, decimals=2)
                    print(str(result_index) + "Prediction : " + str(result))
                    # buffer.append(result_index)
                    # code_to_control = np.argmax(buffer)
                    # print(buffer)
                    # # print(result_index)
                    print(buffer)
                    if(np.average(emg) <= 40):
                        print("idle")
                        buffer.clear()
                    else:
                        if x < 6:
                            result_index = np.argmax(prediction)
                            buffer.append(result_index)
                            x+=1
                            if x == 6:
                                highest = max(set(buffer), key = buffer.count)
                                print(highest)
                                x = 0
                            
                                if(highest == 1):
                                    print("Maju")
                                    mc.forward(0.1)
                                if(highest == 2):
                                    print("Kanan")
                                    mc.right(0.1)
                                if(highest == 3):
                                    print("Kiri")
                                    mc.left(0.1)
                                if(highest == 4):
                                    print("Belok Kanan")
                                    mc.turn_right(0.1)
                                if(highest == 5):
                                    print("Belok Kiri")
                                    mc.turn_left(0.1)
                                buffer.clear()
                                    
                            
            except KeyboardInterrupt:
                print("Quitting")
                mc.land()
                pygame.quit()
                quit()