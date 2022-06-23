################################################################################
# ABYSSESS
#
# Image Classification/Segmentation prototype
#
# Author: Pedro J. Soto Vega
################################################################################

import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox

import os
import numpy as np


stopb = None
font = "Times New Roman"
fontButtons = (font, 10)
fontButtons1 = (font, 12)

class App():
    def __init__(self, window, window_tittle):

        self.window = window

        # get screen width and height
        ws = window.winfo_screenwidth() # width of the screen
        hs = window.winfo_screenheight() # height of the screen

        self.window.title(window_tittle)
        self.window.resizable(width=False, height=False)
        self.window.iconname('Image Recognition')
        self.title = tk.Label(window, text="Image Recognition application",font = fontButtons, height= 1)
        self.title.pack()
        self.t = []
        self.width = 648
        self.height = 648
        # calculate x and y coordinates for the Tk root window
        x = (ws/2) - (self.width + 320/2)
        y = (hs/2) - (self.height/2)

        # set the dimensions of the screen
        # and where it is placed
        self.window.geometry('%dx%d+%d+%d' % (self.width + 320, self.height, x, y))

        positions = 10

        self.canvas = tk.Canvas(window, width=self.width+320, height=self.height)

        self.canvas.create_line(10,   5, 1024,   5)
        self.canvas.create_line(10, 640, 1024, 640)
        self.canvas.create_line(10,   5,   10, 640)
        self.canvas.create_line(1024, 5, 1024, 640)
        self.canvas.pack()
        #self.picdir = os.path.join('best_pb','frozen_ds_best.pb')

        self.window.mainloop()

if __name__ == '__main__':
    App(tk.Tk(), 'Image Recognition Prototype 1.0')
