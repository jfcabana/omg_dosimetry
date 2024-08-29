# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:53:30 2022

@author: Peter Truong (CISSSO)
modified by: JFC (CISSSCA)
@version: 2024-03-12
"""

from omg_dosimetry.analysis import load_analysis
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()                              # Declare root (top-level instance)
root.withdraw()                             # Show only dialog without any other GUI elements by hiding root window
root.attributes("-topmost", True)           # Top-level window display priority

file_path = filedialog.askopenfilename(filetypes = [("Pickle File", ".pkl")])
analysis = load_analysis((file_path))

analysis.show_results()
