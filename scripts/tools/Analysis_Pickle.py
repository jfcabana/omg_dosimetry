# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:53:30 2022

@author: Peter Truong (CISSSO)
@version: 05 decembre 2023
"""

import pickle, bz2
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()                              # Declare root (top-level instance)
root.withdraw()                             # Show only dialog without any other GUI elements by hiding root window
root.attributes("-topmost", True)           # Top-level window display priority

def main(file_path = None, show_results = True):
    if not file_path: file_path = filedialog.askopenfilename(parent = root, filetypes = [("Pickle File", ".pkl")])
    print("\nLoading .pkl file: {}...".format(file_path))
    
    try: analysis = pickle.load(bz2.open(file_path, "rb"))  # Whether or not .pkl was compressed or not.
    except: analysis = pickle.load(open(file_path, "rb"))
    
    if show_results: analysis.show_results(show = True)
    
    return analysis
            
if __name__ == "__main__":
    pkl = main()