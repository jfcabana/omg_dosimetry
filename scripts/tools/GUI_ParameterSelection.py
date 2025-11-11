# -*- coding: utf-8 -*-
__author__ = "Peter Truong"
__contact__ = "petertruong.cissso@ssss.gouv.qc.ca"
__version__ = "11 novembre 2025"

import tkinter as tk
from tkinter import ttk
import json, os, sys

### Site-Specific Initialization
SETTINGS_FILE = (r"\\SVWCT2Out0455\Phys\Répertoires individuels\Peter Truong"
                 r"\GITEA\OMG Film Dosimetry\scripts\qa_patients"
                 r"\CISSSO_settings.json")
physiciens = ["CA", "DW", "JFM", "LA", "MBL", "ML", "PT"]
machines = ["Clinac 1", "Clinac 3", "Truebeam 2"]       # Clinique
film_lot = ["EBT3 C2", "EBT-XD X3"]                     # LUT file association
    
### Load previous settings
def load_settings():
    if os.path.exists(SETTINGS_FILE): 
        with open(SETTINGS_FILE, "r") as f: return json.load(f)
    # Default parameters/settings if .json file not created yet
    return {"author": "", "unit": "", "film_lot": "", "date_exposed": "",
            "date_scanned": "", "wait_time": "", "notes": "", 
            "norm_mode": "Normalization Film", "norm_film_MU": "300", 
            "save_pdf_pkl": True, "dose_threshold": "0.1"}

### Save settings
def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f: json.dump(settings, f)

def run_gui():
    ### Sub-function/command
    def on_selection(event):
        if combo_norm_mode.get() == "Normalization Film": 
            text_norm_film_MU.config(state = "normal")              # Enable
        else: text_norm_film_MU.config(state = "disabled")          # Disable       
        
    def on_ok():
        parameter['author'] = combo_author.get()
        parameter['unit'] = combo_unit.get()
        parameter['film_lot'] = combo_film_lot.get()
        parameter['date_exposed'] = text_date_exposed.get()
        parameter['date_scanned'] = text_date_scanned.get()
        parameter['wait_time'] = text_wait_time.get()
        parameter['notes'] = text_notes.get()
        
        parameter['norm_mode'] = combo_norm_mode.get()
        parameter['norm_film_MU'] = text_norm_film_MU.get()
        
        parameter['save_pdf_pkl'] = check_save_pdf_pkl.get()
        parameter['dose_threshold'] = text_dose_threshold.get()
        
        save_settings(parameter)
        gui.quit()
        gui.destroy()
        
    def on_close():
        gui.destroy()
        sys.exit("Parameter window closed.\nClosing script... ")
      
    ### GUI initialization/start
    gui = tk.Tk()        
    gui.attributes("-topmost", True)   # Top-level window display priority
    gui.title("Optimized Multi-channel Gafchromic Parameter Selection")

    notebook = ttk.Notebook(gui)
    gui.resizable(True, True)
    notebook.pack(expand=True, fill='both')

    ### Parameter initialization
    parameter = load_settings()         # Load from previous values/.json file
    
    ### Information section
    tk.Label(gui, text = "Information", font = 14).pack(anchor = "w", 
                                                         pady = 10)
    tk.Label(gui, text = "Author: ").pack(anchor = "w")
    combo_author = ttk.Combobox(gui, values = physiciens, state = "readonly")
    combo_author.set(parameter["author"])
    combo_author.pack(anchor = "w", pady = 5)
    
    tk.Label(gui, text = "Unit: ").pack(anchor = "w")
    combo_unit = ttk.Combobox(gui, values = machines, state = "readonly")
    combo_unit.set(parameter["unit"])
    combo_unit.pack(anchor = "w", pady = 5)
    
    tk.Label(gui, text = "Film lot: ").pack(anchor = "w")
    combo_film_lot = ttk.Combobox(gui, values = film_lot, state = "readonly")
    combo_film_lot.set(parameter["film_lot"])
    combo_film_lot.pack(anchor = "w", pady = 5)
    
    tk.Label(gui, text = "Date exposed (AAAA-MM-DD): ").pack(anchor = "w")
    text_date_exposed = tk.Entry(gui, width = 20)
    text_date_exposed.insert(0, parameter["date_exposed"])
    text_date_exposed.pack(anchor = "w", pady = 5)
    
    tk.Label(gui, text = "Date scanned (AAAA-MM-DD): ").pack(anchor = "w")
    text_date_scanned = tk.Entry(gui, width = 20)
    text_date_scanned.insert(0, parameter["date_scanned"])
    text_date_scanned.pack(anchor = "w", pady = 5)
    
    tk.Label(gui, text = "Wait time (hours): ").pack(anchor = "w")
    text_wait_time = tk.Entry(gui, width = 20)
    text_wait_time.insert(0, parameter["wait_time"])
    text_wait_time.pack(anchor = "w", pady = 5)
    
    tk.Label(gui, text = "Notes: ").pack(anchor = "w")
    text_notes = tk.Entry(gui, width = 50)
    text_notes.insert(0, parameter["notes"])
    text_notes.pack(anchor = "w", pady = 5)
    
    ### Dose parameters section
    tk.Label(gui, text = "Dose Parameters", font = 14).pack(anchor = "w", 
                                                             pady = 10)
    tk.Label(gui, text = "Normalization mode: ").pack(anchor = "w")
    combo_norm_mode = ttk.Combobox(gui, values = ["Normalization Film", 
                                                   "Reference ROI (Eclipse)"], 
                                   state = "readonly")
    combo_norm_mode.set(parameter["norm_mode"])
    combo_norm_mode.pack(anchor = "w", pady = 5)
    # Configure state of text_norm_film_MU based on selection
    combo_norm_mode.bind("<<ComboboxSelected>>", on_selection)  
    
    tk.Label(gui, text = "Normalization film MU: ").pack(anchor = "w")
    text_norm_film_MU = tk.Entry(gui, width = 50)
    text_norm_film_MU.insert(0, parameter["norm_film_MU"])
    text_norm_film_MU.config(state = "normal")
    text_norm_film_MU.pack(anchor = "w", pady = 5)
    
    tk.Label(gui, text = "Dose Threshold (cutoff): ").pack(anchor = "w")
    text_dose_threshold = tk.Entry(gui, width = 50)
    text_dose_threshold.insert(0, parameter["dose_threshold"])
    text_dose_threshold.pack(anchor = "w", pady = 5)
    
    ### Save file parameters section
    tk.Label(gui, text = "Save File Parameters", font = 14).pack(anchor = "w", 
                                                                  pady = 10)
    check_save_pdf_pkl = tk.BooleanVar(value = parameter["save_pdf_pkl"])
    tk.Checkbutton(gui, text = "Save pdf/pkl analysis file: ", 
                   variable = check_save_pdf_pkl).pack(anchor = "w")
    
    tk.Button(gui, text="OK", command=on_ok).pack(pady=10)

    gui.protocol("WM_DELETE_WINDOW", on_close)
    gui.mainloop()
    return parameter

# Run the GUI
if __name__ == "__main__":
    parameter = run_gui()