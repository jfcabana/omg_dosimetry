"""I/O helper functions"""

from pathlib import Path
import os
from pylinac.core.io import get_url

def retrieve_demo_file(name: str, force: bool = False) -> Path:
    """Retrieve the demo file either by getting it from file or from a URL.

    If the file is already on disk it returns the file name. If the file isn't
    on disk, get the file from the URL and put it at the expected demo file location
    on disk for lazy loading next time.

    Parameters
    ----------
    name : str
        File name.
    """

    urls = {
        "C14_calib-18h-1_001.tif": r"https://raw.githubusercontent.com/jfcabana/omg_dosimetry/OMG_master/scripts/demo/files/calibration/scan/",
        "C14_calib-18h-2_001.tif": r"https://raw.githubusercontent.com/jfcabana/omg_dosimetry/OMG_master/scripts/demo/files/calibration/scan/",
        "BeamProfile.txt": r"https://raw.githubusercontent.com/jfcabana/omg_dosimetry/OMG_master/scripts/demo/files/calibration/",
        "A1A_Multi_6cm_001.tif": r"https://raw.githubusercontent.com/jfcabana/omg_dosimetry/OMG_master/scripts/demo/files/tiff2dose/scan/",
        "RD1.2.752.243.1.1.20230116114244452.3800.11382.3.313.2.dcm": r"https://raw.githubusercontent.com/jfcabana/omg_dosimetry/OMG_master/scripts/demo/files/analysis/DoseRS/"
        }

    url = urls[name] + name
    #demo_path = Path(__file__).parent / "demo_files" / name
    
    
    # Root directory and path to demos
    #demo_dir = demo_path.parent
    root_demo_path = Path(__file__).parent / "demo_files"
    if not root_demo_path.exists():
        os.makedirs(root_demo_path)
    calib_demo_path = root_demo_path / "calibration" / "scan"
    if not calib_demo_path.exists():
        os.makedirs(calib_demo_path)
    tiff2dose_demo_path = root_demo_path / "tiff2dose" / "scan"
    if not tiff2dose_demo_path.exists():
        os.makedirs(tiff2dose_demo_path)
    analysis_demo_path = root_demo_path / "analysis" / "scan"
    if not analysis_demo_path.exists():
        os.makedirs(analysis_demo_path)

    #if not demo_dir.exists():
    #    os.makedirs(demo_dir)
    if name == "C14_calib-18h-1_001.tif" or name == "C14_calib-18h-2_001.tif":
        demo_path = calib_demo_path / name
        if force or not demo_path.exists():
            get_url(url, destination=demo_path)
    elif name == "BeamProfile.txt":
        demo_path = calib_demo_path / name
        if force or not demo_path.exists():
            get_url(url, destination=demo_path)
    elif name == "A1A_Multi_6cm_001.tif":
        demo_path = tiff2dose_demo_path / name
        if force or not demo_path.exists():
            get_url(url, destination=demo_path)
    elif name == "RD1.2.752.243.1.1.20230116114244452.3800.11382.3.313.2.dcm":
        demo_path = analysis_demo_path / name
        if force or not demo_path.exists():
            get_url(url, destination=demo_path)

    return demo_path
