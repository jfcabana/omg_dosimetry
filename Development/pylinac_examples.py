# -*- coding: utf-8 -*-
"""
This is just a copy-paste of the examples detailed here:
https://pypi.org/project/pylinac/
"""

#%% GUI
import pylinac
pylinac.gui()

#%% TG-51
from pylinac import tg51

tg51_6x = tg51.TG51Photon(temp=TEMP, press=PRESS, model=CHAMBER,
                          n_dw=ND_w, p_elec=P_ELEC,
                          measured_pdd=66.4, lead_foil=None,
                          clinical_pdd=66.5, energy=ENERGY,
                          volt_high=-300, volt_low=-150,
                          m_raw=(25.65, 25.66, 25.65),
                          m_opp=(25.64, 25.65, 25.65),
                          m_low=(25.64, 25.63, 25.63),
                          mu=MU, tissue_correction=1.0)

# Done!
print(tg51_6x.dose_mu_dmax)

# examine other parameters
tg51_6x.pddx
tg51_6x.kq
tg51_6x.p_ion

# change readings if you adjust output
tg51_6x.m_raw = (25.44, 25.44, 25.43)
# print new dose value
print(tg51_6x.dose_mu_dmax)

#%% Planar Phantom Analysis (Leeds TOR, StandardImaging QC-3, Las Vegas)
from pylinac import LeedsTOR, StandardImagingQC3, LasVegas

#leeds = LeedsTOR("my_leeds.dcm")
leeds = LeedsTOR.from_demo_image()
leeds.analyze()
leeds.plot_analyzed_image()
leeds.publish_pdf()

#qc3 = StandardImagingQC3("my_qc3.dcm")
qc3 = StandardImagingQC3.from_demo_image()
qc3.analyze()
qc3.plot_analyzed_image()
qc3.publish_pdf('qc3.pdf')

#lv = LasVegas("my_lv.dcm")
lv = LasVegas.from_demo_image()
lv.analyze()
lv.plot_analyzed_image()
lv.publish_pdf('lv.pdf', open_file=True)  # open the PDF after publishing

#%% Winston-Lutz Analysis
from pylinac import WinstonLutz

wl = WinstonLutz("wl/image/directory")  # images are analyzed upon loading
wl.plot_summary()
print(wl.results())
wl.publish_pdf('my_wl.pdf')

#%% Starshot Analysis
from pylinac import Starshot

star = Starshot("mystarshot.tif")
star.analyze(radius=0.75, tolerance=1.0, fwhm=True)
print(star.return_results())  # prints out wobble information
star.plot_analyzed_image()  # shows a matplotlib figure
star.publish_pdf()  # publish a PDF report

#%% VMAT QA
from pylinac import VMAT

vmat = VMAT(images=["DRGSopen.dcm", "DRGSdmlc.dcm"], delivery_types=["open", "dmlc"])
vmat.analyze(test='drgs', tolerance=1.5)
print(vmat.return_results())  # prints out ROI information
vmat.plot_analyzed_image()  # shows a matplotlib figure
vmat.publish_pdf('myvmat.pdf')  # generate a PDF report

#%% CT & CBCT QA 
from pylinac import CatPhan504, CatPhan503, CatPhan600

# for this example, we'll use the CatPhan504
cbct = CatPhan504("my/cbct_image_folder")
cbct.analyze(hu_tolerance=40, scaling_tolerance=1, thickness_tolerance=0.2, low_contrast_threshold=1)
print(cbct.return_results())
cbct.plot_analyzed_image()
cbct.publish_pdf('mycbct.pdf')

#%% Log analysis
from pylinac import load_log

tlog = load_log("tlog.bin")
# after loading, explore any Axis of the Varian structure
tlog.axis_data.gantry.plot_actual()  # plot the gantry position throughout treatment
tlog.fluence.gamma.calc_map(doseTA=1, distTA=1, threshold=10, resolution=0.1)
tlog.fluence.gamma.plot_map()  # show the gamma map as a matplotlib figure
tlog.publish_pdf()  # publish a PDF report

dlog = load_log("dynalog.dlg")

#%% Picket Fence MLC Analysis
from pylinac import PicketFence

pf = PicketFence("mypf.dcm")
pf.analyze(tolerance=0.5, action_tolerance=0.25)
print(pf.return_results())
pf.plot_analyzed_image()
pf.publish_pdf()