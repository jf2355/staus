import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot4 as uproot
import scipy.interpolate
import math
from scipy.interpolate import griddata
from scipy.interpolate import interpn
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                       AutoMinorLocator)
import matplotlib
from matplotlib import colors as mcolors
import colorsys
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

plt.ioff()
matplotlib.use('Agg')

pt_list = []
energy_list = []

model = input('higgs or staus: ')
higgs = 'higgs'
staus = 'staus'
print(model)
if model == higgs:
    f = open('%s_ptandenergy.json'%(model))
    data = json.load(f)
elif model == staus:
    f = open('%s_ptandenergy.json'%(model))
    data = json.load(f)

for i in data["pts"]:
    pt_list.append(i)
for i in data["energy"]:
    energy_list.append(i)

f.close()

if model == staus:
    for i in range(len(pt_list)):
        fig,axs = plt.subplots()
        axs.set_title("pt Staus Hist")
        axs.set_xlabel("Pt (GeV)")
        axs.set_ylabel("Particles")
        axs.set_xscale("log")
        plt.hist(pt_list, 1000)
    for i in range(len(energy_list)):
        fig.savefig('plots/stauspt.pdf')
        fig,axs = plt.subplots()
        axs.set_title("Energy Staus Hist")
        axs.set_xlabel("E (GeV)")
        axs.set_ylabel("Particles")
        axs.set_xscale("log")
        plt.hist(energy_list, 1000)
        fig.savefig('plots/stausE.pdf')

if model == higgs:
    for i in range(len(pt_list)):
        fig,axs = plt.subplots()
        axs.set_title("pt Higgs Hist")
        axs.set_xlabel("Pt (GeV)")
        axs.set_ylabel("Particles")
        axs.set_xscale("log")
        plt.hist(pt_list, 1000)
        fig.savefig('plots/higgspt.pdf')
    for i in range(len(energy_list)):
        fig,axs = plt.subplots()
        axs.set_title("Energy Higgs Hist")
        axs.set_xlabel("E (GeV)")
        axs.set_ylabel("Particles")
        axs.set_xscale("log")
        plt.hist(energy_list, 1000)
        fig.savefig('plots/higgsE.pdf')
