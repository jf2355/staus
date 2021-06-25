#!/usr/bin/env python

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)

event_count = 10000
check = 1

lt_list = []
pt_list = []
d0_list = []
mass_list = []

cf_masses = []
cf_lifetimes = []
cf_values_pt = []
cf_values_d0 = []
cf_values_both = []
cf_cats_pt = ["events", "seen"]
cf_cats_d0 = ["events", "seen"]
cf_cats_both = ["events", "seen"]

print("testing")

f = open('stau_efficiencies.json',)
data = json.load(f)

for i in data["lifetimes"]:
    lt_list.append(i)
for i in data["pts"]:
    pt_list.append(i)
for i in data["d0s"]:
    d0_list.append(i)
for i in data["data"]:
    if i["cmass"] not in mass_list:
        mass_list.append(i["cmass"])
mass_list.sort()


# Structure: [lifetime][pt][d0][mass]
cmasses = np.zeros((len(lt_list), len(pt_list), len(d0_list), len(mass_list))) * np.nan
efficiencies = np.zeros((len(lt_list), len(pt_list), len(d0_list), len(mass_list))) * np.nan
errors = np.zeros((len(lt_list), len(pt_list), len(d0_list), len(mass_list))) * np.nan

# Collect efficiency data in array
for i in data["data"]:
    L = lt_list.index(i["lifetime"])
    P = pt_list.index(i["pt"])
    D = d0_list.index(i["d0"])
    M = mass_list.index(i["cmass"])
    cmasses[L][P][D][M] = i["cmass"]
    efficiencies[L][P][D][M] = i["efficiency"]
    errors[L][P][D][M] = i["error"]


# Structure: [lifetime][mass]
avg_mass = np.zeros((len(lt_list), len(mass_list))) * np.nan
avg_energy = np.zeros((len(lt_list), len(mass_list))) * np.nan
avg_dist = np.zeros((len(lt_list), len(mass_list))) * np.nan
avg_pt = np.zeros((len(lt_list), len(mass_list))) * np.nan
avg_d0 = np.zeros((len(lt_list), len(mass_list))) * np.nan

#Structure: [lifetime, mass, value]
hist_energy = []
hist_dist = []
hist_trans_dist = []
hist_pt = []
hist_d0 = []
hist_eta = []

#Collect averages / histogram data
for i in data["hist"]:
    L = lt_list.index(i["lifetime"])
    M = mass_list.index(i["cmass"])
    temp_en_list = []
    energy_sum = 0
    dist_sum = 0
    pt_sum = 0
    d0_sum = 0

    for j in range(len(i["energy"])):
        temp_en_list.append(i["energy"][j] / 1000)      # /1000 for unit correction
        energy_sum += (i["energy"][j] / 1000)           # /1000 for unit correction
    for j in range(len(i["decay distance"])):
        dist_sum += (i["decay distance"][j])
    for j in range(len(i["pt_list"])):
        pt_sum += (i["pt_list"][j])
    for j in range(len(i["d0s"])):
        d0_sum += (i["d0s"][j])

    avg_mass[L][M] = i["cmass"]
    avg_energy[L][M] = (energy_sum / len(i["energy"]))
    avg_dist[L][M] = (dist_sum / len(i["decay distance"]))
    avg_pt[L][M] = (pt_sum / len(i["pt_list"]))
    avg_d0[L][M] = (d0_sum / len(i["d0s"]))

    hist_energy.append([L, M, i["energy"]])
    hist_dist.append([L, M, i["decay distance"]])
    hist_trans_dist.append([L, M, i["trans decay distance"]])
    hist_pt.append([L, M, i["pt_list"]])
    hist_d0.append([L, M, i["d0s"]])
    hist_eta.append([L, M, i["etas"]])


# Collect cutflow data
for i in data["str_pts"]:
    cf_cats_pt.append(i)
for i in data["str_d0s"]:
    cf_cats_d0.append(i)
for i in range(len(data["str_both"])):
    for j in range(len(data["str_both"][i])):
        cf_cats_both.append(data["str_both"][i][j])


for i in data["cutflow"]:
    cf_masses.append(i["cmass"])
    cf_lifetimes.append(i["lifetime"])
    temp_cf_values_pt = []
    temp_cf_values_d0 = []
    temp_cf_values_both = []
    for j in cf_cats_pt:
        temp_cf_values_pt.append(i[j])
    for j in cf_cats_d0:    
        temp_cf_values_d0.append(i[j])
    for j in cf_cats_both:
        temp_cf_values_both.append(i[j])
    temp_cf_values_pt[0] = event_count
    temp_cf_values_d0[0] = event_count
    temp_cf_values_both[0] = event_count
    cf_values_pt.append(temp_cf_values_pt)
    cf_values_d0.append(temp_cf_values_d0)
    cf_values_both.append(temp_cf_values_both)

print(cf_values_pt[0])

f.close()





#EFFICIENCY PLOTS
#-----------------

# Fixed transvese momentum efficiency plots
for i in range(len(lt_list)):
    for j in range(len(pt_list)):
        fig,axs = plt.subplots()
        axs.set_title("Lifetime: " + str(lt_list[i]) + ", Transverse Momentum: " + str(pt_list[j]) + " GeV")
        axs.set_xlabel("Mass (GeV)")
        axs.set_ylabel("Efficiency")
        for k in range(len(d0_list)):
            axs.errorbar(cmasses[i][j][k], efficiencies[i][j][k], yerr = errors[i][j][k], label = "d0 < " + str(d0_list[k]) + " mm", marker = "o", alpha = 0.5)
        axs.legend(fontsize = "small", frameon = False)
        fig.savefig('Stau_eff_' + str(lt_list[i]) + '_pt' + str(pt_list[j]) + '.pdf' )

# Fixed d0 efficiency plots
for i in range(len(lt_list)):
    for j in range(len(d0_list)):
        fig, axs = plt.subplots()
        axs.set_title("Lifetime: " + str(lt_list[i]) + ", d0: " + str(d0_list[j]) + " mm")
        axs.set_xlabel("Mass (GeV)")
        axs.set_ylabel("Efficiency")
        for k in range(len(pt_list)):
            axs.errorbar(cmasses[i][k][j], efficiencies[i][k][j], yerr = errors[i][k][j], label = "pt > " + str(pt_list[k]) + " GeV", marker = "o", alpha = 0.5)
        axs.legend(fontsize = "small", frameon = False)
        fig.savefig('Stau_eff_' + str(lt_list[i]) + '_d0' + str(d0_list[j]) + '.pdf' )



# AVERAGES PLOTS
#---------------------

# Average energy plots
for i in range(len(lt_list)):
    fig,axs = plt.subplots()
    axs.set_title("Lifetime: " + str(lt_list[i]))
    axs.set_xlabel("Mass (GeV)")
    axs.set_ylabel("Average Energy (GeV)")
    axs.errorbar(avg_mass[i], avg_energy[i])
    fig.savefig('Stau_avg_energy_' + str(lt_list[i]) + '.pdf' )

# Average displacementplots
for i in range(len(lt_list)):
    fig,axs = plt.subplots()
    axs.set_title("Lifetime: " + str(lt_list[i]))
    axs.set_xlabel("Mass (GeV)")
    axs.set_ylabel("Average Displacement (mm)")
    axs.errorbar(avg_mass[i], avg_dist[i])
    fig.savefig('Stau_avg_dist_' + str(lt_list[i]) + '.pdf' )

# Average pt plots
for i in range(len(lt_list)):
    fig,axs = plt.subplots()
    axs.set_title("Lifetime: " + str(lt_list[i]))
    axs.set_xlabel("Mass (GeV)")
    axs.set_ylabel("Average Transverse Momentum (GeV)")
    axs.errorbar(avg_mass[i], avg_pt[i])
    fig.savefig('Stau_avg_pt_' + str(lt_list[i]) + '.pdf' )

# Average d0 plots
for i in range(len(lt_list)):
    fig,axs = plt.subplots()
    axs.set_title("Lifetime: " + str(lt_list[i]))
    axs.set_xlabel("Mass (GeV)")
    axs.set_ylabel("Average d0 (GeV)")
    axs.errorbar(avg_mass[i], avg_d0[i])
    fig.savefig('Stau_avg_d0_' + str(lt_list[i]) + '.pdf' )



# GENERAL HISTOGRAMS
#--------------------

# Energy Histogram
for i in range(len(lt_list)):
    for j in range(len(mass_list)):
        spec_energies = []
        for k in hist_energy:
            if k[0] == i and k[1] == j:
                spec_energies.extend(k[2])
        bebo = 0
        title = "Stau_energy_" + str(lt_list[i]) + "_" + str(mass_list[j]) + ".pdf"
        fig,axs = plt.subplots()
        axs.set_title("Energy (lifetime: " + str(lt_list[i]) + ", mass: " + str(mass_list[j]))
        axs.set_xlabel("Energy (GeV)")
        axs.set_ylabel("Particles")
        axs.set_yscale("log")
        axs.hist(spec_energies, bins = 35)
        fig.savefig(title )

# Decay Distance Histogram
for i in range(len(lt_list)):
    for j in range(len(mass_list)):
        spec_dists = []
        for k in hist_dist:
            if k[0] == i and k[1] == j:
                spec_dists.extend(k[2])
        title = "Stau_decay_dist_" + str(lt_list[i]) + "_" + str(mass_list[j]) + ".pdf"
        fig,axs = plt.subplots()
        axs.set_title("Decay Distance (lifetime: " + str(lt_list[i]) + ", mass: " + str(mass_list[j]))
        axs.set_xlabel("Decay Distance (mm)")
        axs.set_ylabel("Particles")
        axs.set_yscale("log")
        axs.hist(spec_dists, bins = 35)
        fig.savefig(title )

# Transverse Decay Distance Histogram
for i in range(len(lt_list)):
    for j in range(len(mass_list)):
        spec_trans_dists = []
        for k in hist_trans_dist:
            if k[0] == i and k[1] == j:
                spec_trans_dists.extend(k[2])
        title = "Stau_trans_decay_dist_" + str(lt_list[i]) + "_" + str(mass_list[j]) + ".pdf"
        fig,axs = plt.subplots()
        axs.set_title("Transverse Decay Distance (lifetime: " + str(lt_list[i]) + ", mass: " + str(mass_list[j]))
        axs.set_xlabel("Transverse Decay Distance (mm)")
        axs.set_ylabel("Particles")
        axs.set_yscale("log")
        axs.hist(spec_trans_dists, bins = 35)
        fig.savefig(title )

# p_T Histogram
for i in range(len(lt_list)):
    for j in range(len(mass_list)):
        spec_pt = []
        for k in hist_pt:
            if k[0] == i and k[1] == j:
                spec_pt.extend(k[2])
        title = "Stau_pt_" + str(lt_list[i]) + "_" + str(mass_list[j]) + ".pdf"
        fig,axs = plt.subplots()
        axs.set_title("Transverse Momentum (lifetime: " + str(lt_list[i]) + ", mass: " + str(mass_list[j]))
        axs.set_xlabel("Transverse Momentum (GeV)")
        axs.set_ylabel("Particles")
        axs.set_yscale("log")
        axs.hist(spec_pt, bins = 35)
        fig.savefig(title )

# d0 Histogram
for i in range(len(lt_list)):
    for j in range(len(mass_list)):
        spec_d0 = []
        for k in hist_d0:
            if k[0] == i and k[1] == j:
                spec_d0.extend(k[2])
        title = "Stau_d0_" + str(lt_list[i]) + "_" + str(mass_list[j]) + ".pdf"
        fig,axs = plt.subplots()
        axs.set_title("d0 (lifetime: " + str(lt_list[i]) + ", mass: " + str(mass_list[j]))
        axs.set_xlabel("d0 (mm)")
        axs.set_ylabel("Particles")
        axs.set_yscale("log")
        axs.hist(spec_d0, bins = 35)
        fig.savefig(title )

# Eta Histogram
for i in range(len(lt_list)):
    for j in range(len(mass_list)):
        spec_eta = []
        for k in hist_eta:
            if k[0] == i and k[1] == j:
                spec_eta.extend(k[2])
        title = "Stau_eta_" + str(lt_list[i]) + "_" + str(mass_list[j]) + ".pdf"
        fig,axs = plt.subplots()
        axs.set_title("Eta (lifetime: " + str(lt_list[i]) + ", mass: " + str(mass_list[j]))
        axs.set_xlabel("Eta")
        axs.set_ylabel("Particles")
        axs.hist(spec_eta, bins = 35)
        fig.savefig(title )



# CUTFLOW HISTOGRAMS
# --------------------

# pT Cutflow Histograms
for i in range(len(cf_masses)):
    title = "Stau_cf_pT_" + str(cf_masses[i]) + "_" + str(cf_lifetimes[i]) + ".pdf"
    fig,axs = plt.subplots()
    axs.set_title("Surviving Events (mass: " + str(cf_masses[i]) + ", lifetime: " + str(cf_lifetimes[i]) + ")")
    axs.set_ylabel("Events")
    axs.set_yscale("log")
    axs.bar(cf_cats_pt, cf_values_pt[i])
    fig.savefig(title )

# d0 Cutflow Histograms
for i in range(len(cf_masses)):
    title = "Stau_cf_d0_" + str(cf_masses[i]) + "_" + str(cf_lifetimes[i]) + ".pdf"
    fig,axs = plt.subplots()
    axs.set_title("Surviving Events (mass: " + str(cf_masses[i]) + ", lifetime: " + str(cf_lifetimes[i]) + ")")
    axs.set_ylabel("Events")
    axs.set_yscale("log")
    axs.bar(cf_cats_d0, cf_values_d0[i])
    fig.savefig(title )

# Full Cutflow Histograms
for i in range(len(cf_masses)):
    title = "Stau_cf_fullcuts_" + str(cf_masses[i]) + "_" + str(cf_lifetimes[i]) + ".pdf"
    fig,axs = plt.subplots()
    axs.set_title("Surviving Events (mass: " + str(cf_masses[i]) + ", lifetime: " + str(cf_lifetimes[i]) + ")")
    axs.set_ylabel("Events")
    axs.set_yscale("log")
    plt.xticks(rotation = 30)
    axs.bar(cf_cats_both, cf_values_both[i])
    fig.savefig(title )

