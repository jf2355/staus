#!/usr/bin/env python

import pyhepmc_ng as hep
import numpy
import math
import glob
import json
import numpy as np
import uproot4 as uproot
import scipy.interpolate
from scipy.interpolate import griddata
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from random import seed
from random import random
from matplotlib import colors as mcolors
import colorsys

plt.ioff()

model = input('higgs or staus: ')
higgs = 'higgs'
staus = 'staus'
print(model)
file_lists = {}
doTest = False
if model == higgs:
    cap_mod = 'Higgs'
    masses = [5,8,15,25,40,55]
    for m in masses:
        #file_list = glob.glob("/eos/user/k/kdipetri/Snowmass_HepMC/run_higgsportal/higgsportal_125_5_0p1ns/events.hepmc")
        file_lists[m] = glob.glob("/eos/user/k/kdipetri/Snowmass_HepMC/run_higgsportal/higgsportal_125_%d_0p1ns/events.hepmc"%m)
        #file_list = glob.glob("/eos/user/j/jefarr/run_higgsportal/*/events.hepmc")
    #=====================================================================================================
elif model == staus:
    cap_mod = 'Stau'
    masses = [100,200,300,400,500,600]
    for m in masses:
        file_lists[m] = glob.glob("/eos/user/k/kdipetri/Snowmass_HepMC/run_staus/stau_%d_0_0p1ns/events.hepmc"%m)
        #file_list = glob.glob("/eos/user/j/jefarr/run_staus/*/events.hepmc")
        #=====================================================================================================
else:
    print('Please enter either higgs or staus (case sensitive)')

zero = numpy.array([0,0,0])

charged_list = [1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 15, 17, 24, 34, 37, 38, 62, 1000011, 1000013,
    1000015, 2000011, 2000013, 2000015, 2000024, 2000037, 211, 9000211, 100211,
    10211, 9010211, 213, 10213, 20213, 9000213, 100213, 9010213, 9020213, 30213,
    9030213, 9040213, 215, 10215, 9000215, 9010215, 217, 9000217, 9010217, 219,
    321, 9000321, 10321, 100321, 9010321, 9020321, 323, 10323, 20323, 100323,
    9000323, 30323, 325, 9000325, 10325, 20325, 9010325, 9020325, 327, 9010327, 329,
    9000329, 411, 10411, 413, 10413, 20413, 415, 431, 10431, 433, 10433, 20433, 435,
    521, 10521, 523, 10523, 20523, 525, 541, 10541, 543, 10543, 20543, 545, 2224,
    2214, 1114, 3222, 3112, 3224, 3114, 3312, 3314, 3334, 4122, 4222, 4212, 4224,
    4214, 4232, 4322, 4324, 4412, 4422, 4414, 4424, 4432, 4434, 4444, 5112, 5222,
    5114, 5224, 5132, 5312, 5314, 5332, 5334, 5242, 5422, 5424, 5442, 5444, 5512,
    5514, 5532, 5534, 5554, 9221132, 9331122]


#**************************************************************************************************************************

# Depth-first search of particle decay paths
# Based on this stackoverflow example:
#https://stackoverflow.com/questions/59132538/counting-the-length-of-each-branch-in-a-binary-tree-and-print-out-the-nodes-trav
def decaysToSelf(particle) :
    notSelfDecay = True
    for child in particle.end_vertex.particles_out:
        if ( abs(child.pid) == abs(particle.pid) and child.id!=particle.id and child.id < 100000) :
            notSelfDecay = False
            break
    return not notSelfDecay


def findBSMParticles(truthparticles, PDGID=None, decays=False) :
    #print("Start")
    BSM_particles = []

    for iparticle,particle in enumerate(truthparticles):
        # Handed it a PDG ID?
        if PDGID :
            if model == higgs:
                if abs(particle.pid) != PDGID:
                    continue
            elif model == staus:
                if abs(particle.pid) not in PDGID:
                    continue

        # Otherwise, interested in SUSY particles only
        elif abs(particle.pid) < 999999:
            continue
        #Find stable particles or particle not decaying into itself
        if particle.end_vertex :
            if not decaysToSelf(particle) :
                BSM_particles.append(particle)
        if not particle.end_vertex :
            BSM_particles.append(particle)

    if len(BSM_particles) != 2:
        print(len(BSM_particles))
        print("Oops - there aren't 2 BSM particles!!! Evacute Earth")

    return BSM_particles


def dfs_paths(stack, particle, stable_particles = []):

    if particle == None:
        return

    # append this particle ID to the path array
    stack.append(particle.id)

    # If this particle is the end of the chain, save it
    if(not particle.end_vertex):

        # Check status
        if particle.status != 1 :
            print("Uh oh! Stable particle has status",particle.status())
            exit(1)

        # Append
        stable_particles.append(particle)

    # Otherwise try each particle from decay
    else :
        for child in particle.end_vertex.particles_out :
            dfs_paths(stack, child, stable_particles)

    # Magic
    stack.pop()


# Only works if you decayed the parent in the generation
# step, or you're running this on a post-simulation xAOD
def findBSMDecayProducts(particle,charge_eta=True) :

    stable_descendents = []
    dfs_paths([],particle,stable_descendents)
    # If we want charged and eta ok only, subdivide
    if charge_eta :
        children = [i for i in stable_descendents if ((abs(i.pid) in charged_list) and (abs(i.momentum.eta()) <= 2.5))]
        return children
    else :
        return stable_descendents

#**************************************************************************************************************************

#File Level Start ===================================================================================================

output = {}

if model == higgs:
    for m in file_lists:
        #print(len(file_list))
        if 'stable' in file_lists[m]:
            continue
        infile = file_lists[m][0]
        print(infile)
        #n += 1
        pt_list = []
        energy_list = []
        dec_dist_list = []
        eta_list = []


        with hep.open(infile) as f:

        #Event Level Start ==============================================================================================

            while True :
                evt = f.read()
                if not evt:
                    break
                if doTest and evt.event_number > 10:
                    break
                if evt.event_number % 1000 == 0:
                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    print("Current time is:", current_time, " On file:", m+1, " Event:", evt.event_number)

                #Event tracking lists
                decay_pt = []           # Location of decay vertex list
                hats = []               # Direction of daughter particle tracks list
                line_points = []        # Second point on daughter particle tracks list
                dists = []              # d0's list
                tracks = 0              # Number of trackable particles in an event


                #finds all the BSM pdgid 35 particles that decay into something else
                if model == higgs:
                    BSM_particles = findBSMParticles(evt.particles, PDGID = 35)
                elif model == staus:
                    BSM_particles = findBSMParticles(evt.particles, PDGID = [1000015, 2000015])


        #Particle Level Start =========================================================================================

                for bsm_part in BSM_particles:
                    event_prods = findBSMDecayProducts(bsm_part,charge_eta=True)

                    for part in event_prods:
                        # Vertex info
                        prodvtx = part.production_vertex
                        fourvec = prodvtx.position
                        point = numpy.array([fourvec.x, fourvec.y, fourvec.z])
                        dec_dist = numpy.sqrt(numpy.square(point[0]) + numpy.square(point[1]) + numpy.square(point[2]))
                        trans_dec_dist = numpy.sqrt(numpy.square(point[0]) + numpy.square(point[1]))

                        # Particle info
                        x = part.momentum.x
                        y = part.momentum.y
                        z = part.momentum.z
                        pt = (math.sqrt((x ** 2) + (y ** 2))) / 1000
                        decay_pt.append(pt)
                        pt_list.append(pt)
                        mom_mag = math.sqrt((x ** 2) + (y ** 2) + (z ** 2))
                        energy_list.append((numpy.sqrt(numpy.square(mom_mag) + numpy.square(part.generated_mass)))/1000)

        output[m] = {"pt":pt_list, "energy":energy_list}

if model == staus:
    for m in file_lists:
        #print(len(file_list))
        if 'stable' in file_lists[m]:
            continue
        infile = file_lists[m][0]
        print(infile)
        #n += 1
        pt_list = []
        energy_list = []
        dec_dist_list = []
        eta_list = []


        with hep.open(infile) as f:

        #Event Level Start ==============================================================================================

            while True :
                evt = f.read()
                if not evt:
                    break
                if doTest and evt.event_number > 10:
                    break
                if evt.event_number % 1000 == 0:
                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    print("Current time is:", current_time, " On file:", m+1, " Event:", evt.event_number)

                #Event tracking lists
                decay_pt = []           # Location of decay vertex list
                hats = []               # Direction of daughter particle tracks list
                line_points = []        # Second point on daughter particle tracks list
                dists = []              # d0's list
                tracks = 0              # Number of trackable particles in an event


                #finds all the BSM pdgid 35 particles that decay into something else
                if model == higgs:
                    BSM_particles = findBSMParticles(evt.particles, PDGID = 35)
                elif model == staus:
                    BSM_particles = findBSMParticles(evt.particles, PDGID = [1000015, 2000015])


        #Particle Level Start =========================================================================================

                for bsm_part in BSM_particles:
                    event_prods = findBSMDecayProducts(bsm_part,charge_eta=True)

                    for part in event_prods:
                        # Vertex info
                        prodvtx = part.production_vertex
                        fourvec = prodvtx.position
                        point = numpy.array([fourvec.x, fourvec.y, fourvec.z])
                        dec_dist = numpy.sqrt(numpy.square(point[0]) + numpy.square(point[1]) + numpy.square(point[2]))
                        trans_dec_dist = numpy.sqrt(numpy.square(point[0]) + numpy.square(point[1]))

                        # Particle info
                        x = part.momentum.x
                        y = part.momentum.y
                        z = part.momentum.z
                        pt = (math.sqrt((x ** 2) + (y ** 2))) / 1000
                        decay_pt.append(pt)
                        pt_list.append(pt)
                        mom_mag = math.sqrt((x ** 2) + (y ** 2) + (z ** 2))
                        energy_list.append((numpy.sqrt(numpy.square(mom_mag) + numpy.square(part.generated_mass)))/1000)

        output[m] = {"pt":pt_list, "energy":energy_list}

fig,axs = plt.subplots()
axs.set_title("$\mathrm{p_{t}}$ Distribution for Varying Mass %s Decays"%(cap_mod))
axs.set_xlabel("$\mathrm{p_{t}}$ (GeV)")
axs.set_ylabel("Particles")
axs.set_xscale("log")
for m in masses:
    pt_list = output[m]["pt"]
    #pt_list_norm = []
    #len_pt_list = len(pt_list)
    #print('len pt list is: ')
    #print(len_pt_list)
    #for i in range(len(pt_list)):
        #pt_list_norm.append(pt_list[i]/len_pt_list)
    #print('pt list is: ')
    #print(pt_list)
    #print('norm pt list is: ')
    #print(pt_list_norm)
    mean = np.mean(pt_list)
    round_mean = round(mean,2)
    plt.hist(pt_list, 1000, histtype='step', density =True, alpha=0.75, label="%d GeV Mean: "%m + str(round_mean))
plt.legend(loc='upper right')
fig.savefig('plots/%spt.pdf'%(model))

fig,axs = plt.subplots()
axs.set_title("Energy Distribution for Varying Mass %s Decays"%(cap_mod))
axs.set_xlabel("Energy (GeV)")
axs.set_ylabel("Particles")
axs.set_xscale("log")
for m in masses:
    energy_list = output[m]["energy"]
    #energy_list_norm = []
    #len_energy_list = len(energy_list)
    #print('len energy list is: ')
    #print(len_energy_list)
    #for i in range(len(energy_list)):
        #energy_list_norm.append(energy_list[i]/len_energy_list)
    #print('energy list is: ')
    #print(energy_list)
    #print('norm energy list is: ')
    #print(energy_list_norm)
    mean = np.mean(energy_list)
    round_mean = round(mean,2)
    plt.hist(energy_list, 1000, histtype='step', density=True, alpha=0.75, label="%d GeV Mean: "%m + str(round_mean))
plt.legend(loc='upper right')
fig.savefig('plots/%sE.pdf'%(model))

#File Level End =====================================================================================================
save_name = '%s_ptandenergy.json'%(model)
with open(save_name, 'w') as fp:
        json.dump(output, fp)


