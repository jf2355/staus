#!/usr/bin/env python

import pyhepmc_ng as hep
import numpy
import math
import glob
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from random import seed
from random import random


# file_list = glob.glob("/eos/user/k/kpachal/TrackTrigStudies/RunDirectories/run_higgsportal_testTaus/higgsportal_testTaus_125_*/events.hepmc")
file_list = glob.glob("/eos/user/k/kdipetri/Snowmass_HepMC/run_staus/*/events.hepmc")

#Hardcoded stuff, cheange this section ===============================================================
pt_pass_checks = [0.5, 1.0, 2.0, 5.0]
d0_pass_checks = [10, 20, 50, 100]
d0_min_check = 2
track_low_cut = 2
max_track_eff = 1
seed(1)
doTest = False
use_slope_eff = False
#=====================================================================================================

pt_cuts = []
d0_cuts = []
full_cuts = []
for i in range(len(pt_pass_checks)):
    full_cuts.append([])

for i in pt_pass_checks:
    pt_cuts.append("pt" + str(i))
for i in d0_pass_checks:
    d0_cuts.append("d" + str(i))
for i in range(len(pt_pass_checks)):
    for j in range(len(d0_pass_checks)):
        full_cuts[i].append("pt" + str(pt_pass_checks[i]) + "_d" + str(d0_pass_checks[j]))


lifetime_list = []
zero = numpy.array([0, 0, 0])
data_list = []
cutflow_list = []
histogram_list = []
seen_event_count_total = 0
events = 0
n = 0
#mass_order_tracking = []

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
#**************************************************************************************************************************
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
            #if (abs(particle.pid)==PDGID and particle.status==62) :
            if abs(particle.pid) not in PDGID:
                continue

        # Otherwise, interested in SUSY particles only
        elif abs(particle.pid) < 999999:
            continue
        #print("Here")
        #print("Particle status: ", particle.status)
        #if (particle.status==62) :
         #   BSM_particles.append(particle)
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
#**************************************************************************************************************************
#**************************************************************************************************************************

def findGoodTrackCount(pt, ptcut, d0, d0cut, use_slope_eff=True):
    temp = [0, 0, 0]
    #rng_check = random()
    #print(rng_check)
    if pt > ptcut:
        temp[0] = 1
    #if pt > ptcut and rng_check < max_track_eff:
        #temp[0] = 1
    #if d0 < d0cut and d0 > d0_min_check:
    if d0 > d0_min_check and passes_d0_cut(d0, d0cut, use_slope_eff):
       temp[1] = 1
    if (pt > ptcut and d0 < d0cut and d0 > d0_min_check):
        temp[2] = 1
    return temp

def passes_d0_cut(d0, d0cut, use_slope_eff):
    # Checks maximum value cut
    rng_check = random()
    if not use_slope_eff:
        print("doesnt use slope")
        print("rng", rng_check)
        print("max_track_eff", max_track_eff)
        print("passes check", d0 < d0cut and rng_check < max_track_eff)
        if d0 < d0cut and rng_check < max_track_eff:
            return True
        else: return False
    #Calculates efficiency using line
    #TODO define slope and yinter)
    y_inter = max_track_eff
    effslope = -max_track_eff/d0cut
    eff = effslope*d0 + y_inter
    print("uses slope")
    print("rng", rng_check)
    print("eff", eff)
    print("passes check", rng_check < eff)
    if rng_check < eff: return True
    else: return False






#File Level Start ===================================================================================================

for m in range(len(file_list)):
    print(len(file_list))
    infile = file_list[m]
    n += 1
    seen_event_count = 0
    event_count = 0

    #histogram pass counters
    pt_ok_list = numpy.zeros(len(pt_pass_checks))
    d0_ok_list = numpy.zeros(len(d0_pass_checks))

    #File tracking lists
    eta_list = []
    d0_list = []
    pt_list = []
    energy_list = []
    dec_dist_list = []
    trans_dec_dist_list = []

    event_passes = numpy.zeros([len(pt_pass_checks), len(d0_pass_checks)])

    with hep.open(infile) as f:
        dirname = infile.split("/")[-2]
        tokens = dirname.split("_")
        #CHANGE THIS BACK WHEN SWITCHING FROM ONLY TAUS!!!!!!!
        #mPar = tokens[1]
        mChild  = int(tokens[1])
        lifetime = tokens[3]
        if lifetime not in lifetime_list:
            lifetime_list.append(lifetime)
    # if mChild not in mass_order_tracking:
    #   mass_order_tracking.append(mChild)


    #Event Level Start ==============================================================================================

        while True :
            evt = f.read()
            if not evt:
                break
            if doTest and evt.event_number > 100:
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

            event_pt_ok_list = numpy.zeros(len(pt_pass_checks))
            event_d0_ok_list = numpy.zeros(len(d0_pass_checks))
            event_num_good_tracks = numpy.zeros([len(pt_pass_checks), len(d0_pass_checks)])
            track_ids = []
            for i in range(len(pt_pass_checks)):
                track_ids.append([])
                for j in range(len(d0_pass_checks)):
                    track_ids[i].append([])
            event_count += 1


            #finds all the BSM pdgid 35 particles that decay into something else
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

                    # Other Stage 1 Checks
                    tdd_check = True
                    if trans_dec_dist >= 300:
                        tdd_check = False
                    dec_vtx_check = True
                    if part.end_vertex:
                        decvtx = part.end_vertex
                        decfourvec = decvtx.position
                        decpoint = numpy.array([decfourvec.x, decfourvec.y, decfourvec.z])
                        prod_dec_dist = numpy.array([decpoint[0] - point[0], decpoint[1] - point[1], decpoint[2] - point[2]])
                        trans_prod_dec_dist = numpy.squt(numpy.square(prod_dec_dist[0]) + nupy.square(prod_dec_dist[1]))
                        if trans_prod_dec_dist < 200:
                            dec_vtx_check = False


                    # Particle info
                    x = part.momentum.x
                    y = part.momentum.y
                    z = part.momentum.z
                    pt = (math.sqrt((x ** 2) + (y ** 2))) / 1000
                    decay_pt.append(pt)
                    pt_list.append(pt)
                    mom_mag = math.sqrt((x ** 2) + (y ** 2) + (z ** 2))
                    mom_hat = numpy.array([x / mom_mag, y / mom_mag, z / mom_mag])
                    hat = mom_hat
                    line_point = hat * 10 + point
                    d = numpy.cross(point - line_point, zero - line_point) / numpy.linalg.norm(point - line_point)
                    d_mag = numpy.sqrt(numpy.square(d[0]) + numpy.square(d[1]))

                    # If pass all step 1 cuts:
                    if (tdd_check == True and dec_vtx_check == True):
                        tracks += 1

                        # Update tracking lists
                        dec_dist_list.append(dec_dist)
                        trans_dec_dist_list.append(trans_dec_dist)
                        d0_list.append(d_mag)
                        dists.append(d_mag)
                        eta_list.append(part.momentum.eta())
                        energy_list.append(numpy.sqrt(numpy.square(mom_mag) + numpy.square(part.generated_mass)))

                        #checks if track passes the pt / d0
                        for i in range(len(pt_pass_checks)):
                            for j in range(len(d0_pass_checks)):
                                tracker = findGoodTrackCount(pt, pt_pass_checks[i], d_mag, d0_pass_checks[j], use_slope_eff)
                                if tracker[0] == 1:
                                    event_pt_ok_list[i] += 1
                                if tracker[1] == 1:
                                    event_d0_ok_list[j] += 1
                                if tracker[2] == 1:
                                    event_num_good_tracks[i][j] += 1
                                    #keep track ids so we can do other things later
                                    track_ids[i][j].append(part.id)



    #Particle Level End ===========================================================================================

            if tracks > 0:
                seen_event_count += 1
            for i in range(len(event_pt_ok_list)):
                if event_pt_ok_list[i] >= track_low_cut:
                    pt_ok_list[i] += 1
            for i in range (len(event_d0_ok_list)):
                if event_d0_ok_list[i] >= track_low_cut:
                    d0_ok_list[i] += 1
            for i in range(len(event_num_good_tracks)):
                for j in range(len(event_num_good_tracks[i])):
                    if event_num_good_tracks[i][j] >= track_low_cut:
                        event_passes[i][j] += 1


                    #Event Level End ================================================================================================

    events += event_count
    seen_event_count_total += seen_event_count

    efficiencies = numpy.empty([len(event_passes), len(event_passes[0])])
    errors = numpy.empty([len(event_passes), len(event_passes[0])])
    for i in range(len(event_passes)):
        for j in range(len(event_passes[i])):
            efficiencies[i][j] = (event_passes[i][j] / seen_event_count)
            errors[i][j] = (math.sqrt((event_passes[i][j] / seen_event_count) *
                        (1 - (event_passes[i][j] / seen_event_count)) / seen_event_count))

    for i in range(len(efficiencies)):
        for j in range(len(efficiencies[i])):
            data_list.append({"cmass": mChild, "lifetime": lifetime, "pt": pt_pass_checks[i], "d0": d0_pass_checks[j],
                    "efficiency": efficiencies[i][j], "error": errors[i][j]})

    cf_dict = {"cmass": mChild, "lifetime": lifetime, "events": event_count, "seen": seen_event_count}
    for i in range(len(pt_cuts)):
        cf_dict[pt_cuts[i]] = pt_ok_list[i]
    for i in range(len(d0_cuts)):
        cf_dict[d0_cuts[i]] = d0_ok_list[i]
    for i in range(len(full_cuts)):
        for j in range(len(full_cuts[i])):
            cf_dict[full_cuts[i][j]] = event_passes[i][j]
    cutflow_list.append(cf_dict)

    histogram_list.append({"cmass": mChild, "lifetime": lifetime, "etas": eta_list, "d0s": d0_list, "pt_list": pt_list,
            "energy": energy_list, "decay distance": dec_dist_list,
            "trans decay distance": trans_dec_dist_list})

    print("Done with file ", n, " of 36.")

    if doTest: break

#File Level End =====================================================================================================

data = {"data": data_list, "cutflow": cutflow_list, "hist": histogram_list, "lifetimes": lifetime_list,
        "pts": pt_pass_checks, "d0s": d0_pass_checks, "str_pts": pt_cuts, "str_d0s": d0_cuts, "str_both": full_cuts}
save_name = 'stau_%dtrack_%.1fefficiencies.json'%(track_low_cut,max_track_eff)
if use_slope_eff:
    save_name = 'stau_%dtrack_%.1fefficiencies_slope.json'%(track_low_cut,max_track_eff)
with open(save_name, 'w') as fp:
    json.dump(data, fp)


print("Percent of \"seen\" events: ", 100 * seen_event_count_total / events, "%")

