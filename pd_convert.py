import uproot
import pandas as pd
import os
import re
import socket
import math
import numpy as np

JETS_TO_RECORD = 10

process_list = (["tttt", "ttbar", "top", "ttHandZ", "ttWandXY", "EW"])
variables_list = ["FTALepton{}_pt".format(i) for i in [1, 2]] + ["FTALepton{}_eta".format(i) for i in [1, 2]] + ["FTALepton{}_phi".format(i) for i in [1, 2]] 
variables_list = variables_list + ["FTAJet{}__nom_pt".format(i) for i in range(1, JETS_TO_RECORD + 1)] + ["FTAJet{}__nom_eta".format(i) for i in range(1, JETS_TO_RECORD + 1)] + ["FTAJet{}__nom_phi".format(i) for i in range(1, JETS_TO_RECORD + 1)] + ["FTAJet{}__nom_DeepJetB".format(i) for i in range(1, JETS_TO_RECORD + 1)] + ["FTAJet{}__nom_mass".format(i) for i in range(1, JETS_TO_RECORD + 1)]
variables_list = variables_list + ["ST__nom", "HT__nom", "HT2M__nom", "HTRat__nom", "dRbb__nom", "FTALepton_dRll", "H__nom", "H2M__nom", "HTH__nom", "HTb__nom", "nMediumDeepJetB__nom", "nFTAJet__nom"]

if "lxplus" in socket.gethostname(): 
    print("Running on lxplus.")
    ElMu_startdir = "/eos/user/n/nmangane/analysis/Ntupletest/Ntuples/ElMu/2017___"
else:
    print("Not running on lxplus.")
    ElMu_startdir = "ElMu/2017___"

ElMu_files = {
    "EW": "DYJets_DL.root",
    "ttbar": ["ttbb_DL-GF_fr.root", "ttbb_DL_fr.root", "ttbb_DL_nr.root", "ttbb_SL-GF_fr.root", "ttbb_SL_fr.root", "ttbb_SL_nr.root", "ttother_DL-GF_fr.root", "ttother_DL_fr.root", "ttother_DL_nr.root", "ttother_SL-GF_fr.root", "ttother_SL_fr.root", "ttother_SL_nr.root"],
    "top": ["ST_tW.root", "ST_tbarW.root"],
    "ttHandZ": ["ttH.root", "ttZJets.root"],
    "ttWandXY": ["ttWJets.root", "ttHH.root", "tttJ.root", "tttW.root", "ttWH.root", "ttWW.root", "ttWZ.root", "ttZH.root", "ttZZ.root"],
    "tttt": "tttt.root"
}

all_df = pd.DataFrame()

process_dfs = []

for process_num, process in enumerate(process_list):
    process_df = pd.DataFrame()
    print(type(ElMu_files[process]))
    if type(ElMu_files[process])==list:
        for file in ElMu_files[process]:
            print(file)

            try:
                file_tree = uproot.open(ElMu_startdir + file)["Events"]
                file_df = file_tree.pandas.df(variables_list)
                file_df["target"] = process_num
                file_df["target_name"] = process
                file_df["isElEl"] = 0
                file_df["isElMu"] = 1
                file_df["isMuMu"] = 0
                if process_df.empty: process_df = file_df
                else: process_df = pd.concat([process_df, file_df])
            except KeyError:
                print("KeyError occured with file {}".format(ElMu_startdir + file))

            try:
                file_tree = uproot.open(re.sub("ElMu", "MuMu", ElMu_startdir + file))["Events"]
                file_df = file_tree.pandas.df(variables_list)
                file_df["target"] = process_num
                file_df["target_name"] = process
                file_df["isElEl"] = 0
                file_df["isElMu"] = 0
                file_df["isMuMu"] = 1
                if process_df.empty: process_df = file_df
                else: process_df = pd.concat([process_df, file_df])
            except KeyError:
                print("KeyError occured with file {}".format(re.sub("ElMu", "MuMu", ElMu_startdir + file)))
    else:
        print(ElMu_files[process])

        try:
            file_tree = uproot.open(ElMu_startdir + ElMu_files[process])["Events"]
            file_df = file_tree.pandas.df(variables_list)
            file_df["target"] = process_num
            file_df["target_name"] = process
            file_df["isElEl"] = 0
            file_df["isElMu"] = 1
            file_df["isMuMu"] = 0
            if process_df.empty: process_df = file_df
            else: process_df = pd.concat([process_df, file_df])
        except KeyError:
            print("KeyError occured with file {}".format(ElMu_startdir + ElMu_files[process]))

        try:
            file_tree = uproot.open(re.sub("ElMu", "MuMu", ElMu_startdir + ElMu_files[process]))["Events"]
            file_df = file_tree.pandas.df(variables_list)
            file_df["target"] = process_num
            file_df["target_name"] = process
            file_df["isElEl"] = 0
            file_df["isElMu"] = 0
            file_df["isMuMu"] = 1
            if process_df.empty: process_df = file_df
            else: process_df = pd.concat([process_df, file_df])
        except KeyError:
            print("KeyError occured with file {}".format(re.sub("ElMu", "MuMu", ElMu_startdir + ElMu_files[process])))

    process_dfs.append(process_df)

all_df = pd.concat(process_dfs)
all_df.reset_index(drop=True, inplace=True)

print("Finalising")
sphericity = []
for index, event in all_df.iterrows():
    sphericity_matrix = np.zeros([3, 3])
    sumP2 = 0
    for i in range(1, min(event["nFTAJet__nom"], 11)):
        jetpt = event["FTAJet{}__nom_pt".format(i)]
        jeteta = event["FTAJet{}__nom_eta".format(i)]
        jetphi = event["FTAJet{}__nom_phi".format(i)]
        sumP2 += (jetpt*math.cosh(jeteta))**2
        sphericity_matrix[0][0] += (jetpt*math.cos(jetphi))**2
        sphericity_matrix[0][1] += (jetpt*math.cos(jetphi))*(jetpt*math.sin(jetphi))
        sphericity_matrix[0][2] += (jetpt*math.cos(jetphi))*(jetpt*math.sinh(jeteta))
        sphericity_matrix[1][0] += (jetpt*math.cos(jetphi))*(jetpt*math.sin(jetphi))
        sphericity_matrix[1][1] += (jetpt*math.sin(jetphi))**2
        sphericity_matrix[1][2] += (jetpt*math.sin(jetphi))*(jetpt*math.sinh(jeteta))
        sphericity_matrix[2][0] += (jetpt*math.cos(jetphi))*(jetpt*math.sinh(jeteta))
        sphericity_matrix[2][1] += (jetpt*math.sin(jetphi))*(jetpt*math.sinh(jeteta))
        sphericity_matrix[2][2] += (jetpt*math.sinh(jeteta))**2
    sphericity_matrix = sphericity_matrix/sumP2
    eigenvals, _ = np.linalg.eigh(sphericity_matrix)
    eigenvals.sort()
    sphericity.append(sum(eigenvals[0:2])*3/2)
all_df["sphericity"] = sphericity

all_df.to_pickle("train_dataset.p")
