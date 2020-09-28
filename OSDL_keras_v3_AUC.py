import keras
from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import pandas as pd 
import pickle
from sklearn.model_selection import train_test_split
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("model_path")
parser.add_argument("train_mode", type=int)
parser.add_argument("--is_sorted", action="store_true")
parser.add_argument("--is_tanh", action="store_true")

args = parser.parse_args()

JETS_TO_USE = 4

combined_variables_list = ["FTALepton{}_pt".format(i) for i in [1, 2]]
combined_variables_list = combined_variables_list + ["FTAJet{}__nom_pt".format(i) for i in range(1, JETS_TO_USE + 1)] + ["FTAJet{}__nom_DeepJetB".format(i) for i in range(1, JETS_TO_USE + 1)]
combined_variables_list = combined_variables_list + ["HT__nom", "HTRat__nom", "dRbb__nom", "FTALepton_dRll", "H__nom", "HTH__nom", "HTb__nom", "nMediumDeepJetB__nom", "nFTAJet__nom", "sphericity", "isElMu"]

individual_variables_list = combined_variables_list[:-1]

combined_variables_list_sortedDeepJetB = ["FTALepton{}_pt".format(i) for i in [1, 2]]
combined_variables_list_sortedDeepJetB = combined_variables_list_sortedDeepJetB + ["FTAJet{}__nom_pt".format(i) for i in range(1, JETS_TO_USE + 1)] + ["FTAJet{}__nom_DeepJetB_sorted".format(i) for i in range(1, JETS_TO_USE + 1)]
combined_variables_list_sortedDeepJetB = combined_variables_list_sortedDeepJetB + ["HT__nom", "HTRat__nom", "dRbb__nom", "FTALepton_dRll", "H__nom", "HTH__nom", "HTb__nom", "nMediumDeepJetB__nom", "nFTAJet__nom", "sphericity", "isElMu"]

individual_variables_list_sortedDeepJetB = combined_variables_list_sortedDeepJetB[:-1]

input_data = pd.read_pickle("train_dataset.p")
input_data = input_data[input_data["target"] <= 1]
input_data["target_new"] = 1 - input_data["target"]
input_data = input_data.sample(frac = 1., random_state=42)

ElMu_input_data = input_data[input_data["isElMu"] == 1]
MuMu_input_data = input_data[input_data["isMuMu"] == 1]

combined_training_size = 35000
combined_training_input = pd.concat([input_data[input_data["target"]==target].iloc[:combined_training_size] for target in range(2)])
combined_testing_input  = pd.concat([input_data[input_data["target"]==target].iloc[combined_training_size:] for target in range(2)])

ElMu_training_size = 23000
ElMu_training_input = pd.concat([ElMu_input_data[ElMu_input_data["target"]==target].iloc[:ElMu_training_size] for target in range(2)])
ElMu_testing_input  = pd.concat([ElMu_input_data[ElMu_input_data["target"]==target].iloc[ElMu_training_size:] for target in range(2)])

MuMu_training_size = 11500
MuMu_training_input = pd.concat([MuMu_input_data[MuMu_input_data["target"]==target].iloc[:MuMu_training_size] for target in range(2)])
MuMu_testing_input  = pd.concat([MuMu_input_data[MuMu_input_data["target"]==target].iloc[MuMu_training_size:] for target in range(2)])

model = keras.models.load_model(args.model_path)
variables_list_used = []
if args.train_mode == 1:
    if args.is_sorted: variables_list_used = combined_variables_list_sortedDeepJetB
    else: variables_list_used = combined_variables_list
else:
    if args.is_sorted: variables_list_used = individual_variables_list_sortedDeepJetB
    else: variables_list_used = individual_variables_list

if args.train_mode == 1:
    train_ttbar_output = model.predict(combined_training_input[combined_training_input["target"]==1][variables_list_used])
    train_tttt_output  = model.predict(combined_training_input[combined_training_input["target"]==0][variables_list_used])
    test_ttbar_output = model.predict(combined_testing_input[combined_testing_input["target"]==1][variables_list_used])
    test_tttt_output  = model.predict(combined_testing_input[combined_testing_input["target"]==0][variables_list_used])
elif args.train_mode == 2:
    train_ttbar_output = model.predict(ElMu_training_input[ElMu_training_input["target"]==1][variables_list_used])
    train_tttt_output  = model.predict(ElMu_training_input[ElMu_training_input["target"]==0][variables_list_used])
    test_ttbar_output = model.predict(ElMu_testing_input[ElMu_testing_input["target"]==1][variables_list_used])
    test_tttt_output  = model.predict(ElMu_testing_input[ElMu_testing_input["target"]==0][variables_list_used])
elif args.train_mode == 3:
    train_ttbar_output = model.predict(MuMu_training_input[MuMu_training_input["target"]==1][variables_list_used])
    train_tttt_output  = model.predict(MuMu_training_input[MuMu_training_input["target"]==0][variables_list_used])
    test_ttbar_output = model.predict(MuMu_testing_input[MuMu_testing_input["target"]==1][variables_list_used])
    test_tttt_output  = model.predict(MuMu_testing_input[MuMu_testing_input["target"]==0][variables_list_used])

train_sig_acceptance_list = [0]
test_sig_acceptance_list = [0]

train_ttbar_sorted = np.sort(train_ttbar_output, axis=None)
train_tttt_sorted  = np.sort(train_tttt_output, axis=None)
test_ttbar_sorted  = np.sort(test_ttbar_output, axis=None)
test_tttt_sorted   = np.sort(test_tttt_output, axis=None)
for i in range(1, 50):
    train_bg_cut = train_ttbar_sorted[int(len(train_ttbar_sorted)*(1-i/50))]
    train_sig_acceptance = np.count_nonzero(train_tttt_sorted > train_bg_cut)/len(train_tttt_sorted)
    train_sig_acceptance_list.append(train_sig_acceptance)
    test_bg_cut = test_ttbar_sorted[int(len(test_ttbar_sorted)*(1-i/50))]
    test_sig_acceptance = np.count_nonzero(test_tttt_sorted > test_bg_cut)/len(test_tttt_sorted)
    test_sig_acceptance_list.append(test_sig_acceptance)

train_sig_acceptance_list.append(1)
test_sig_acceptance_list.append(1)

train_AUC = 0
test_AUC  = 0
for i in range(50):
    train_AUC += (train_sig_acceptance_list[i] + train_sig_acceptance_list[i+1])*(1/50)/2
    test_AUC += (test_sig_acceptance_list[i] + test_sig_acceptance_list[i+1])*(1/50)/2

out_filename = ""
if (not args.is_sorted) and (not args.is_tanh):
    out_filename = "v3/AUC_summary.txt"
if args.is_sorted and (not args.is_tanh):
    out_filename = "v3_sortedDeepJetB/AUC_summary.txt"
if (not args.is_sorted) and args.is_tanh:
    out_filename = "v3_tanh/AUC_summary.txt"
if args.is_sorted and args.is_tanh:
    out_filename = "v3_sortedDeepJetB_tanh/AUC_summary.txt"

with open(out_filename, "a") as AUC_file:
    AUC_file.write(args.model_path + ' ' + str(args.train_mode) + ' train ' + str(train_sig_acceptance_list) + '\n')
    AUC_file.write(args.model_path + ' ' + str(args.train_mode) + ' test ' + str(test_sig_acceptance_list) + '\n')