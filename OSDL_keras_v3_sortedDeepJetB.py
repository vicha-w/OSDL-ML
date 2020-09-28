import keras
from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import pandas as pd 
import pickle
from sklearn.model_selection import train_test_split
import argparse
import os

os.system("mkdir v3_sortedDeepJetB")

parser = argparse.ArgumentParser()
parser.add_argument("mode", type = int)
parser.add_argument("dropout", type = float, default = 0.2)

args = parser.parse_args()

JETS_TO_USE = 4

combined_variables_list = ["FTALepton{}_pt".format(i) for i in [1, 2]]
combined_variables_list = combined_variables_list + ["FTAJet{}__nom_pt".format(i) for i in range(1, JETS_TO_USE + 1)] + ["FTAJet{}__nom_DeepJetB_sorted".format(i) for i in range(1, JETS_TO_USE + 1)]
combined_variables_list = combined_variables_list + ["HT__nom", "HTRat__nom", "dRbb__nom", "FTALepton_dRll", "H__nom", "HTH__nom", "HTb__nom", "nMediumDeepJetB__nom", "nFTAJet__nom", "sphericity", "isElMu"]

individual_variables_list = combined_variables_list[:-1]

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

def base_model(input_len):
    input_layer = Input(shape=(input_len,))
    model_layer = BatchNormalization()(input_layer)
    model_layer = Dropout(float(args.dropout))(model_layer)
    model_layer = Dense(50, activation="relu")(model_layer)
    model_layer = Dense(50, activation="relu")(model_layer)
    model_layer = Dense(50, activation="relu")(model_layer)
    model_layer = Dense(1, activation="sigmoid")(model_layer)
    model = Model(input = [input_layer], output = [model_layer])
    model.compile(loss="binary_crossentropy", optimizer="adam")
    return model

if args.mode == 1:
    combined_model = base_model(len(combined_variables_list))
    combined_filepath="v3_sortedDeepJetB/OSDL_keras_v3_sortedDeepJetB_combined-{epoch:03d}" + "-{:.2f}.hdf5".format(args.dropout)
    combined_checkpoint = ModelCheckpoint(combined_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode="auto")
    combined_callbacks_list = [combined_checkpoint]
    combined_history = combined_model.fit(combined_training_input[combined_variables_list], combined_training_input["target_new"], epochs=500, verbose=True, validation_data=(combined_testing_input[combined_variables_list], combined_testing_input["target_new"]), callbacks=combined_callbacks_list)
    with open("v3_sortedDeepJetB/OSDL_keras_v3_sortedDeepJetB_combined_{:.2f}_history.p".format(args.dropout), "wb") as history_pickle:
        pickle.dump(combined_history.history, history_pickle)
    combined_model.save("v3_sortedDeepJetB/OSDL_keras_v3_sortedDeepJetB_combined_{:.2f}.hdf5".format(args.dropout))
elif args.mode == 2:
    ElMu_model = base_model(len(individual_variables_list))
    ElMu_filepath="v3_sortedDeepJetB/OSDL_keras_v3_sortedDeepJetB_ElMu-{epoch:03d}" + "-{:.2f}.hdf5".format(args.dropout)
    ElMu_checkpoint = ModelCheckpoint(ElMu_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode="auto")
    ElMu_callbacks_list = [ElMu_checkpoint]
    ElMu_history = ElMu_model.fit(ElMu_training_input[individual_variables_list], ElMu_training_input["target_new"], epochs=500, verbose=True, validation_data=(ElMu_testing_input[individual_variables_list], ElMu_testing_input["target_new"]), callbacks=ElMu_callbacks_list)
    with open("v3_sortedDeepJetB/OSDL_keras_v3_sortedDeepJetB_ElMu_{:.2f}_history.p".format(args.dropout), "wb") as history_pickle:
        pickle.dump(ElMu_history.history, history_pickle)
    ElMu_model.save("v3_sortedDeepJetB/OSDL_keras_v3_sortedDeepJetB_ElMu_{:.2f}.hdf5".format(args.dropout))
elif args.mode == 3:
    MuMu_model = base_model(len(individual_variables_list))
    MuMu_filepath="v3_sortedDeepJetB/OSDL_keras_v3_sortedDeepJetB_MuMu-{epoch:03d}" + "-{:.2f}.hdf5".format(args.dropout)
    MuMu_checkpoint = ModelCheckpoint(MuMu_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode="auto")
    MuMu_callbacks_list = [MuMu_checkpoint]
    MuMu_history = MuMu_model.fit(MuMu_training_input[individual_variables_list], MuMu_training_input["target_new"], epochs=500, verbose=True, validation_data=(MuMu_testing_input[individual_variables_list], MuMu_testing_input["target_new"]), callbacks=MuMu_callbacks_list)
    with open("v3_sortedDeepJetB/OSDL_keras_v3_sortedDeepJetB_MuMu_{:.2f}_history.p".format(args.dropout), "wb") as history_pickle:
        pickle.dump(MuMu_history.history, history_pickle)
    MuMu_model.save("v3_sortedDeepJetB/OSDL_keras_v3_sortedDeepJetB_MuMu_{:.2f}.hdf5".format(args.dropout))