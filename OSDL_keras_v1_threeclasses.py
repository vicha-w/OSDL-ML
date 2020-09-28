import keras
from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import pandas as pd 
import pickle
from sklearn.model_selection import train_test_split

JETS_TO_USE = 4

variables_list = ["FTALepton{}_pt".format(i) for i in [1, 2]] + ["FTALepton{}_eta".format(i) for i in [1, 2]] + ["FTALepton{}_phi".format(i) for i in [1, 2]] 
variables_list = variables_list + ["FTAJet{}__nom_pt".format(i) for i in range(1, JETS_TO_USE + 1)] + ["FTAJet{}__nom_eta".format(i) for i in range(1, JETS_TO_USE + 1)] + ["FTAJet{}__nom_phi".format(i) for i in range(1, JETS_TO_USE + 1)] + ["FTAJet{}__nom_DeepJetB".format(i) for i in range(1, JETS_TO_USE + 1)] + ["FTAJet{}__nom_mass".format(i) for i in range(1, JETS_TO_USE + 1)]
variables_list = variables_list + ["ST__nom", "HT__nom", "HT2M__nom", "HTRat__nom", "dRbb__nom", "FTALepton_dRll", "H__nom", "H2M__nom", "HTH__nom", "HTb__nom", "nTightDeepJetB__nom", "nFTAJet__nom", "sphericity"]

input_data = pd.read_pickle("train_dataset.p")
input_data["target_new"] = 0*(input_data["target"] == 0) + 1*(input_data["target"] == 1) + 2*(input_data["target"] >= 2)
input_data = input_data.sample(frac = 1., random_state=42)

training_size = 35000
training_input = pd.concat([input_data[input_data["target_new"]==target].iloc[:training_size] for target in range(3)])
training_target = keras.utils.to_categorical(training_input["target_new"], num_classes=3)
testing_input = pd.concat([input_data[input_data["target_new"]==target].iloc[training_size:] for target in range(3)])
testing_target = keras.utils.to_categorical(testing_input["target_new"], num_classes=3)

inputs = Input(shape=(len(variables_list),))
model_layer = BatchNormalization()(inputs)
model_layer = Dense(100, activation="relu")(model_layer)
model_layer = Dense(100, activation="relu")(model_layer)
model_layer = Dense(100, activation="relu")(model_layer)
model_layer = Dense(3, activation="softmax")(model_layer)

model = Model(input = [inputs], output = [model_layer])
model.compile(loss="categorical_crossentropy", optimizer="sgd")

filepath="OSDL_keras_v1_threeclasses-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode="auto")
callbacks_list = [checkpoint]

history = model.fit(training_input[variables_list], training_target, epochs=100, verbose=True, validation_data=(testing_input[variables_list], testing_target), callbacks=callbacks_list)
with open("OSDL_keras_v1_threeclasses_history.p", "wb") as history_pickle:
    pickle.dump(history.history, history_pickle)
model.save("OSDL_model_v1_threeclasses.hdf5")