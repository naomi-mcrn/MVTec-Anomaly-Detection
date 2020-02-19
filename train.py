#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 19:46:17 2019

@author: adnene33
This Script is meant to train on a single object category of MVTec, unlike train_mvtec.py
"""
import os
import sys

import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import custom_loss_functions
import models
import utils
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

import requests
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import datetime
import csv
import pandas as pd
import json

import argparse


def main(args):
    # Get training data setup
    directory = args.directory
    train_data_dir = os.path.join(directory, "train")
    nb_training_images_aug = args.images
    batch_size = args.batch
    validation_split = 0.1

    architecture = args.architecture
    comment = args.comment

    # NEW TRAINING
    if args.command == "new":
        loss = args.loss.upper()

        if loss == "SSIM":
            channels = 1
            color_mode = "grayscale"
        else:
            channels = 3
            color_mode = "rgb"

        # Build model
        model = models.build_model(architecture, channels)

        # specify model name and directory to save model to
        now = datetime.datetime.now()
        save_dir = os.path.join(
            os.getcwd(), "saved_models", loss, now.strftime("%d-%m-%Y_%H:%M:%S")
        )
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_name = "CAE_" + architecture + "_b{}".format(batch_size)
        model_path = os.path.join(save_dir, model_name + ".h5")

        # specify logging directory for tensorboard visualization
        log_dir = os.path.join(save_dir, "logs")
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        learning_rate = 2e-4  # 0.001 0.002

        # set loss function, optimizer, metric and callbacks
        if loss == "SSIM":
            loss_function = custom_loss_functions.ssim

        elif loss == "MSSIM":
            loss_function = custom_loss_functions.mssim

        elif loss == "L2":
            loss_function = custom_loss_functions.l2

        elif loss == "MSE":
            loss_function = "mean_squared_error"

        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, decay=1e-5
        )

        model.compile(
            loss=loss_function,
            optimizer=optimizer,
            metrics=[loss_function, "mean_squared_error"],
        )

        # callbacks
        early_stopping_cb = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=12, mode="min", verbose=1,
        )
        checkpoint_cb = keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            # monitor="val_loss",
            verbose=1,
            save_best_only=False,  # True
            save_weights_only=False,
            period=1,
        )
        tensorboard_cb = keras.callbacks.TensorBoard(
            log_dir=log_dir, write_graph=True, update_freq="epoch"
        )

    # RESUME TRAINING
    # elif args.command == "resume":
    #     model_path = args.model

    #     # load model
    #     model, train_setup, _ = utils.load_SavedModel(model_path)
    #     color_mode = train_setup["color_mode"]
    #     validation_split = train_setup["validation_split"]

    # =============================== TRAINING =================================

    if architecture == "mvtec":
        rescale = 1.0 / 255
        shape = (256, 256)
        preprocessing_function = None
        preprocessing = None
    elif architecture == "resnet":
        rescale = None
        shape = (299, 299)
        preprocessing_function = keras.applications.inception_resnet_v2.preprocess_input
        preprocessing = "keras.applications.inception_resnet_v2.preprocess_input"
    elif architecture == "nasnet":
        rescale = None
        shape = (224, 224)
        preprocessing_function = keras.applications.nasnet.preprocess_input
        preprocessing = "keras.applications.inception_resnet_v2.preprocess_input"
        pass

    print("Using Keras's flow_from_directory method.")
    # This will do preprocessing and realtime data augmentation:
    train_datagen = ImageDataGenerator(
        # randomly rotate images in the range (degrees, 0 to 180)
        rotation_range=10,
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.05,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.05,
        # set mode for filling points outside the input boundaries
        fill_mode="nearest",
        # value used for fill_mode = "constant"
        cval=0.0,
        # randomly change brightness (darker < 1 < brighter)
        brightness_range=[0.95, 1.05],
        # set rescaling factor (applied before any other transformation)
        rescale=rescale,
        # set function that will be applied on each input
        preprocessing_function=preprocessing_function,
        # image data format, either "channels_first" or "channels_last"
        data_format="channels_last",
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=validation_split,
    )

    # For validation dataset, only rescaling
    validation_datagen = ImageDataGenerator(
        rescale=rescale,
        data_format="channels_last",
        validation_split=validation_split,
        preprocessing_function=preprocessing_function,
    )

    # Generate training batches with datagen.flow_from_directory()
    train_generator = train_datagen.flow_from_directory(
        directory=train_data_dir,
        target_size=shape,
        color_mode=color_mode,
        batch_size=batch_size,
        class_mode="input",
        subset="training",
        shuffle=True,
    )

    # Generate validation batches with datagen.flow_from_directory()
    validation_generator = validation_datagen.flow_from_directory(
        directory=train_data_dir,
        target_size=shape,
        color_mode=color_mode,
        batch_size=batch_size,  # 1
        class_mode="input",
        subset="validation",
        shuffle=True,
    )

    # Print command to paste in broser for visualizing in Tensorboard
    print("\ntensorboard --logdir={}\n".format(log_dir))

    # Try with this number of epochs
    epochs = nb_training_images_aug // train_generator.samples

    # Fit the model on the batches generated by datagen.flow_from_directory()
    history = model.fit_generator(
        generator=train_generator,
        epochs=epochs,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        # callbacks=[checkpoint_cb],
    )

    # Save model
    tf.keras.models.save_model(
        model, model_path, include_optimizer=True, save_format="h5"
    )
    print("Saved trained model at %s " % model_path)

    # save training history
    hist_df = pd.DataFrame(history.history)
    hist_csv_file = os.path.join(save_dir, "history.csv")
    with open(hist_csv_file, mode="w") as f:
        hist_df.to_csv(f)
    print("Saved training history at %s " % hist_csv_file)

    epochs_trained = utils.get_epochs_trained(history.history)

    # save training setup and model configuration
    if args.command == "new":
        setup = {
            "data_setup": {
                "directory": directory,
                "nb_training_images": train_generator.samples,
                "nb_validation_images": validation_generator.samples,
            },
            "preprocessing_setup": {
                "rescale": rescale,
                "shape": shape,
                "preprocessing": preprocessing
            },
            "train_setup": {
                "architecture": architecture,
                "nb_training_images_aug": nb_training_images_aug,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "loss": loss,
                "color_mode": color_mode,
                "channels": channels,
                "validation_split": validation_split,
                "epochs_trained": epochs_trained,
            },
            "comment": comment,
        }

    # elif args.command == "resume":
    #     train_setup = {
    #         "directory": directory,
    #         "epochs": epochs,
    #         "batch_size": batch_size,
    #         "loss": loss,
    #         "color_mode": color_mode,
    #         "channels": channels,
    #         "validation_split": validation_split,
    #         "path_to_previous_model": args.model,
    #     }

    with open(os.path.join(save_dir, "setup.json"), "w") as json_file:
        json.dump(setup, json_file)


# create top level parser
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(
    help="help for subcommand", title="commands", dest="command"
)

# create the subparser to begin training a new model
parser_new_training = subparsers.add_parser("new")

parser_new_training.add_argument(
    "-d", "--directory", type=str, required=True, metavar="", help="training directory"
)

parser_new_training.add_argument(
    "-a",
    "--architecture",
    type=str,
    required=True,
    metavar="",
    choices=["mvtec", "resnet", "nasnet"],
    help="model to use in training",
)

parser_new_training.add_argument(
    "-i",
    "--images",
    type=int,
    default=10000,
    metavar="",
    help="number of training images",
)
parser_new_training.add_argument(
    "-b", "--batch", type=int, required=True, metavar="", help="batch size"
)
parser_new_training.add_argument(
    "-l",
    "--loss",
    type=str,
    required=True,
    metavar="",
    choices=["mssim", "ssim", "l2", "mse"],
    help="loss function used during training",
)

parser_new_training.add_argument(
    "-c", "--comment", type=str, help="write comment regarding training")

# create the subparser to resume the training of an existing model
parser_resume_training = subparsers.add_parser("resume")
parser_resume_training.add_argument(
    "-p", "--path", type=str, required=True, metavar="", help="path to existing model"
)
parser_resume_training.add_argument(
    "-e",
    "--epochs",
    type=int,
    required=True,
    metavar="",
    help="number of training epochs",
)
parser_resume_training.add_argument(
    "-b", "--batch", type=int, required=True, metavar="", help="batch size"
)

args = parser.parse_args()

if __name__ == "__main__":
    if tf.test.is_gpu_available():
        print("GPU was detected.")
    else:
        print("No GPU was detected. CNNs can be very slow without a GPU.")
    print("Tensorflow version: {}".format(tf.__version__))
    print("Keras version: {}".format(keras.__version__))
    main(args)

# Examples to initiate training

# python3 train.py new -d mvtec/capsule -a mvtec -b 24 -l mse
# python3 train.py new -d mvtec/capsule -a mvtec -b 24 -l mssim
# python3 train.py new -d mvtec/capsule -a mvtec -b 24 -l l2

# python3 train.py new -d mvtec/capsule -a resnet -b 24 -l mse
# python3 train.py new -d mvtec/capsule -a resnet -b 24 -l mssim
# python3 train.py new -d mvtec/capsule -a resnet -b 24 -l l2