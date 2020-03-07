from skimage.util import img_as_ubyte
from validate import label_images as label_images
from validate import threshold_images as threshold_images
import argparse
import json
import pandas as pd
import csv
import datetime
import os
import sys
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import custom_loss_functions
import utils
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from numpy import expand_dims
import scipy.stats

import requests
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# plt.style.use("seaborn-darkgrid")

# =========================================================================
# import visualization functions


def plot_img_at_index(X, index):
    _, _, _, channels = X.shape
    plt.figure()
    if channels == 1:
        plt.imshow(X[index, :, :, 0], cmap=plt.cm.gray)
    elif channels == 3:
        plt.imshow(X[index, :, :, 0])
    plt.show()


def plot_img(img):
    ndims = len(img.shape)
    plt.figure()
    if ndims == 2:
        plt.imshow(img, cmap=plt.cm.gray)
    elif ndims == 3:
        _, _, channels = img.shape
        if channels == 3:
            plt.imshow(img)
        else:
            plt.imshow(img[:, :, 0], cmap=plt.cm.gray)
    # plt.show()
    return plt


def hist_image(img):
    img_1d = img.flatten()
    plt.figure()
    plt.hist(img_1d, bins=200, density=True,
             stacked=True, label="image histogram")
    # plot pdf
    mu = img_1d.mean()
    sigma = img_1d.std()
    minimum = np.amin(img_1d)
    maximum = np.amax(img_1d)
    X = np.linspace(start=minimum, stop=maximum, num=400, endpoint=True)
    pdf_x = [scipy.stats.norm(mu, sigma).pdf(x) for x in X]
    plt.plot(X, pdf_x, label="pixel distribution")
    plt.legend()
    plt.show()


# =========================================================================

# import validation functions


def main(args):
    model_path = args.path
    save = args.save
    img_val = args.val
    img_test = args.test

    # load model, setup and history
    model, setup, history = utils.load_model_HDF5(model_path)

    # data setup
    directory = setup["data_setup"]["directory"]
    val_data_dir = os.path.join(directory, "train")

    nb_training_images = setup["data_setup"]["nb_training_images"]
    nb_validation_images = setup["data_setup"]["nb_validation_images"]

    # preprocessing_setup
    rescale = setup["preprocessing_setup"]["rescale"]
    shape = setup["preprocessing_setup"]["shape"]
    preprocessing = setup["preprocessing_setup"]["preprocessing"]

    # train_setup
    color_mode = setup["train_setup"]["color_mode"]
    learning_rate = setup["train_setup"]["learning_rate"]
    decay = setup["train_setup"]["decay"]
    epochs_trained = setup["train_setup"]["epochs_trained"]
    nb_training_images_aug = setup["train_setup"]["nb_training_images_aug"]
    epochs = setup["train_setup"]["epochs"]
    batch_size = setup["train_setup"]["batch_size"]
    channels = setup["train_setup"]["channels"]
    validation_split = setup["train_setup"]["validation_split"]
    architecture = setup["train_setup"]["architecture"]
    loss = setup["train_setup"]["loss"]

    tag = setup["tag"]

    # create directory to save results
    model_dir_name = os.path.basename(str(Path(model_path).parent))
    save_dir = os.path.join(os.getcwd(), "results", model_dir_name, "finetune")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # This will do preprocessing
    if architecture in ["mvtec", "mvtec2"]:
        preprocessing_function = None
    elif architecture == "resnet":
        preprocessing_function = keras.applications.inception_resnet_v2.preprocess_input
    elif architecture == "nasnet":
        preprocessing_function = keras.applications.nasnet.preprocess_input

    # same preprocessing as in training
    validation_datagen = ImageDataGenerator(
        rescale=rescale,
        data_format="channels_last",
        validation_split=validation_split,
        preprocessing_function=preprocessing_function,
    )

    # retrieve preprocessed validation images as a numpy array
    validation_generator = validation_datagen.flow_from_directory(
        directory=val_data_dir,
        target_size=shape,
        color_mode=color_mode,
        batch_size=nb_validation_images,
        shuffle=False,
        class_mode="input",
        subset="validation",
    )
    imgs_val_input = validation_generator.next()[0]

    # get reconstructed images (i.e predictions) on validation dataset
    imgs_val_pred = model.predict(imgs_val_input)

    # converts rgb to grayscale
    if color_mode == "rgb":
        imgs_val_input = tf.image.rgb_to_grayscale(imgs_val_input)
        imgs_val_pred = tf.image.rgb_to_grayscale(imgs_val_pred)

    # compute residual maps on validation dataset
    resmaps_val = imgs_val_input - imgs_val_pred

    if save:
        utils.save_np(imgs_val_input, save_dir, "imgs_val_input.npy")
        utils.save_np(imgs_val_pred, save_dir, "imgs_val_pred.npy")
        utils.save_np(resmaps_val, save_dir, "resmaps_val.npy")

    # ============== PLOT A SAMPLE VALIDATION IMAGE =======================
    if img_val != None:

        # compute index of text image
        index_val = validation_generator.filenames.index(img_val)

        fig = utils.plot_input_pred_resmaps_val(
            imgs_val_input, imgs_val_pred, resmaps_val, index_val)
        fig.savefig(os.path.join(save_dir, "val_plots.png"))

    # ===================================================================

    # Convert to 8-bit unsigned int
    # (unnecessary if working exclusively with scikit image, see .img_as_float())
    resmaps_val = img_as_ubyte(resmaps_val)

    nb_regions = []
    mean_area_size = []
    std_area_size = []
    threshold_max = np.amax(resmaps_val)

    for threshold in range(threshold_max):
        # threshold residual maps
        resmaps_th = threshold_images(resmaps_val, threshold)

        # compute connected components
        resmaps_labeled, areas_all = label_images(resmaps_th)

        # check if area of largest anomalous region is below the minimum area
        areas_all_flat = [item for sublist in areas_all for item in sublist]
        nb_regions.append(len(areas_all_flat))
        mean_area_size.append(np.mean(areas_all_flat))
        std_area_size.append(np.std(areas_all_flat))

        print("current threshold = {}".format(threshold))

    df_out = pd.DataFrame.from_dict(
        {
            "threshold": list(range(np.amax(resmaps_val))),
            "nb_regions": nb_regions,
            "mean_area_size": mean_area_size,
            "std_area_size": std_area_size,
        }
    )
    df_out.plot(
        x="threshold",
        y=["mean_area_size", "std_area_size"],
        xticks=np.arange(start=0, stop=threshold_max, step=5, dtype=int),
        title="mean and standard deviation of areas with increasing thresholds",
        figsize=(8, 5),
    )
    plt.savefig(os.path.join(save_dir, "stat_area.pdf"))
    # plt.show()
    df_out.plot(
        x="threshold",
        y=["nb_regions"],
        xticks=np.arange(start=0, stop=threshold_max, step=5, dtype=int),
        title="number of regions with increasing thresholds",
        figsize=(8, 5),
    )
    plt.savefig(os.path.join(save_dir, "nb_of_regions.pdf"))
    # plt.show()

    # ===================================================================
    # ===================================================================

    # COMMENT -----------------------------------------------------------
    # resmaps_val = np.load(
    #     "results/25-02-2020_08:54:06/validation/02-03-2020_11:13:25/resmaps_val.npy",
    #     allow_pickle=True,
    # )

    # Convert to 8-bit unsigned int
    # (unnecessary if working exclusively with scikit image, see .img_as_float())
    # resmaps_val = img_as_ubyte(resmaps_val)
    # -------------------------------------------------------------------

    counts = []
    nb_bins = 200
    max_pixel_value = np.amax(resmaps_val)
    thresholds_to_plot = 9
    step = max_pixel_value // thresholds_to_plot
    # areas_per_threshold = np.zeros(shape=(nb_bins, max_pixel_value))

    for threshold in range(max_pixel_value):
        # threshold residual maps
        resmaps_th = threshold_images(resmaps_val, threshold)

        # compute connected components
        resmaps_labeled, areas_all = label_images(resmaps_th)

        # flatten area
        areas_all_flat = [item for sublist in areas_all for item in sublist]

        if threshold == 0:
            areas_all_flat.sort(reverse=True)
            # min_area = areas_all_flat[-1]
            max_area = areas_all_flat[0]

            fig3 = plt.figure(num=3, figsize=(12, 8))
            count, bins, ignored = plt.hist(
                areas_all_flat,
                range=(0, max_area / 4),
                bins=nb_bins,  # max_area - min_area 200
                density=False,
            )
            plt.xticks(
                np.arange(start=0, stop=max_area / 10, step=10)
            )  # stop=max_area / 10
            # plt.legend()
            plt.title(
                "Distribution of anomaly areas' sizes for validation ResMaps with Threshold = 0"
            )
            plt.xlim([0, max_area / 10])  # max_area / 10, 100
            plt.xlabel("area size in pixel")
            plt.ylabel("count")
            # plt.show()
            fig3.savefig(os.path.join(save_dir, "distr_area_th_0.pdf"))

        else:
            fig4 = plt.figure(num=4, figsize=(12, 5))
            if divmod(threshold, step)[1] == 0:
                count, edges = np.histogram(
                    areas_all_flat,
                    range=(0, max_area / 4),
                    bins=nb_bins,  # max_area - min_area 200
                    density=False,
                )
                bins_middle = edges[:-1] + (edges[0] + edges[1]) / 2
                plt.plot(
                    bins_middle,
                    count,
                    linestyle="-",
                    linewidth=0.5,
                    marker="o",
                    markersize=0.5,
                    label="threshold = {}".format(threshold),
                )
    plt.title(
        "Distribution of anomaly areas' sizes for validation ResMaps with various Thresholds"
    )
    plt.xticks(np.arange(start=0, stop=max_area, step=2))  # max_area / 10
    plt.legend()
    plt.xlim([0, 100])  # 20
    plt.xlabel("area size in pixel")
    plt.ylabel("count")
    # plt.show()
    fig4.savefig(os.path.join(save_dir, "distr_area_th_multiple.pdf"))

    # ============== PLOT A SAMPLE TEST IMAGE =======================
    if img_test != None:
        test_data_dir = os.path.join(directory, "test")
        total_number = utils.get_total_number_test_images(test_data_dir)

        test_datagen = ImageDataGenerator(
            rescale=rescale,
            data_format="channels_last",
            preprocessing_function=preprocessing_function,
        )

        # retrieve preprocessed test images as a numpy array
        test_generator = test_datagen.flow_from_directory(
            directory=test_data_dir,
            target_size=shape,
            color_mode=color_mode,
            batch_size=total_number,
            shuffle=False,
            class_mode="input",
        )
        imgs_test_input = test_generator.next()[0]

        # predict on test images
        imgs_test_pred = model.predict(imgs_test_input)

        # compute residual maps on test set
        resmaps_test = imgs_test_input - imgs_test_pred

        # compute index of text image
        index_test = test_generator.filenames.index(img_test)

        # save three images
        fig = utils.plot_input_pred_resmaps_test(
            imgs_test_input, imgs_test_pred, resmaps_test, index_test)
        fig.savefig(os.path.join(save_dir, "test_plots.png"))

    # ===================================================================


if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--path", type=str, required=True, metavar="", help="path to saved model"
    )
    parser.add_argument(
        "-s",
        "--save",
        type=bool,
        required=False,
        default=False,
        metavar="",
        help="save inputs, predictions and reconstructions of validation dataset",
    )
    parser.add_argument(
        "-v",
        "--val",
        type=str,
        default=None,
        metavar="",
        help="path to sample test image relative to validation directory for visualization"
    )
    parser.add_argument(
        "-t",
        "--test",
        type=str,
        default=None,
        metavar="",
        help="path to sample test image relative to test directory for visualization"
    )
    args = parser.parse_args()

    main(args)


# model_path = "saved_models/MSE/25-02-2020_08:54:06/CAE_mvtec2_b12.h5"


# python3 finetune.py -p saved_models/MSE/25-02-2020_08:54:06/CAE_mvtec2_b12.h5 -v "good/000.png" -t "poke/000.png"
# python3 finetune.py -p saved_models/MSE/07-03-2020_21:36:53/CAE_resnet_b12.h5 -v "good/000.png" -t "poke/000.png"
