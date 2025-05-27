from utils import parse_args, create_experiment_dirs, calculate_flops
from model import MobileNet
from train import Train
from data_loader import DataLoader
from summarizer import Summarizer
import tensorflow as tf
import os
import cv2
import random
import numpy as np
from sklearn.model_selection import KFold


def create_folds():
    # Take all data and split it into 5 folds 80/20 (random seed: 42)
    auth_path = "data/everything/authentic"
    spoof_path = "data/everything/spoof"

    auth_files = os.listdir(auth_path)
    spoof_files = os.listdir(spoof_path)

    auth_files = [os.path.join(auth_path, file) for file in auth_files]
    spoof_files = [os.path.join(spoof_path, file) for file in spoof_files]

    auth_images = []
    auth_images_names = []
    spoof_images = []
    spoof_images_names = []

    for auth_file in auth_files:
        if not (auth_file.endswith('.jpg') or auth_file.endswith('.png') or auth_file.endswith('.jpeg') or auth_file.endswith('.webp')):
            continue
        img = cv2.imread(auth_file)
        auth_images.append(img)
        auth_images_names.append(auth_file)

    for spoof_file in spoof_files:
        if not (spoof_file.endswith('.jpg') or spoof_file.endswith('.png') or spoof_file.endswith('.jpeg') or spoof_file.endswith('.webp')):
            continue
        img = cv2.imread(spoof_file)
        spoof_images.append(img)
        spoof_images_names.append(spoof_file)

    # Concatenate
    X = auth_images + spoof_images
    names = auth_images_names + spoof_images_names
    y = np.concatenate([np.zeros(len(auth_images)), np.ones(len(spoof_images))])

    random.seed(42)
    # Create 5 folds

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    i = 0

    # Save the files in the folds to different subdirectories
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train = [X[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for idx, img in enumerate(X_train):
            name = f"image{i}"
            i += 1
            if y_train[idx] == 0:
                cv2.imwrite(f'data/folds/fold{fold}/train/authentic/{name}.png', img)
            else:
                cv2.imwrite(f'data/folds/fold{fold}/train/spoof/{name}.png', img)

        for idx, img in enumerate(X_test):
            name = f"image{i}"
            i += 1
            if y_test[idx] == 0:
                cv2.imwrite(f'data/folds/fold{fold}/test/authentic/{name}.png', img)
            else:
                cv2.imwrite(f'data/folds/fold{fold}/test/spoof/{name}.png', img)


def main():
    # Parse the JSON arguments
    try:
        config_args = parse_args()
    except:
        print("Add a config file using \'--config file_name.json\'")
        exit(1)

    # Create the experiment directories
    _, config_args.summary_dir, config_args.checkpoint_dir = create_experiment_dirs(config_args.experiment_dir)

    # Reset the default Tensorflow graph
    tf.reset_default_graph()

    # Tensorflow specific configuration
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Data loading
    data = DataLoader(config_args.batch_size, config_args.shuffle)
    print("Loading Data...")
    config_args.img_height, config_args.img_width, config_args.num_channels, \
    config_args.train_data_size, config_args.test_data_size = data.load_data()
    print("Data loaded\n\n")

    # Model creation
    print("Building the model...")
    model = MobileNet(config_args)
    print("Model is built successfully\n\n")

    # Summarizer creation
    summarizer = Summarizer(sess, config_args.summary_dir)
    # Train class
    trainer = Train(sess, model, data, summarizer)

    if config_args.to_train:
        try:
            print("Training...")
            trainer.train()
            print("Training Finished\n\n")
        except KeyboardInterrupt:
            trainer.save_model()

    if config_args.to_test:
        print("Final test!")
        trainer.test('val')
        print("Testing Finished\n\n")


if __name__ == '__main__':
    #create_folds()
    main()
