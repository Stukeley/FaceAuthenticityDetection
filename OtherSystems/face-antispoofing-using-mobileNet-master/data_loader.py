import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split


class DataLoader:
    """Data Loader class"""

    def __init__(self, batch_size, shuffle=False):
        self.X_train = None
        self.y_train = None
        self.img_mean = None
        self.train_data_len = 0

        self.X_val = None
        self.y_val = None
        self.val_data_len = 0

        self.X_test = None
        self.y_test = None
        self.test_data_len = 0

        self.shuffle = shuffle
        self.batch_size = batch_size

    def load_data(self):

        train_data, train_labels = get_train_data()
        test_data, test_labels = get_test_data()

        self.X_train, self.X_val, self.y_train, self.y_val = train_data, test_data, train_labels, test_labels

        #self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(train_data, train_labels, test_size=0.20,
        #                                                                      random_state=61185)

        print('self.X_train.shape')
        print(self.X_train.shape)
        print('self.y_train.shape')
        print(self.y_train.shape)

        self.train_data_len = self.X_train.shape[0]
        self.val_data_len = self.X_val.shape[0]
        img_height = 224
        img_width = 224
        num_channels = 3
        return img_height, img_width, num_channels, self.train_data_len, self.val_data_len

    def generate_batch(self, type='train'):
        """Generate batch from X_train/X_test and y_train/y_test using a python DataGenerator"""
        if type == 'train':
            # Training time!
            new_epoch = True
            start_idx = 0
            mask = None
            while True:
                if new_epoch:
                    start_idx = 0
                    if self.shuffle:
                        mask = np.random.choice(self.train_data_len, self.train_data_len, replace=False)
                    else:
                        mask = np.arange(self.train_data_len)
                    new_epoch = False

                # Batch mask selection
                X_batch = self.X_train[mask[start_idx:start_idx + self.batch_size]]
                y_batch = self.y_train[mask[start_idx:start_idx + self.batch_size]]
                start_idx += self.batch_size

                # Reset everything after the end of an epoch
                if start_idx >= self.train_data_len:
                    new_epoch = True
                    mask = None
                yield X_batch, y_batch
        elif type == 'test':
            # Testing time!
            start_idx = 0
            while True:
                # Batch mask selection
                X_batch = self.X_test[start_idx:start_idx + self.batch_size]
                y_batch = self.y_test[start_idx:start_idx + self.batch_size]
                start_idx += self.batch_size

                # Reset everything
                if start_idx >= self.test_data_len:
                    start_idx = 0
                yield X_batch, y_batch
        elif type == 'val':
            # Testing time!
            start_idx = 0
            while True:
                # Batch mask selection
                X_batch = self.X_val[start_idx:start_idx + self.batch_size]
                y_batch = self.y_val[start_idx:start_idx + self.batch_size]
                start_idx += self.batch_size

                # Reset everything
                if start_idx >= self.val_data_len:
                    start_idx = 0
                yield X_batch, y_batch
        else:
            raise ValueError("Please select a type from \'train\', \'val\', or \'test\'")


def get_train_data():
    all_image_list = []
    all_label_list = []
    real_dir = 'data/folds/fold4/train/authentic'
    fake_dir = 'data/folds/fold4/train/spoof'
    # load the real image
    count_real = 0
    count_fake = 0
    for file_name in os.listdir(real_dir):
        if not (file_name.endswith('.png') or file_name.endswith('.jpg') or file_name.endswith('.jpeg')
                or file_name.endswith('.webp')):
            continue
        # all_image_list.append(plt.imread(real_dir + '/' + sub_dir + '/' + file_name))
        image = cv2.imread(real_dir + '/' + file_name, cv2.IMREAD_COLOR)
        all_image_list.append(cv2.resize(image, (224, 224)))
        all_label_list.append(1)
        count_real += 1

    for fake_file_name in os.listdir(fake_dir):
        if not (fake_file_name.endswith('.png') or fake_file_name.endswith('.jpg') or fake_file_name.endswith('.jpeg')
                or fake_file_name.endswith('.webp')):
            continue
        image = cv2.imread(fake_dir + '/' + fake_file_name, cv2.IMREAD_COLOR)
        all_image_list.append(cv2.resize(image, (224, 224)))
        all_label_list.append(0)
        count_fake += 1

    print('There are %d real images\nThere are %d fake images' % (count_real, count_fake))
    all_image_list = np.array(all_image_list).reshape((len(all_image_list), 224, 224, 3))

    return all_image_list, np.array([label for label in all_label_list])

def get_test_data():
    all_image_list = []
    all_label_list = []
    real_dir = 'data/folds/fold4/test/authentic'
    fake_dir = 'data/folds/fold4/test/spoof'

    # load the real image
    count_real = 0
    count_fake = 0
    for file_name in os.listdir(real_dir):
        if not (file_name.endswith('.png') or file_name.endswith('.jpg') or file_name.endswith('.jpeg')
                or file_name.endswith('.webp')):
            continue
        # all_image_list.append(plt.imread(real_dir + '/' + sub_dir + '/' + file_name))
        image = cv2.imread(real_dir + '/' + file_name, cv2.IMREAD_COLOR)
        all_image_list.append(cv2.resize(image, (224, 224)))
        all_label_list.append(1)
        count_real += 1

    for fake_file_name in os.listdir(fake_dir):
        if not (fake_file_name.endswith('.png') or fake_file_name.endswith('.jpg') or fake_file_name.endswith('.jpeg')
                or fake_file_name.endswith('.webp')):
            continue
        image = cv2.imread(fake_dir + '/' + fake_file_name, cv2.IMREAD_COLOR)
        all_image_list.append(cv2.resize(image, (224, 224)))
        all_label_list.append(0)
        count_fake += 1

    print('There are %d real images\nThere are %d fake images' % (count_real, count_fake))
    all_image_list = np.array(all_image_list).reshape((len(all_image_list), 224, 224, 3))

    return all_image_list, np.array([label for label in all_label_list])