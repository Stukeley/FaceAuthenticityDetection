from imutils import paths
import numpy as np
import cv2,os,random
from model import MiniVGG
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import img_to_array
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import classification_report,confusion_matrix

def train():
	img_height=128 # height of the training image
	img_width=128 #width of the training image
	EPOCHS = 10 #number of epochs to be trained for
	num_classes=2 #number of labels
	INIT_LR = 1e-3 #Initial Learning rate
	BS = 32 # Bach size to feed

	# initialize the data and labels
	data = []
	labels = []
	# grab the image paths and randomly shuffle them
	#imagePaths = sorted(list(paths.list_images("folds/fold4/train")))
	random.seed(42)

	# Train
	auth_path = "../folds/fold4/train/authentic"
	spoof_path = "../folds/fold4/train/spoof"

	auth_images = []
	spoof_images = []

	for img in os.listdir(auth_path):
		if img.endswith(".png") or img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith(".webp"):
			loaded = cv2.imread(os.path.join(auth_path, img))
			loaded = cv2.resize(loaded, (img_height, img_width))
			auth_images.append(loaded)

	for img in os.listdir(spoof_path):
		if img.endswith(".png") or img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith(".webp"):
			loaded = cv2.imread(os.path.join(spoof_path, img))
			loaded = cv2.resize(loaded, (img_height, img_width))
			spoof_images.append(loaded)

	X_train = auth_images + spoof_images
	y_train = [0]*len(auth_images) + [1]*len(spoof_images)

	X_train = np.array(X_train, dtype="float") / 255.0
	y_train = np.array(y_train)

	# Test
	auth_path = "../folds/fold4/test/authentic"
	spoof_path = "../folds/fold4/test/spoof"

	auth_images = []
	spoof_images = []

	for img in os.listdir(auth_path):
		if img.endswith(".png") or img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith(".webp"):
			loaded = cv2.imread(os.path.join(auth_path, img))
			loaded = cv2.resize(loaded, (img_height, img_width))
			auth_images.append(loaded)

	for img in os.listdir(spoof_path):
		if img.endswith(".png") or img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith(".webp"):
			loaded = cv2.imread(os.path.join(spoof_path, img))
			loaded = cv2.resize(loaded, (img_height, img_width))
			spoof_images.append(loaded)

	X_test = auth_images + spoof_images
	y_test = [0]*len(auth_images) + [1]*len(spoof_images)

	X_test = np.array(X_test, dtype="float") / 255.0
	y_test = np.array(y_test)

	# scale the raw pixel intensities to the range [0, 1]
	# data = np.array(data, dtype="float") / 255.0
	# np.save('data.npy',data)
	# labels = np.array(labels)
	# np.save('labels.npy',labels)
	# data=np.load('data.npy')
	# labels=np.load('labels.npy')


	# partition the data into training and testing splits using 75% of
	# the data for training and the remaining 25% for testing
	# (trainX, testX, trainY, testY) = train_test_split(data,
	# 	labels, test_size=0.25, random_state=42)
	channels=X_train.shape[3]
	# convert the labels from integers to vectors
	trainY = to_categorical(y_train, num_classes)
	testY = to_categorical(y_test, num_classes)

	# construct the image generator for data augmentation
	aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,horizontal_flip=True, fill_mode="nearest")

	# initialize the model
	print("Compiling model...")
	model = MiniVGG(width=img_width, height=img_height, depth=channels, classes=num_classes)
	opt = Adam(lr=INIT_LR) #Optimise uisng Adam
	model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

	# train the network
	print("Training network")
	H = model.fit_generator(aug.flow(X_train, trainY, batch_size=BS),
							validation_data=(X_test, testY),
							epochs=EPOCHS, verbose=1)
	label_name=["real","fake"]
	print("[INFO] evaluating network...")
	predictions = model.predict(X_test, batch_size=BS)
	print(classification_report(testY.argmax(axis=1),
								predictions.argmax(axis=1)))

	cm = confusion_matrix(testY.argmax(axis=1), predictions.argmax(axis=1))
	total = sum(sum(cm))
	acc = (cm[0, 0] + cm[1, 1]) / total
	print("Total accuracy: {:.4f}".format(acc))
	print(cm)


if __name__ == "__main__":
	train()
