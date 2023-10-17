

import numpy as np
import pickle
import cv2
from os import listdir
import tensorflow as tf
from tensorflow.keras import optimizers
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K



EPOCHS = 25
INIT_LR = 1e-3
BS = 32
default_image_size = tuple((256, 256))
image_size = 0
#directory_root = '../input/plantvillage/'
directory_root = 'C:\\Users\\16028\\Downloads\\archive (2)\\PlantVillage\\'
width=256
height=256
depth=3

"""Function to convert images to array"""

def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

"""Fetch images from directory"""

image_list, label_list = [], []
try:
    print("[INFO] Loading images ...")
    root_dir = listdir(directory_root)
    
    for directory in root_dir :
        
        if directory == ".DS_Store" :
            root_dir.remove(directory)

      
    plant_disease_folder_list = listdir(f"{directory_root}")
    for disease_folder in plant_disease_folder_list :
        
        if disease_folder == ".DS_Store" :
            plant_disease_folder_list.remove(disease_folder)

    for plant_disease_folder in plant_disease_folder_list:
        print(f"[INFO] Processing {plant_disease_folder} ...")
        plant_disease_image_list = listdir(f"{directory_root}/{plant_disease_folder}/")
            
        for single_plant_disease_image in plant_disease_image_list :
            if single_plant_disease_image == ".DS_Store" :
                plant_disease_image_list.remove(single_plant_disease_image)

        for image in plant_disease_image_list[:220]:
            image_directory = f"{directory_root}/{plant_disease_folder}/{image}"
            if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                image_list.append(convert_image_to_array(image_directory))
                label_list.append(plant_disease_folder)

    print("[INFO] Image loading completed")  
except Exception as e:
    print(f"Error : {e}")




image_list_val, label_list_val=[], []
try:
    print("[INFO] Loading images ...")
    root_dir = listdir(directory_root)
    
    for directory in root_dir :
        
        if directory == ".DS_Store" :
            root_dir.remove(directory)

      
    plant_disease_folder_list = listdir(f"{directory_root}")
    for disease_folder in plant_disease_folder_list :
        
        if disease_folder == ".DS_Store" :
            plant_disease_folder_list.remove(disease_folder)

    for plant_disease_folder in plant_disease_folder_list:
        print(f"[INFO] Processing {plant_disease_folder} ...")
        plant_disease_image_list = listdir(f"{directory_root}/{plant_disease_folder}/")
            
        for single_plant_disease_image in plant_disease_image_list :
            if single_plant_disease_image == ".DS_Store" :
                plant_disease_image_list.remove(single_plant_disease_image)

        if (plant_disease_folder != "Potato___healthy") :
            for image in plant_disease_image_list[220:250]:
                image_directory = f"{directory_root}/{plant_disease_folder}/{image}"
                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                    image_list_val.append(convert_image_to_array(image_directory))
                    label_list_val.append(plant_disease_folder)
        else:
            for image in plant_disease_image_list[120:150]:
                image_directory = f"{directory_root}/{plant_disease_folder}/{image}"
                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                    image_list_val.append(convert_image_to_array(image_directory))
                    label_list_val.append(plant_disease_folder)          

    print("[INFO] Image loading completed")  
except Exception as e:
    print(f"Error : {e}")

"""Get Size of Processed Image"""

image_size = len(image_list)

image_size

image_list[:1]

"""Transform Image Labels uisng [Scikit Learn](http://scikit-learn.org/)'s LabelBinarizer"""

label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)

pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))
n_classes = len(label_binarizer.classes_)


label_binarizer2 = LabelBinarizer()
image_labels_val = label_binarizer2.fit_transform(label_list_val)

"""Print the classes"""

image_labels[0]

print(label_binarizer.classes_)

np_image_list = np.array(image_list, dtype=np.float16) / 255.0


np_image_list_val = np.array(image_list_val, dtype=np.float16) / 255.0

np_image_list[:1]

print("[INFO] Spliting data to train, test")
x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state = 42)

y_test[0]

y_train[0]

aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, 
    zoom_range=0.2,horizontal_flip=True, 
    fill_mode="nearest")

x_train[0].shape

model = Sequential()
inputShape = (height, width, depth)
chanDim = -1
if K.image_data_format() == "channels_first":
    inputShape = (depth, height, width)
    chanDim = 1
model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.33))
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.33))
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.33))
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.33))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.33))
model.add(Dense(n_classes))
model.add(Activation("softmax"))

"""Model Summary"""

model.summary()

opt =  tf.keras.optimizers.legacy.Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

print("[INFO] training network...")

EPOCHS = 25

checkpoint_filepath = f'Trained model1'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='auto',
    save_best_only=True)

history = model.fit(
    aug.flow(x_train, y_train, batch_size=BS),
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // BS,
    epochs=EPOCHS, verbose=2,
    callbacks=[model_checkpoint_callback] )

"""Plot the train and val curve"""

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

"""Model Accuracy"""
model = tf.keras.models.load_model("Trained model1") 
print("[INFO] Calculating model accuracy")
scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")
scores = model.evaluate(np_image_list_val, image_labels_val)
print(f"Test Accuracy: {scores[1]*100}")


from sklearn.metrics import confusion_matrix

y_pred = model.predict(np_image_list_val)
y_pred = np.argmax(y_pred, axis=1)
image_labels_val = np.argmax(image_labels_val, axis=1)

conf_mat = confusion_matrix(image_labels_val, y_pred)

print(conf_mat)

from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1', 'class 2','class 3','class 4','class 5','class 6','class 7','class 8','class 9','class 10','class 11','class 12','class 13','class 14']
print(classification_report(image_labels_val, y_pred, target_names=target_names))
