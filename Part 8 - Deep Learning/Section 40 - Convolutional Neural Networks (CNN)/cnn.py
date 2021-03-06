# Convolutional Neural Network

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# Steps: Convolution -> Max Pooling -> Flattening -> Full Connection (ANN)
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
# Add the Convolution layer
classifier.add(Convolution2D(filters=32, kernel_size=(3, 3), input_shape=(64, 64, 3), activation='relu'))

# Step 2 - Max Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(filters=32, kernel_size=(3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection (Create ANN)
classifier.add(Dense(units=128, activation='relu'))

# Output layer
classifier.add(Dense(units=1, activation='sigmoid'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Creating the training set
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# Create the test set
test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

# Fit the data to the CNN model
classifier.fit_generator(
    training_set,
    steps_per_epoch=8000,
    epochs=25,
    validation_data=test_set,
    validation_steps=2000)
