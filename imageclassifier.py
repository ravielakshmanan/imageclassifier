# Name: Ravie Lakshmanan (RL2857)
# Email: rl2857@columbia.edu

# Answer to Part 4. 4. Is the binary classification task easier or more difficult than classification into 10 categories? Justify your response.
# The binary classification task was much more easier in terms of overall training time (approx. 15 mins) and it also yields a higher accuracy of 90%.
# In contrast, the multi-label classification takes close to 150 mins (approx. 5 minutes per epoch) and has an accuracy of just 76.5%.
# This shows that multi-label classification tasks are much more harder to resolve, as they require a significantly larger training dataset
# to yield highly accurate results and map an image to the correct label. It also depends on selecting relevant features in the dataset and eliminate
# irrelevant features.

import numpy as np
from keras.datasets import cifar10
from keras import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras import optimizers
from keras.utils import to_categorical


# function to rescale training data to the range 0 to 1
def normalize(data):
    d_min = data.min(axis=(1, 2), keepdims=True)
    d_max = data.max(axis=(1, 2), keepdims=True)

    data_n = (data - d_min) / (d_max - d_min)

    return data_n


# function to get the binary one-hot vector of gold labels
def get_binary_onehot_data(data):

    # iterate over the training dataset to check if the value > 0
    onehot_data = []
    for item in data:
        if item[0] > 0:
            onehot_data.append(1)
        else:
            onehot_data.append(0)

    onehot_array = np.array(onehot_data)

    return onehot_array


# function to process training data
def process_data(xtrain, ytrain, xtest, ytest):

    # get the one-hot vectors of gold labels
    ytrain_1hot = to_categorical(ytrain, num_classes=10)
    ytest_1hot = to_categorical(ytest, num_classes=10)

    # normalize training data
    normalized_xtrain = normalize(xtrain)
    normalized_xtest = normalize(xtest)

    return normalized_xtrain, ytrain_1hot, normalized_xtest, ytest_1hot


# function to the load the cifar10 data
def load_cifar10():

    # load the cifar10 data
    train, test = cifar10.load_data()
    xtrain, ytrain = train
    xtest, ytest = test

    return xtrain, ytrain, xtest, ytest


# function to the build the multi-layer neural net
# output after 30 epochs: 50000/50000 [==============================] - 9s 183us/step - loss: 1.2228 - acc: 0.5715
# output of evaluate(): Loss: 1.22284243889 Accuracy: 0.5715
def build_multilayer_nn():
    nn_model = Sequential()
    nn_model.add(Flatten(input_shape=(32,32,3,)))
    nn_model.add(Dense(units=100, activation="relu", input_shape=(3072,)))
    nn_model.add(Dense(units=10, activation="softmax"))

    return nn_model


# function to the train the multi-layer neural net
def train_multilayer_nn(model, xtrain, ytrain):
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(xtrain, ytrain, epochs=30, batch_size=32)
 

# function to the build the convolution neural net
# output after 30 epochs: 50000/50000 [==============================] - 356s 7ms/step - loss: 0.6276 - acc: 0.7772
# output of evaluate(): Loss: 0.674853273439 Accuracy: 0.7656
def build_convolution_nn():
    convolution_nn_model = Sequential()
    convolution_nn_model.add(Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(32,32,3,)))
    convolution_nn_model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    convolution_nn_model.add(MaxPooling2D(pool_size=(2,2)))
    convolution_nn_model.add(Dropout(0.25))
    convolution_nn_model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    convolution_nn_model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    convolution_nn_model.add(MaxPooling2D(pool_size=(2,2)))
    convolution_nn_model.add(Dropout(0.5))
    convolution_nn_model.add(Flatten())
    convolution_nn_model.add(Dense(units=500, activation="relu"))
    convolution_nn_model.add(Dense(units=100, activation="relu"))
    convolution_nn_model.add(Dense(units=10, activation="softmax"))

    return convolution_nn_model


# function to the train the convolutional neural net
def train_convolution_nn(model, xtrain, ytrain):
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(xtrain.shape)
    print(ytrain.shape)
    model.fit(xtrain, ytrain, epochs=30, batch_size=32)
    

# function to load the binary cifar10 data
def get_binary_cifar10():
    xtrain, ytrain, xtest, ytest = load_cifar10()

    normalized_xtrain = normalize(xtrain)
    normalized_xtest = normalize(xtest)

    ytrain_1hot = get_binary_onehot_data(ytrain)
    ytest_1hot = get_binary_onehot_data(ytest)

    return normalized_xtrain, ytrain_1hot, normalized_xtest, ytest_1hot


# function to the build the binary classifier
# output after 30 epochs: 50000/50000 [==============================] - 40s 799us/step - loss: 1.5942 - acc: 0.9000
# output of evaluate: Loss: 1.59423857307 Accuracy: 0.9
def build_binary_classifier():
    binary_nn_model = Sequential()
    binary_nn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32,32,3,)))
    binary_nn_model.add(MaxPooling2D((1, 2)))
    binary_nn_model.add(Flatten())
    binary_nn_model.add(Dense(1, activation='sigmoid'))

    return binary_nn_model


# function to train the binary classifier
def train_binary_classifier(model, xtrain, ytrain):
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(xtrain, ytrain, epochs=30, batch_size=32)


if __name__ == "__main__":

    # loading the CIFAR-10 data
    x_train, y_train, x_test, y_test = load_cifar10()
    x_train_normalized, y_train_1hot, x_test_normalized, y_test_1hot = process_data(x_train, y_train, x_test, y_test)

    # building the multi-layer model
    multi_nn_model = build_multilayer_nn()
    multi_nn_model.summary()

    # training the model
    train_multilayer_nn(multi_nn_model, x_train_normalized, y_train_1hot)

    # evaluate the trained model
    scores = multi_nn_model.evaluate(x_test_normalized, y_test_1hot, verbose=0)
    print("Loss: " + str(scores[0]) + " Accuracy: " + str(scores[1]))

    # building the CNN model
    conv_nn_model = build_convolution_nn()
    conv_nn_model.summary()

    # training the model
    train_convolution_nn(conv_nn_model, x_train_normalized, y_train_1hot)

    # evaluate the trained model
    conv_scores = conv_nn_model.evaluate(x_test_normalized, y_test_1hot, verbose=0)
    print("Loss: " + str(conv_scores[0]) + " Accuracy: " + str(conv_scores[1]))

    # get the binary CIFAR-10 data
    b_x_train, b_y_train, b_x_test, b_y_test = get_binary_cifar10()

    # build the binary classifier model
    binary_nn_model = build_binary_classifier()
    binary_nn_model.summary()

    # train the model
    train_binary_classifier(binary_nn_model, b_x_train, b_y_train)

    # evaluate the trained model
    bin_scores = binary_nn_model.evaluate(b_x_test, b_y_test, verbose=0)
    print("Loss: " + str(bin_scores[0]) + " Accuracy: " + str(bin_scores[1]))