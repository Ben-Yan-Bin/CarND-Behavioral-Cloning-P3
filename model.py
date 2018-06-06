import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measures = []
count = 0
for line in lines:
    if 0 < count:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = '../data/IMG/' + filename
        print(count, end=', ')
        image = cv2.imread(current_path)
        images.append(image)
        measure = float(line[3])
        measures.append(measure)
    count += 1
X_train = np.array(images)
# X_train = X_train/255. - 0.5
y_train = np.array(measures)
# plt.imshow(images[0])
# plt.show()
# print(count - 1)


from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
import numpy as np

seed = 7
np.random.seed(seed)

model = Sequential()
model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=(160, 320, 3), padding='valid', activation='relu',
                 kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(1, activation='relu'))

# model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['accuracy', 'loss'])
model.summary()
model.fit(X_train, y_train, validation_split=0.1, shuffle=True, nb_epoch=3)

model.save('model.h5')
