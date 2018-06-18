import csv
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Lambda, BatchNormalization, Cropping2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard

%matplotlib inline

version = 'v006'
verbose = 2
epoch = 100
correction = 0.05

lines = []
with open('./data/driving_log.csv') as csvfile:
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
        current_path = './data/IMG/' + filename
        left_path = current_path.replace('center', 'left')
        right_path = current_path.replace('center', 'right')
#         print(current_path)
#         print(left_path)
#         print(right_path)
        
        image = cv2.imread(current_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        measure = float(line[3])
        measures.append(measure)
        
        image_flipped = np.fliplr(image)
        images.append(image_flipped)
        measure_flipped = - measure
        measures.append(measure_flipped)
        
        measure_left = measure + correction
        measure_right = measure - correction

        image_left = cv2.imread(left_path)
        image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)
        images.append(image_left)
        measures.append(measure_left)

        image_left_flipped = np.fliplr(image_left)
        images.append(image_left_flipped)
        measure_left_flipped = - measure_left
        measures.append(measure_left_flipped)

        image_right = cv2.imread(right_path)
        image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2RGB)
        images.append(image_right)
        measures.append(measure_right)

        image_right_flipped = np.fliplr(image_right)
        images.append(image_right_flipped)
        measure_right_flipped = - measure_right
        measures.append(measure_right_flipped)
        

        
        if count % 1000 == 0:
            print(count, measure, measure_flipped,
                  measure_left, measure_left_flipped,
                  measure_right, measure_right_flipped)
    count += 1

X_train = np.array(images)
y_train = np.array(measures)
print("Images loading done.")

plt.figure(figsize=(6, 16))
plt.subplot(611)
plt.imshow(images[0])
plt.subplot(612)
plt.imshow(images[1])
plt.subplot(613)
plt.imshow(images[2])
plt.subplot(614)
plt.imshow(images[3])
plt.subplot(615)
plt.imshow(images[4])
plt.subplot(616)
plt.imshow(images[5])

checkpoint = ModelCheckpoint(
    version+'_E{epoch:003d}_L{val_loss:.4f}.h5', 
    monitor='val_loss', verbose=verbose, 
    save_best_only=True, 
    save_weights_only=False, 
    mode='auto', period=1)

seed = 7
np.random.seed(seed)

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255. - .5))
model.add(Conv2D(96, (11, 11), strides=(4, 4), padding='valid', activation='relu',
                 kernel_initializer='uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(BatchNormalization())
model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.summary()
model.fit(X_train, y_train, validation_split=0.1, shuffle=True, epochs=epoch, callbacks=[checkpoint])

print('Training completed.')
