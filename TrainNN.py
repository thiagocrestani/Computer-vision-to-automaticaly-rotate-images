from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import save_img
#import keras
import os
import pandas as pd 

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

#start of definition of model
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())

model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 4, activation = 'softmax'))

#opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                      metrics = ['accuracy'])


train_generator = ImageDataGenerator(rescale = 1./255,
                                         rotation_range = 7,
                                         shear_range = 0.2,
                                         height_shift_range = 0.07,
                                         zoom_range = 0.2)


test_generator = ImageDataGenerator(rescale = 1./255)

# necessary load separateclasses.py script to rum after this part
data_train = train_generator.flow_from_directory('dataset/train',
                                                           target_size = (64, 64),
                                                           batch_size = 32,
                                                           class_mode = 'categorical')


data_test = test_generator.flow_from_directory('dataset/test',
                                               target_size = (64, 64),
                                               batch_size = 32,
                                               class_mode = 'categorical')


model.fit_generator(data_train, steps_per_epoch = 39118 / 64,
                            epochs = 10, validation_data = data_test,
                            validation_steps = 9778 / 64,
                            shuffle=True)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)

# create array with the classes
class_indices = []
for class_indice in data_train.class_indices:
    class_indices.append(class_indice)



predictions = []
corrected_images = []
if not os.path.exists('test_corrected'):
            os.mkdir('test_corrected')
for file in os.listdir('test'):
    #open image on folder test to predict
    image_test = image.load_img('test/'+file,
                              target_size = (64,64))
    image_test_array = image.img_to_array(image_test)
    image_test = image_test_array/255
    image_test = np.expand_dims(image_test, axis = 0)
    #predict result
    previsao = model.predict(image_test)
    # classify image with classes
    prediction = class_indices[np.where(previsao == np.amax(previsao))[1][0]]
    # add classification on array of predicitons
    predictions.append((file, prediction))
    # set degree of rotate image according the predicion
    degree = 0
    if(prediction == 'upside_down'):
        degree = 180
    if(prediction == 'rotated_left'):
        degree = 90
    if(prediction == 'rotated_right'):
        degree = -90
    #rotate and save the image
    datagen = ImageDataGenerator()
    image_test_array = datagen.apply_transform(x=image_test_array, transform_parameters={'theta':degree})
    save_img('test_corrected/'+file.replace(".jpg","")+'.png', image_test_array)
    image_test = np.expand_dims(image_test_array, axis = 0)
    # save image rotate on list to create numpy array
    corrected_images.append(image_test_array)
    
#export array of the predicitons in CSV
df = pd.DataFrame(predictions, 
               columns =['fn', 'label']) 
df.to_csv('corrected_images.csv', index = None)

#save array of images in Numpy
correct_images = np.asarray(corrected_images)



