# Computer vision to automaticaly rotate images
This software receives images at different rotations, sorts according to their position and rotates them upside.
To know the position of an image the software has a neural network that has been trained a set of test images.
# Inputs
For the software training i used the set of images from the train.rotfaces/train folder, 
which are classified according to the document train.rotfaces/train.truth.csv.
To make the process easier, there is the separateclasses.py script, which reads the training folder images and the 
classification file and inserts the separate images into their respective folders in the dataset folder. 
This script also randomly separates data into training and testing to facilitate insertion into the neural network.
# Training
The model used to classify the images was based on Keras CIFAR10 (https://keras.io/examples/cifar10_cnn/) 
with occasional changes to improve your perfomance.
I used the value of 64 in input_shape because all images have these dimensions. 
I also removed the dropout from the convolution layer, basically just for testing and ended up saving the model like that, 
but it didn't influence the final result.
I added BatchNormalization to convolution layers to increase training speed. 
The optimizer I used was adam, it showed better results than RMSprop.
Finally I used dataAugumentation to test, it did not influence the results because I was already testing with a smaller 
amount of images, but as it builds, I left the model for future improvement.
# Execution
With images sorted by the separateclasses.py script we can separate training and testing bases with the 
flow_from_directory method.
Then training can be performed. 
For the tests I performed in 10 epochs, and i sended just 1/64 of the data each epoch, 
to speed up the process, if you want to perform with all test images, just remove 
the division by 64 of the steps of the fit method.
# Results
During some tests, the results were usually 90%+. However, I believe it is possible to get close to 97%-98%.
# Completing the code
After training, the software predicts all images in the test folder. It sorts the images and keeps the
prediction data stored in an array, to ultimately write csv document in the disk. In addition, 
it rotates the image as predicted by the neural network using the ImageDataGenerator() method and 
saves the images in the test_corrected folder. Finally, 
the system stores the corrected image data and writes it to a numpy array, allowing future imports.
# Data Results
After running the scripts the corrected images will be available in the test_corrected folder, however, you can download the software corrected images folder via the link https://1drv.ms/u/s!Au7LN9Kj-TAtlUIMGMB-XmpnUFdJ?e=d9bghV.
# Detailed steps for implementation
- Download the files https://1drv.ms/u/s!Au7LN9Kj-TAtlUN0xnmDuklOp4J5?e=TPDCcC and https://1drv.ms/u/s!Au7LN9Kj-TAtlUEeFR2PCU-KkrI3?e=O68P17 and unzip on the root of the project.
- Run the separateclasses.py script.
- Run the TrainNN.py script
#
Outputs:
- The dataset folder will be created with the data of the organized train.rotfaces folder.
- The corrected_images.csv file will be created with the predictions of images from the test folder.
- The test_corrected folder with the upside oriented images will be created.
- A numpy array will be created with the correctly oriented images.
