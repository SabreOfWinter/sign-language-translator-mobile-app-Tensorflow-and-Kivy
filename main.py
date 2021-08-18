import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.layers.recurrent import LSTM

frame_height = 64
frame_width = 36
cnn_layer_1 = 32
cnn_layer_2 = 64
cnn_layer_3 = 128

def extract_videos():
    videos_train_path = '' #Directory for videos to be classified. Every video should be divided into folders for each class (E.g. './Data/Sign_hello/' would have all the videos of the sign hello) 
    frames_train_path = '' #Directory for extracted sequences of frames to be placed

    #Check for if frame folders have been created and create them if not
    for label in os.listdir(videos_train_path):
        if label not in os.listdir(frames_train_path):
            os.mkdir(frames_train_path + '/' + label)
    
    #Go through all training videos label by label to extract all frames 
    for label in os.listdir(videos_train_path):
        videos_path = videos_train_path + '/' + label
        frames_path = frames_train_path + '/' + label

        #Loop through each video in the current class to extract the frames 
        for video in os.listdir(videos_path):
            video_path = videos_path + '/' + video
            video_name = video.split('.')[0]

##########
    #Builds an array of the sequence of images
##########
def build_dataset(train_x, train_y, data_path):
    training_data = []
    number_of_classes = 0 #Number of video types
    number_of_samples = 0 #Number of videos

    for path in os.listdir(data_path): #Loop through each class
        number_of_classes += 1
        
        for video in os.listdir(data_path + '/' + path): #Each video
            current_frame = 0 #Track the number of frames processed in a video
            number_of_samples += 1 #Need to know how many samples there are to reshape the dataset array to fit the network
            train_y.append(number_of_classes - 1) #Minus 1 to maintain index range from 0 - x, while also counting the total number of classes

            for frame in os.listdir(data_path + '/' + path + '/' + video): #Each frame
                current_frame += 1
                img = cv2.imread(data_path + '/' + path + '/' + video + '/' + frame, cv2.IMREAD_GRAYSCALE)
                saved_img = np.array(img)
                saved_img = saved_img.astype(np.float32) / 255.0 #Normalize pixels of images
                training_data.append(saved_img) #Add frame in to training dataset

            train_x = np.array(training_data, dtype=object)

    return(train_x, train_y, number_of_samples, number_of_classes)

##########
    #LRCN model
    #the model was trained and loaded 5 times for each implementation
##########
def train(train_x, train_y, number_of_classes):   
    model = Sequential()
    model.add( TimeDistributed( Conv2D(cnn_layer_1, (3,3), activation='relu' ), input_shape=(None, frame_height, frame_width, 1) ) )
    model.add( TimeDistributed( MaxPooling2D((2,2), strides=(1,1)) ) )

    model.add( TimeDistributed( Conv2D(cnn_layer_2, (4,4), activation='relu') ) )
    model.add( TimeDistributed( MaxPooling2D((2,2), strides=(2,2)) ) )

    model.add( TimeDistributed( Conv2D(cnn_layer_3, (4,4), activation='relu')) ) 
    model.add( TimeDistributed( MaxPooling2D((3,3), strides=(2,2)) ) )

    model.add(TimeDistributed(Flatten()))
    
    #LSTM with no dropout applied
    model.add(LSTM(256, return_sequences=False))
    
    #LSTM with dropout applied before and after
    # model.add(Dropout(0.8))
    # model.add(LSTM(256, return_sequences=False, dropout=0.8))

##########
    #Used to convert LRCN SavedModel format to .tflite format
##########
def convert_model():
    saved_model_dir = '' #path to SavedModel format of selected model to convert
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()

    # Save the model
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)

def main():
    pass

if __name__ == '__main__':
    main()