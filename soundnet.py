from keras.layers import BatchNormalization, Activation, Conv1D, MaxPooling1D, ZeroPadding1D, InputLayer
from keras.models import Sequential
import numpy as np
import librosa
from glob import glob
import os
import argparse
import os
import shutil



#### Parsing the arguments
parser = argparse.ArgumentParser()

parser.add_argument('--path_to_clip',metavar='path_to_clip',  default='', type=str, help='Path of the clip')
args= parser.parse_args()

base_path = args.path_to_clip



def preprocess(audio):
    audio *= 256.0  # SoundNet needs the range to be between -256 and 256
    # reshaping the audio data so it fits into the graph (batch_size, num_samples, num_filter_channels)
    audio = np.reshape(audio, (1, -1, 1))
    return audio


def load_audio(audio_file):
    sample_rate = 22050  # SoundNet works on mono audio files with a sample rate of 22050.
    audio, sr = librosa.load(audio_file, dtype='float32', sr=22050, mono=True)
    audio = preprocess(audio)
    return audio


def build_model():

    model_weights = np.load('sound8.npy', allow_pickle=True,
                            encoding='latin1').item()
    model = Sequential()
    model.add(InputLayer(batch_input_shape=(1, None, 1)))

    filter_parameters = [{'name': 'conv1', 'num_filters': 16, 'padding': 32,
                          'kernel_size': 64, 'conv_strides': 2,
                          'pool_size': 8, 'pool_strides': 8},

                         {'name': 'conv2', 'num_filters': 32, 'padding': 16,
                          'kernel_size': 32, 'conv_strides': 2,
                          'pool_size': 8, 'pool_strides': 8},

                         {'name': 'conv3', 'num_filters': 64, 'padding': 8,
                          'kernel_size': 16, 'conv_strides': 2},

                         {'name': 'conv4', 'num_filters': 128, 'padding': 4,
                          'kernel_size': 8, 'conv_strides': 2},

                         {'name': 'conv5', 'num_filters': 256, 'padding': 2,
                          'kernel_size': 4, 'conv_strides': 2,
                          'pool_size': 4, 'pool_strides': 4},

                         {'name': 'conv6', 'num_filters': 512, 'padding': 2,
                          'kernel_size': 4, 'conv_strides': 2},

                         {'name': 'conv7', 'num_filters': 1024, 'padding': 2,
                          'kernel_size': 4, 'conv_strides': 2},

                         {'name': 'conv8_2', 'num_filters': 401, 'padding': 0,
                          'kernel_size': 8, 'conv_strides': 2},

                         ]

    for x in filter_parameters:
        model.add(ZeroPadding1D(padding=x['padding']))
        model.add(Conv1D(x['num_filters'],
                         kernel_size=x['kernel_size'],
                         strides=x['conv_strides'],
                         padding='valid'))
        weights = model_weights[x['name']]['weights'].reshape(model.layers[-1].get_weights()[0].shape)
        biases = model_weights[x['name']]['biases']

        model.layers[-1].set_weights([weights, biases])

        if 'conv8' not in x['name']:
            gamma = model_weights[x['name']]['gamma']
            beta = model_weights[x['name']]['beta']
            mean = model_weights[x['name']]['mean']
            var = model_weights[x['name']]['var']


            model.add(BatchNormalization())
            model.layers[-1].set_weights([gamma, beta, mean, var])
            model.add(Activation('relu'))
        if 'pool_size' in x:
            model.add(MaxPooling1D(pool_size=x['pool_size'],
                                   strides=x['pool_strides'],
                                   padding='valid'))

    return model


def predict_scene_from_audio_file(audio_file):
    model = build_model()
    #model.summary()
    audio = load_audio(audio_file)

    return model.predict(audio)


def predictions_to_scenes(prediction):
    scenes = []
    with open('categories/categories_places2.txt', 'r') as f:
        categories = f.read().split('\n')
        for p in range(prediction.shape[1]):
            scenes.append(categories[np.argmax(prediction[0, p, :])])
    return scenes

def predictions_to_objects(prediction):
    objects = []
    with open('categories/categories_imagenet.txt', 'r') as f:
        categories = f.read().split('\n')
        for p in range(prediction.shape[1]):
            objects.append(categories[np.argmax(prediction[0, p, :])])
    return objects


# parser = argparse.ArgumentParser()
# parser.add_argument('Load_folder', metavar='Load_folder', type=str,
#                     help='specify the test folder')
# args= parser.parse_args()

# base_path = args.Load_folder

# base_path = '/Users/christos/PycharmProjects/pythonProject/extra_gn_scenes'
# print(base_path)


if __name__ == '__main__':
    output_path = 'soundnet_prediction_outputs'+os.sep
    if not os.path.exists(output_path):
        print("aaaaaa")
        os.makedirs(output_path)


    test_files = glob(base_path+'*wav')+glob(base_path+'*mp3')
    for f in test_files:
        print(f)
        prediction = predict_scene_from_audio_file(f)
        predicted_objects = predictions_to_objects(prediction)
        predicted_scenes = predictions_to_scenes(prediction)

        print(f, "aaaaa")
        print ('Scene is',predicted_scenes )
        print('Object is', predicted_objects)
        print('----------------------------------------------------')
        output_filename = f.split(os.sep)[-1].split('.')[0]
        x = np.save(f,prediction
                )
        #print(len(test_files))


        with open(output_path+output_filename+'.csv','w') as prediction_file:
            prediction_file.write('filename:'+ str(f)+ '\n')

            #prediction_file.write('predicted_vector:'+ str(prediction) + '\n')
            prediction_file.write('predicted_objects:' + str(predicted_objects)+'\n')
            prediction_file.write('predicted_scenes:'+ str(predicted_scenes))

        prediction_file.close()


# sourcepath=base_path
# sourcefiles = os.listdir(sourcepath)
# destinationpath = output_path
# for file in sourcefiles:
#     if file.endswith('.npy'):
#       shutil.move(os.path.join(sourcepath,file), os.path.join(destinationpath,file))

