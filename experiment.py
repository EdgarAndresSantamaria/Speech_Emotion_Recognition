# Python 3.6 synthax
# general imports
import numpy as np # linear algebra
import os # IO functions
import matplotlib.pyplot as plt # plotting functions
# evaluation libraries
import seaborn as sn
from sklearn.metrics import confusion_matrix, classification_report
# feature extraction and preprocessing libraries
import librosa
from librosa import display
# model libraries
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import *
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
# real test libraries
import pyaudio
import wave

'''
You may simply run the script and comment or uncomment the last lines (train or real test) as you need ...
'''

def train(dataSong, dataSpeech, analysisPath, outputPath, data_input, showCases = True):
    tags2classes = {0: "neutral", 1: "calm", 2: "happy", 3: "sad", 4: "angry", 5: "fearful", 6: "disgust",
                    7: "surprised"}

    # Cepstrum coefficient analysis
    def extract_analysis(wav_file_name, task, tag, actor_name, labels_info):
        '''This function extracts mfcc features and obtain the mean of each dimension
           Input : path_to_wav_file
           Output: mfcc_features'''
        y, sr = librosa.load(wav_file_name)
        yt, index = librosa.effects.trim(y,30)  # quit silences (noise)
        if task == "spectrum":
            # mel Spectrogram
            D = np.abs(librosa.stft(yt, n_fft=1023)) ** 2
            D = librosa.util.fix_length(D, 90, mode='edge')
            S = librosa.feature.melspectrogram(y=yt, sr=sr, n_mels=512, fmax=8000)
            S = librosa.util.fix_length(S, 90, mode='edge')
            S_dB = librosa.power_to_db(S, ref=np.max)
            S_dB = librosa.util.fix_length(S_dB, 90, mode='edge')

            def show_cases():
                plt.figure(figsize=(10, 4))
                display.specshow(D, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
                plt.colorbar(format='%+2.0f freq')
                plt.title('Mel-frequency spectrogram')
                plt.tight_layout()
                plt.savefig(analysisPath+"D-" + str(
                    actor_name) + "-" + str(labels_info) + "-" + str(tags2classes[tag]) + ".png")
                plt.clf()
                plt.cla()
                plt.close()

                plt.figure(figsize=(10, 4))
                display.specshow(S, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
                plt.colorbar(format='%+2.0f freq')
                plt.title('Mel-frequency spectrogram')
                plt.tight_layout()
                plt.savefig(analysisPath+"S-" + str(
                    actor_name) + "-" + str(labels_info) + "-" + str(tags2classes[tag]) + ".png")
                plt.clf()
                plt.cla()
                plt.close()

                plt.figure(figsize=(10, 4))
                display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
                plt.colorbar(format='%+2.0f dB')
                plt.title('Mel-frequency spectrogram')
                plt.tight_layout()
                plt.savefig(analysisPath+"S_dB-" + str(
                    actor_name) + "-" + str(labels_info) + "-" + str(tags2classes[tag]) + ".png")
                plt.clf()
                plt.cla()
                plt.close()

            if showCases:
                show_cases()
            result = np.stack([D, S, S_dB], -1)
        elif task == "mfccs":
            # mfccs
            result = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            result = librosa.util.fix_length(result, 128, mode='edge')
            def show_cases():
                plt.figure(figsize=(10, 4))
                librosa.display.specshow(result, x_axis='time')
                plt.colorbar()
                plt.title('MFCC')
                plt.tight_layout()
                plt.savefig(analysisPath + str(
                        actor_name) + "-" + str(labels_info) + "-" + str(tags2classes[tag]) + ".png")
                plt.clf()
                plt.cla()
                plt.close()

            if showCases:
                show_cases()
        return result

    def load_data(root_dir, task):
        labels = []
        data = []
        for actor_dir in sorted(os.listdir(root_dir)):
            actor_name = os.path.join(root_dir, actor_dir)
            print("loading: {}".format(actor_name))
            for file in os.listdir(actor_name):
                labels_info = file.split(".")[0]
                labels_list = labels_info.split("-")
                label = int(labels_list[2]) - 1
                if label in [0, 1, 2, 3, 4, 5, 6, 7]:
                    labels.append(label)  # normalize labeling (0..7)
                    wav_file_name = os.path.join(root_dir, actor_dir, file)
                    data.append(extract_analysis(wav_file_name, task, label, actor_dir, labels_info))

        return data, labels


    print('loading data ...')
    print('0% ..')
    ##### load radvess song data #####
    ravdess_song_data, radvess_song_labels = load_data(dataSong, data_input)
    print('50% ..')
    ##### load radvess speech data #####
    ravdess_speech_data, radvess_speech_labels = load_data(dataSpeech, data_input)
    print('100% ..')

    def create_baseline():
        ''': 0.4 acc
        This model uses:

        Mel Frequency Cepstral Coefficients (MFCC) as input

        Kuchibhotla, S. , Vankayalapati, H. , Vaddi, R. , Anne, K.R. , 2014. A comparative analysis
    of classifiers in emotion recognition through acoustic features. Int. J. Speech Technol.
    17 (4), 401–408 .

        BiLSTM (representeation)
        Conv1D (representation)
        MaxPooling1D (sample selection)
        Conv1D (feature selection)
        Softmax (prediction)

        Lim, W. , Jang, D. , Lee, T. , 2016. Speech emotion recognition using convolutional and
    recurrent neural networks. In: 2016 Asia-Pacific Signal and Information Processing
    Association Annual Summit and Conference (APSIPA). IEEE, pp. 1–4 .

        '''

        # todo optional fine tune the net
        ### LSTM model, referred to the model A in the report
        model = Sequential()
        # input the ceptrum coefficient
        model.add(
            Bidirectional(LSTM(40, return_sequences=True, input_shape=(40, 1))))  # generate 128 features ctx (RNNs)
        model.add(Dense(80))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        # analyze Cepstral context (forward and backward) based on RNNs
        model.add(Conv1D(32, 5, padding='same'))  # generate 256 features ctx (CNNs)
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=(4)))  # get the most representative 5 examples (40 : 8 pools)
        model.add(Conv1D(64, 5, padding='same'))  # get the most likely 128 features
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Flatten())  # flat into only one dimension
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(8, activation='softmax'))
        # add inference layer based on convolutional CNNs
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        return model

    def create_deep_model():
        '''
        overall architecture:

        Mel Spectral analysis as input
        filtering blocks (32, 64, 128):
            Conv2D (representation)
            MaxPooling2D (sample selection) // based on strides
        Dense (fully connected layer)
        Softmax (prediction)

        tools selection : Emotion Recognition from Speech Kannan Venkataramanan, Haresh Rengaraj Rajamohan, 2019

        implementation details : https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/

        fine tuning: https://arxiv.org/pdf/1707.09725.pdf
        :return:
        '''

        reg = l2(0.05)
        init = "he_normal"
        chanDim = -1
        inputShape = (512, 90, 3)
        opt = Adam(lr=1e-4, decay=1e-4)
        # building the model
        model = Sequential()
        # treat the input (from big kernels to little)
        # each filtering stage takes care of every detail in the Spectrogram
        model.add(Conv2D(16, (7, 7), strides=(2, 2), padding="valid",
                         kernel_initializer=init, kernel_regularizer=reg,
                         input_shape=inputShape))
        # here we stack two CONV layers on top of each other where
        # each layerswill learn a total of 32 (3x3) filters
        model.add(Conv2D(32, (3, 3), padding="same",
                         kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), strides=(2, 2), padding="valid",
                         kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))
        # stack two more CONV layers, keeping the size of each filter
        # as 3x3 but increasing to 64 total learned filters
        model.add(Conv2D(64, (3, 3), padding="same",
                         kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="valid",
                         kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))
        # increase the number of filters again, this time to 128
        model.add(Conv2D(128, (3, 3), padding="same",
                         kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="valid",
                         kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))
        # fully-connected layer
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.7))
        # softmax classifier
        model.add(Dense(8))
        model.add(Activation("softmax"))
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    print('formating data ...')
    # format the data flow

    data = np.r_[ravdess_song_data, ravdess_speech_data]
    print(data.shape)
    labels = np.r_[radvess_song_labels, radvess_speech_labels]
    labels_categorical = to_categorical(labels)
    number_of_samples = data.shape[0]
    training_samples = int(number_of_samples * 0.8)

    tags, counts = np.unique(labels[:training_samples], return_counts=True)
    tags = [tags2classes[tag] for tag in tags]
    plt.pie(counts, labels=tags, autopct="%0.1f %%")
    plt.axis("equal")
    plt.savefig(analysisPath+"distribution_train.png")
    plt.clf()
    plt.cla()
    plt.close()

    tags, counts = np.unique(labels[training_samples:], return_counts=True)
    tags = [tags2classes[tag] for tag in tags]
    plt.pie(counts, labels=tags, autopct="%0.1f %%")
    plt.axis("equal")
    plt.savefig(analysisPath+"distribution_test.png")
    plt.clf()
    plt.cla()
    plt.close()

    print('creating the model ...')
    ### train using the Deep model
    if data_input == "mfccs":
        model = create_baseline()
    elif data_input == "spectrum":
        model = create_deep_model()

    # patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

    history = model.fit(data[:training_samples], labels_categorical[:training_samples],
                        validation_data=(data[training_samples:], labels_categorical[training_samples:]),
                        epochs=1000, shuffle=True, callbacks=[es])

    print('Net topography')
    print(model.summary())

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(data[training_samples:], batch_size=32)
    print(classification_report(labels[training_samples:], predictions.argmax(axis=1),
                                target_names=list(tags2classes.values())))

    ### loss plots
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'ro', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(outputPath+"loss_curve.png")
    plt.clf()
    plt.cla()
    plt.close()

    ### accuracy plots
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(epochs, acc, 'ro', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(outputPath+"acurracy_curve.png")
    plt.clf()
    plt.cla()
    plt.close()

    # provide training results
    cm = confusion_matrix(model.predict_classes(data[training_samples:]), labels[training_samples:])
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(cm, annot=True, annot_kws={"size": 16})  # font size
    plt.savefig(outputPath+"prediction_eval.png")
    plt.clf()
    plt.cla()
    plt.close()

    # save model
    model.save(outputPath+"model.h5")

def real_test(intermediumPath, modelPath):
    # Setup:
    # $ sudo apt-get install portaudio19-dev python-pyaudio python3-pyaudio
    # pip3.6 install PyAudio
    p = pyaudio.PyAudio()
    FRAMES_PERBUFF = 2048  # number of frames per buffer
    FORMAT = pyaudio.paInt16  # 16 bit int
    CHANNELS = 1  # I guess this is for mono sounds
    FRAME_RATE = 44100  # sample rate

    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=FRAME_RATE,
                input=True,
                frames_per_buffer=FRAMES_PERBUFF) #buffer

    frames = []

    RECORD_SECONDS = 4
    nchunks = int(RECORD_SECONDS * FRAME_RATE / FRAMES_PERBUFF)
    print("* start speaking")
    for i in range(0, nchunks):
        data = stream.read(FRAMES_PERBUFF)
        frames.append(data)  # 2 bytes(16 bits) per channel
    print("* done recording")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # save as wave real_test
    wf = wave.open(intermediumPath+'test.wav', 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(FRAME_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print("initializing the process")
    print("laoding the test")
    y, sr = librosa.load(intermediumPath+"test.wav")
    yt, index = librosa.effects.trim(y, 30)  # quit silences (noise)
    D = np.abs(librosa.stft(yt, n_fft=1023)) ** 2
    D = librosa.util.fix_length(D, 90, mode='edge')
    S = librosa.feature.melspectrogram(y=yt, sr=sr, n_mels=512, fmax=8000)
    S = librosa.util.fix_length(S, 90, mode='edge')
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_dB = librosa.util.fix_length(S_dB, 90, mode='edge')

    plt.figure(figsize=(10, 4))
    display.specshow(D, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.savefig("example_analysis_D.png")
    plt.clf()
    plt.cla()
    plt.close()

    plt.figure(figsize=(10, 4))
    display.specshow(S, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.savefig(intermediumPath+"example_analysis_S.png")
    plt.clf()
    plt.cla()
    plt.close()

    plt.figure(figsize=(10, 4))
    display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.savefig(intermediumPath+"example_analysis_SDB.png")
    plt.clf()
    plt.cla()
    plt.close()

    tags2classes = {0: "neutral", 1: "calm", 2: "happy", 3: "sad", 4: "angry", 5: "fearful", 6: "disgust",
                    7: "surprised"}

    data = np.stack([D, S, S_dB], -1)
    data = np.expand_dims(data, 0)
    model = load_model(modelPath+"model.h5")
    predictions = model.predict(data).argmax(axis=1)
    print(" I think you are bip bop....: {}".format(tags2classes[predictions[0]]))


# show cases retrieves feedback of the feature extraction (over all data examples to check)
'''
dataSongPath =  "/media/edgar/407d4115-9ff4-45c6-9279-01b62aee0730/Speech_processing/ravdess-emotional-song-audio/audio_song_actors_01-24/"
dataSpeechPath = "/media/edgar/407d4115-9ff4-45c6-9279-01b62aee0730/Speech_processing/ravdess-emotional-speech-audio/audio_speech_actors_01-24/"
analysisPath = "/media/edgar/407d4115-9ff4-45c6-9279-01b62aee0730/Speech_processing/data_analysis/"
outputPath = "/media/edgar/407d4115-9ff4-45c6-9279-01b62aee0730/Speech_processing/model/"
data_input = "mfccs"
train(dataSongPath, dataSpeechPath, analysisPath, outputPath, data_input showCases = False)

Notice: concole logging isn't implemented so you may copy the outputs ...
'''

# path to generate the live test (with your microphone)
intermediumPath = "/media/edgar/407d4115-9ff4-45c6-9279-01b62aee0730/Speech_processing/real_test/"
# path to the model that guesses (the quality of the microphone matters I don't have a nice one but ... still works xD)
modelPath ="/media/edgar/407d4115-9ff4-45c6-9279-01b62aee0730/Speech_processing/final_model/model/"
real_test(intermediumPath, modelPath)

