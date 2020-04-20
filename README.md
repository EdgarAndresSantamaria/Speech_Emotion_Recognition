# Speech_Emotion_Recognition

In order to use the experiment, first of all download the RAVDESS dataset: https://www.kaggle.com/uwrfkaggler/ravdess-emotional-song-audio and https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio . Then, set up the requirements: Keras, Tensorflow, Matplotlib, numpy, Librosa, seaborn and sklearn. (Note: remember to change paths on the code bottom)

First, train your own Deep_model / Baseline using the input configuration {"spectrum" or "mfccs"} (e.g data_input= "mfccs"), the "spectrum" trains the 2-D CNN based architecture (Deep Model). Finally decide if you want example by example analysis output (to see the data generated for the training) (e.g showCases = True). The script is fined tuned manually, and parameters are changed internally, we didn't provide further interface. You can change code chunks, input shapes, architecture details ... The script is modular designed.

We you have your own deep model, test it with your microphone.

Remember that both main functionality calls are in the bottom of the script, feel free to change, test, play and feedback
eandres011@ikasle.ehu.eus
