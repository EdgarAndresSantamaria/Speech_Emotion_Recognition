# Speech_Emotion_Recognition

In order to use the experiment, first of all download the RAVDESS dataset: https://www.kaggle.com/uwrfkaggler/ravdess-emotional-song-audio and https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio . Then, set up the requirements: Keras, Tensorflow, Matplotlib, numpy, Librosa, seaborn and sklearn. (Note: remember to change paths on the code bottom)

First, train your own Deep_model (medium / big) / Baseline using the input configuration {"2-D", "3-D" and "base"} (e.g data_input= "2-D"), the "3-D" trains the 2-D CNN based architecture (Big Model),the "2-D" trains the 1-D CNN based architecture (Medium Model), finally "base" trains baseline . Then decide if you want example by example analysis output (to see the data generated for the training) (e.g showCases = True). The script is fined tuned manually, and parameters are changed internally, we didn't provide further interface. You can change code chunks, input shapes, architecture details ... The script is modular designed. When you have your own deep model, test it with your microphone.

Notice: that you shoul generate the structure for every required path on the bottom of the script.

For further details of the project you can check the slides :)

Remember that both main functionality calls are in the bottom of the script, feel free to change, test, play and feedback
eandres011@ikasle.ehu.eus
