# Music-Reconstruction
Translating Thoughts Into Music

This project aims at developing a deep neural net capable of predicting the song a patient was listening to while an EEG signal was recorded. Features extracted from this EEG data is what is fed into the algorithm as inputs. A regression is then carried out to output a waveform prediction of the original song.

## Data Acquisition & Preprocessing

EEG data is often noisy and subject to a variety of undesirable artifacts. Therefore, a preprocessing pipeline was developed to help with this. This can be found in the preprocessing_helpers file. Bad channels were rejected, a bandpass filter was applied, and the data was downsampled.

A special thanks to Sebastian Stober, Head of Machine Learning in Cognitive Science research group at the University of Potsdam, for providing the OpenMIIR dataset. This dataset housed the unprocessed EEG recording along with the audio files that were played during each trial.

The structure of the final EEG data was comprised of a total of 12 patients, with 12 songs each, and a total of 540 trials. Each trial resulted in an EEG recording of 6.87 seconds in length with a sampling frequency of 512 Hz. This leads to a total of 3,518 samples per trial.

## Training

