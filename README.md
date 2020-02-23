# Music-Reconstruction
Translating Thoughts Into Music

This project aims at developing a deep neural net capable of predicting the song a patient was listening to while an EEG signal was recorded. Features extracted from this EEG data is what is fed into the algorithm as inputs. A regression is then carried out to output a waveform prediction of the original song.

## Data Acquisition & Preprocessing

EEG data is often noisy and subject to a variety of undesirable artifacts. Therefore, a preprocessing pipeline was developed to help with this. This can be found in the preprocessing_helpers file. Bad channels were rejected, a bandpass filter was applied, and the data was downsampled.

A special thanks to Sebastian Stober, Head of Machine Learning in Cognitive Science research group at the University of Potsdam, for providing the OpenMIIR dataset. This dataset housed the unprocessed EEG recording along with the audio files that were played during each trial.

The structure of the final EEG data was comprised of a total of 12 patients, with 12 songs each, and a total of 540 trials. Each trial resulted in an EEG recording of 6.87 seconds in length with a sampling frequency of 512 Hz. This leads to a total of 3,518 samples per trial.

## Training and Feature Selection

A moving window is used on the EEG time-series data to extract the relevant features sequentially. For each window, a total of 6 features were computed for the time-series data. These include:

* The original time-series data (384 samples)
* Power per each frequency band
* Area 
* Energy
* Number of Zero-Crossings
* Line Length

Therefore, for each 384 sample window, the end-result training matrix consisted of 64 rows (one per EEG input channel) and 390 columns, corresponding to the features listed above.

Thereafter, a principal component analysis was executed to reduce the dimensionality of the data with regards to the number of channels being used. It was reduced to its 3 main principal components, which in total explained over 99% of the original variance of the dataset. In this manner, training was performed only on the channel combinations that contain the most information with regards to predicting the song waveform.

In this sliding window approach, the features along with their corresponding output (a point on the original song waveform) were passed through a deep neural net for training. The DNN consists of 9 hidden layers with 50 neurons per layer and hyperbolic tangent activation functions.


## Results

The results and predictions of the algorithm can be seen in the presentation file found in the repository. The predictions are not extremely accurate but some underlying waveform patterns from the original song audio can be traced in the prediction waveforms.

## Conclusion 

Original EEG data was preprocessed by getting rid of faulty channels and filtering the time-series data in order to get rid of faulty artifacts. Thereafter, the features to be used were engineered and calculated in a moving window approach across the EEG time-series data. As the features were iteratively calculated, they were then fed into a DNN regression model to be trained. 

This trained model was then used to reconstruct the audio from the EEG testing set.



