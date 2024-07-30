# Arabic Voice Recognition and Transcription (2022)

This project involves developing a machine learning program using Python to recognize Arabic voice and transcribe it into text. The model is trained on a large dataset of Arabic speech to ensure accuracy and effectiveness.

## Features

- **Voice Recognition**: Recognizes spoken Arabic and transcribes it into text.
- **Machine Learning**: Utilizes advanced machine learning algorithms for speech recognition.
- **Large Dataset**: Trained on an extensive dataset of Arabic speech for improved accuracy.

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/arabic-voice-recognition.git
    cd arabic-voice-recognition
    ```

2. **Create and activate a virtual environment**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The model is trained on a large dataset of Arabic speech. Ensure you have the dataset in the correct format before running the training script. 


## Usage

### Data Processing

Before training the model, the dataset needs to be processed. The Jupyter notebook `Data Processing and Training.ipynb` includes steps for:

1. **Loading the Data**: Loading audio files and their corresponding labels.
2. **Preprocessing**: Normalizing and augmenting the audio data.
3. **Feature Extraction**: Extracting features like MFCC (Mel-frequency cepstral coefficients) from the audio data.

### Training the Model

To train the model, run the `Data Processing and Training.ipynb` notebook. This will:

1. Load and preprocess the dataset.
2. Train the model on the processed data.
3. Evaluate the model on the test set.

### Running the Notebook

1. **Start Jupyter Notebook**:

    ```bash
    jupyter notebook
    ```

2. **Open and run `Data Processing and Training.ipynb`**.

### Recognizing and Transcribing Speech

Once the model is trained, use the trained model to recognize and transcribe Arabic speech from audio files. This can be done using the following script:

```python
import speech_recognition as sr

def recognize_speech(audio_path, model):
    # Load and preprocess the audio file
    # Use the model to predict the transcription
    # Return the transcription
    pass

audio_path = 'path_to_audio_file.wav'
transcription = recognize_speech(audio_path, trained_model)
print("Transcription:", transcription)


