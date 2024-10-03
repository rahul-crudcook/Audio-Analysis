# Audio File Processing Module

This module is designed to process audio files (specifically in `.wav` format) to perform various analyses including cutting audio by sentences, converting audio to text, calculating acoustic and text features, and summarizing the overall content.

## Features

- **Audio Splitting**: Cuts audio file by sentences, excluding long pauses.
- **Speech-to-Text**: Converts audio to text using speech recognition.
- **Acoustic Feature Calculation**: Calculates various acoustic features for each cut audio file.
- **Text Feature Calculation**: Calculates text features such as sentiment score.
- **Data Compilation**: Creates a dataframe storing both text and acoustic features.
- **Text Summarization**: Summarizes the overall text and saves it to a separate file.
- **Named Entity Recognition**: Outputs a list of named persons in the overall audio file to a separate file.

## How It Works

The module is structured around the `AudioProcessor` class, which accepts an audio file path and an output folder path as inputs. It can also be configured to generate summaries and list named entities.

### Core Functions

- **`process()`**: Orchestrates the processing of the audio file, including slicing the audio, transcribing to text, extracting features, and generating summaries and named entity lists.
- **`extract_features()`**: Extracts various audio features from a given audio file.
- **`get_sentiment_score()`**: Analyzes the sentiment of the provided text.
- **`transcribe_audio()`**: Converts audio to text using speech recognition.
- **`summarize_text()`**: Generates a summary of the provided text.
- **`extract_persons()`**: Identifies named entities (persons) in the text.

### Usage

To use this module, initialize an instance of the `AudioProcessor` class with the desired audio file path and output folder. Then, call the `process()` method to perform the analysis.


### Notes
This script is designed for .wav audio files. Ensure your input file is in the correct format.
The accuracy of text conversion and named entity recognition might vary based on the quality of the audio file and the clarity of speech.
You might need to adjust file paths and dependencies based on your environment.
### Conclusion
This module is a comprehensive tool for analyzing audio files, extracting a wealth of information, and facilitating further analysis or machine learning tasks. By providing detailed insights into both the acoustic and textual aspects of audio, it's an invaluable resource for various applications in data science and NLP.