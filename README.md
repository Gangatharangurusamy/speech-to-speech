# Voice Assistant Text-to-Speech (TTS) Project

This project implements a text-to-speech pipeline using the Parler-TTS model. The pipeline converts input text into spoken audio with adjustable parameters for speed, pitch, and volume. The project is designed to run on a CPU and can be used to generate audio files or play the audio in real-time.

## Features

- **Text-to-Speech Conversion**: Convert any input text into speech using the Parler-TTS model.
- **Adjustable Parameters**:
  - **Speed**: Control the speed of the generated speech.
  - **Pitch**: (Placeholder) Intended for pitch control, though not currently implemented.
  - **Volume**: Amplify or reduce the volume of the output speech.
- **Real-time Audio Playback**: Optionally play audio in real-time as it is being generated.
- **Audio Saving**: Save the generated audio to a `.wav` file.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Clone the Repository

```bash
git clone https://github.com/your-username/voice-assistant-tts.git
cd voice-assistant-tts
```
## Install Dependencies
```bash
pip install -r requirements.txt
```

## Set Up Environment Variables
### Create a .env file in the root directory and add your Google API key:

```bash
GOOGLE_API_KEY=your_google_api_key_here
```

# Usage
## Running the TTS Script
### You can run the TTS script with an example input by executing:

```bash
python t2s.py
```
This script will generate audio from the provided text and save it as output.wav.

## Adjustable Parameters
### Speed: Adjust the speed of speech. Default is 1.0 (normal speed). Values greater than 1.0 increase the speed, while values less than 1.0 decrease it.
### Pitch: Adjust the pitch of speech. Default is 1.0 (normal pitch). Note: This is a placeholder and not currently implemented.
### Volume: Adjust the volume of the output. Default is 1.5, which amplifies the volume by 50%.

## Example
```bash
text = "This is a test of the Parler-TTS model running on CPU."
description = "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch."

# Adjustable parameters
speed = 1.5  # Increase speed by 50%
pitch = 1.0  # Normal pitch
volume = 1.5  # Amplified volume by 50%

# Generate and save the audio
audio = text_to_speech(text, description, speed, pitch, volume)
sf.write("output.wav", audio, sampling_rate)
```

## Troubleshooting
Common Issues
Model Loading Error: Ensure that the Parler-TTS model and tokenizer are correctly installed and compatible with your environment.
Slow Performance: Running the TTS model on a CPU can be slow. Consider using a GPU if available.
Missing .wav File: Ensure that the script has write permissions to the directory.

## Logging and Debugging
The script includes print statements to provide real-time feedback on the progress of the audio generation process.
If you encounter any issues, check the console output for error messages.

## Future Improvements
Pitch Control: Implement functionality to adjust the pitch of the generated speech.
Command-Line Interface (CLI): Develop a CLI to simplify parameter adjustments without modifying the script.
Extended Model Support: Explore compatibility with other TTS models or enhanced versions of Parler-TTS.
