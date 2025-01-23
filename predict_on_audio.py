"""
Predict multiple F0 output from input audio or folder.
This code is linked to the ISMIR paper:

Helena Cuesta, Brian McFee and Emilia Gómez (2020).
Multiple F0 Estimation in Vocal Ensembles using Convolutional Neural Networks.
In Proceedings of the 21st International Society for Music Information Retrieval Conference (ISMIR).
Montreal, Canada (virtual).
"""


from __future__ import print_function
import models
import utils
import utils_train
import tensorflow as tf # changeId 1
K=tf.keras.backend # changeID 1
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed #changeID 15

from pydub import AudioSegment  # To handle audio splitting #changeID 19
import tempfile  # For temporary storage of chunks #changeID 19

import os
import argparse

# this is our replacement function because model.load_weights would call the keras 3 version
# which complains about incompatibility 
from load_weights import load_weights

physical_devices = tf.config.list_physical_devices('GPU') #changeID 9
for device in physical_devices: #changeID 9
    tf.config.experimental.set_memory_growth(device, True) #changeID 9
#    tf.config.set_logical_device_configuration(device, [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]) #changeID14
tf.config.optimizer.set_jit(True)  # Enable XLA (Accelerated Linear Algebra) compiler #changeID 10


def main(args):

    model_name = args.model_name
    audiofile = args.audiofile
    audio_folder = args.audio_folder

    # Initialize the model once, outside the loop
    if model_name == 'model1':
        save_key = 'exp1multif0'
        model_path = "./models/{}.pkl".format(save_key)
        model = models.build_model1()
        load_weights(model, model_path)
        thresh = 0.4

    elif model_name == 'model2':
        save_key = 'exp2multif0'
        model_path = "./models/{}.pkl".format(save_key)
        model = models.build_model2()
        load_weights(model, model_path)
        thresh = 0.5

    elif model_name == 'model3':
        save_key = 'exp3multif0'
        model_path = "./models/{}.pkl".format(save_key)
        model = models.build_model3()
        load_weights(model, model_path)
        thresh = 0.5

    elif model_name == 'model4':
        save_key = 'exp4multif0'
        model_path = "./models/{}.pkl".format(save_key)
        model = models.build_model3()
        load_weights(model, model_path)
        thresh = 0.4

    elif model_name == 'model7':
        save_key = 'exp7multif0'
        model_path = "./models/{}.pkl".format(save_key)
        model = models.build_model3_mag()
        load_weights(model, model_path)
        thresh = 0.4

    else:
        raise ValueError("Specified model must be model1, model2, or model3.")

    # Compile the model once
    model.compile(
        loss=utils_train.bkld, metrics=['mse', utils_train.soft_binary_accuracy],
        optimizer='adam'
    )
    print("Model compiled")

    # Select operation mode and compute predictions
    if audiofile != "0":
#        process_single_audio_file(model, audiofile, audio_folder, model_name, thresh)
        process_single_audio_file_split(model, audiofile, audio_folder, model_name, thresh) #changeID 19
    elif audio_folder != "0":
        process_audio_folder(model, audio_folder, model_name, thresh) #changeID 15, ChangeID 16,changeID 17
#        process_audio_folder_parallel(model, audio_folder, model_name, thresh, num_threads=4) #changeID 15, ChangeID 16, changeID 17
    else:
        raise ValueError("One of audiofile and audio_folder must be specified.")

    # Clear session only once after processing everything
    K.clear_session()
    print("Session cleared after processing all files.")

def process_single_audio_file(model, audiofile, audio_folder, model_name, thresh):
    """Process a single audio file and save predictions."""
    print("Processing file:", audiofile)
    if model_name == 'model7':
        predicted_output, _ = get_single_test_prediction_phase_free(
            model, audio_file=os.path.join(audio_folder, audiofile)

        )
    else:
        predicted_output, _, _ = get_single_test_prediction(
            model, audio_file=audiofile
        )

    predicted_output = predicted_output.astype(np.float32)
    est_times, est_freqs = utils_train.pitch_activations_to_mf0(predicted_output, thresh)
    save_predictions(audiofile, est_times, est_freqs)

def process_single_audio_file_split(model, audiofile, audio_folder, model_name, thresh): #changeID 19
    """Process a single audio file and save predictions."""
    print("Processing file:", audiofile)

    # Check the audio file's duration
    # audio_path = os.path.join(audio_folder, audiofile)
    # audio = AudioSegment.from_file(audio_path)
    audio = AudioSegment.from_file(audiofile)
    max_duration_ms = 5 * 60 * 1000  # 5 minutes in milliseconds

    if len(audio) > max_duration_ms:
        print(f"Audio file {audiofile} exceeds 5 minutes. Splitting into chunks...")
        # chunks = split_audio_file(audio_path, max_duration_ms)
        chunks = split_audio_file(audiofile, max_duration_ms)
        total_frames = 0
        combined_est_freqs = []

        for idx, chunk_path in enumerate(chunks):
            print(f"Processing chunk {idx + 1}/{len(chunks)}")
            if model_name == 'model7':
                predicted_output, _ = get_single_test_prediction_phase_free(model, audio_file=chunk_path)
            else:
                predicted_output, _, _ = get_single_test_prediction(model, audio_file=chunk_path)

            predicted_output = predicted_output.astype(np.float32)
            est_times, est_freqs = utils_train.pitch_activations_to_mf0(predicted_output, thresh)
            
            combined_est_freqs.extend(est_freqs)
            total_frames += len(est_times)  # Increment the total frame count
            
            # Clean up the temporary chunk file
            os.remove(chunk_path)
        
        # Recreate `est_times` with a consistent step size
        step_size = est_times[1] - est_times[0]  # Step size from the first chunk
        combined_est_times = [i * step_size for i in range(total_frames)]
        
        # Save combined predictions
        save_predictions(audiofile, combined_est_times, combined_est_freqs)
    else:
        # Process normally if the file is under 5 minutes
        if model_name == 'model7':
            predicted_output, _ = get_single_test_prediction_phase_free(
                model, audio_file=os.path.join(audio_folder, audiofile)
            )
        else:
            predicted_output, _, _ = get_single_test_prediction(
                model, audio_file=audiofile
            )

        predicted_output = predicted_output.astype(np.float32)
        est_times, est_freqs = utils_train.pitch_activations_to_mf0(predicted_output, thresh)
        save_predictions(audiofile, est_times, est_freqs)


def process_audio_folder(model, audio_folder, model_name, thresh): #changeID 8
    """
    Process all valid audio files in a folder, skipping files if their 
    extraction already exists in the output folder.
    """
    print(f"Processing folder: {audio_folder}")

    # Define output directory
    output_dir = os.path.join(audio_folder, 'multif0')
    os.makedirs(output_dir, exist_ok=True)

    # Filter only .wav files in the folder
    audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]#[::-1] #changeID 13

    total_files = len(audio_files)

    if not audio_files:
        print("No audio files found in the folder.")
        return

    print(f"Found {total_files} audio files to process.")

    for idx, audiofile in enumerate(audio_files, start=1):
        # Check if the output CSV already exists
        output_path = os.path.join(output_dir, audiofile.replace('.wav', '.csv'))
        if os.path.exists(output_path):
            print(f"[{idx}/{total_files}] Skipping {audiofile}, output already exists.")
            continue

        print(f"[{idx}/{total_files}] Processing file: {audiofile}")
        try:
            process_single_audio_file_split(model, os.path.join(audio_folder, audiofile), output_dir, model_name, thresh) #changeID 19
#            process_single_audio_file(model, os.path.join(audio_folder, audiofile), output_dir, model_name, thresh)
        except Exception as e:
            print(f"Error processing {audiofile}") #changeID 12
#            print(f"Error processing {audiofile}: {e}") #changeID 17
            continue

    print("Processing complete.")


def save_predictions(audiofile, est_times, est_freqs):
    """Save predictions to CSV."""
    for i, (tms, fqs) in enumerate(zip(est_times, est_freqs)):
        if any(fqs <= 0):
            est_freqs[i] = np.array([f for f in fqs if f > 0])

    output_dir = os.path.dirname(audiofile) + '/multif0'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(audiofile).replace('wav', 'csv'))

    utils_train.save_multif0_output(est_times, est_freqs, output_path)
    print(f"Predictions saved for {audiofile} at {output_path}.")


def get_single_test_prediction_phase_free(model, audio_file=None):
    """Generate output from a model given an input numpy file
    """

    if audio_file != None: #changeID 6
        # should not be the case
        pump = utils.create_pump_object()
        features = utils.compute_pump_features(pump, audio_file)
        input_hcqt = features['dphase/mag'][0]


    else:
        raise ValueError("one of npy_file or audio_file must be specified")

    input_hcqt = input_hcqt.transpose(1, 2, 0)[np.newaxis, :, :, :]

    n_t = input_hcqt.shape[2]
    t_slices = list(np.arange(0, n_t, 5000))
    output_list = []
    # we need two inputs
    for t in t_slices:
        p = model.predict(np.transpose(input_hcqt[:, :, t:t+5000, :], (0, 1, 3, 2)))[0, :, :]

        output_list.append(p)
    K.clear_session() #changeID 7

    predicted_output = np.hstack(output_list)
    return predicted_output, input_hcqt

def get_single_test_prediction(model, audio_file=None):
    """Generate output from a model given an input numpy file.
       Part of this function is part of deepsalience
    """

    if audio_file != None: #changeID 6

        pump = utils.create_pump_object()
        features = utils.compute_pump_features(pump, audio_file)
        input_hcqt = features['dphase/mag'][0]
        input_dphase = features['dphase/dphase'][0]

    else:
        raise ValueError("One audio_file must be specified")

    input_hcqt = input_hcqt.transpose(1, 2, 0)[np.newaxis, :, :, :]
    input_dphase = input_dphase.transpose(1, 2, 0)[np.newaxis, :, :, :]

    n_t = input_hcqt.shape[2]
    t_slices = list(np.arange(0, n_t, 5000))
    output_list = []

    for t in t_slices:
        p = model.predict([np.transpose(input_hcqt[:, :, t:t+5000, :], (0, 1, 3, 2)),
                           np.transpose(input_dphase[:, :, t:t+5000, :], (0, 1, 3, 2))]
                          )[0, :, :]

        output_list.append(p)

    predicted_output = np.hstack(output_list)
    return predicted_output, input_hcqt, input_dphase


def process_audio_folder_parallel(model, audio_folder, model_name, thresh, num_threads=4): #changeID 15
    """
    Process all valid audio files in a folder in parallel, skipping files if their 
    extraction already exists in the output folder.
    """
    print(f"Processing folder in parallel: {audio_folder}")

    # Define output directory
    output_dir = os.path.join(audio_folder, 'multif0')
    os.makedirs(output_dir, exist_ok=True)

    # Filter only .wav files in the folder
    audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')][::-1]
    total_files = len(audio_files)

    if not audio_files:
        print("No audio files found in the folder.")
        return

    print(f"Found {total_files} audio files to process.")

    # Define a helper function for processing a single file
    def process_file(audiofile):
        try:
            # Check if the output CSV already exists
            output_path = os.path.join(output_dir, audiofile.replace('.wav', '.csv'))
            if os.path.exists(output_path):
                return f"Skipping {audiofile}, output already exists."
            process_single_audio_file_split(model, os.path.join(audio_folder, audiofile), output_dir, model_name, thresh) #changeID 19
#            process_single_audio_file(model, os.path.join(audio_folder, audiofile), output_dir, model_name, thresh)
            return f"Processed {audiofile}"
        except Exception as e:
            return f"Error processing {audiofile}: {str(e)}"

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(process_file, audiofile): audiofile for audiofile in audio_files}

        for future in as_completed(futures):
            print(future.result())

    print("Parallel processing complete.")

def split_audio_file(audio_path, max_duration_ms): #changeID 19
    """
    Splits an audio file into chunks of `max_duration_ms` milliseconds.
    Returns a list of temporary file paths for the chunks.
    """
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    for i in range(0, len(audio), max_duration_ms):
        chunk = audio[i:i + max_duration_ms]
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        chunk.export(temp_file.name, format="wav")
        chunks.append(temp_file.name)
    return chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict multiple F0 output of an input audio file or all the audio files inside a folder.")

    parser.add_argument("--model",
                        dest='model_name',
                        type=str,
                        help="Specify the ID of the model"
                             "to use for the prediction: model1 (Early/Deep) / "
                             "model2 (Early/Shallow) / "
                             "model3 (Late/Deep, recommended)")

    parser.add_argument("--audiofile",
                        dest='audiofile',
                        default="0",
                        type=str,
                        help="Path to the audio file to analyze. If using the folder mode, this should be skipped.")

    parser.add_argument("--audio_folder",
                        dest='audio_folder',
                        default="0",
                        type=str,
                        help="Directory with audio files to analyze. If using the audiofile mode, this should be skipped.")

    main(parser.parse_args())

