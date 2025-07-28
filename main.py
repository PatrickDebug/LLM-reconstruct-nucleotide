# ##################################################################################
# main_machine_custom_loss_7_single : This is to build models target single position 

import re
from collections import defaultdict
from io import BytesIO, StringIO
from pathlib import Path
import matplotlib.pyplot as plt
from Bio import AlignIO, Phylo, SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqIO.FastaIO import SimpleFastaParser
from ncbi.datasets import GeneApi 
import os
import random
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, RepeatVector
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import register_keras_serializable
from datetime import datetime

samplesize = int(sys.argv[1])
infilename = sys.argv[2]
targetmodel = sys.argv[3]
targetweight = int(sys.argv[4])

dfm1 = pd.read_csv(infilename)

nucleotide_to_int = {'a': 0, 'c': 1, 'g': 2, 't': 3}
int_to_nucleotide = {0: 'a', 1: 'c', 2: 'g', 3: 't'}
num_classes = 4
seq_length = 16
NV_Len = 32
epoch = 25
lr = 0.0005
batch = 32 
count_loss_ratio = 4*NV_Len

for i in range(15):

    modeltype = f'{targetmodel}_{i}'
    dfm = dfm1[dfm1['maskpos'] == i]
    datasize = len(dfm)
    if datasize > samplesize:
        df_build = dfm.sample(samplesize, random_state=2008).reset_index(drop=True).copy()
    else:
        df_build = dfm.copy()

    sequences = df_build['subseq'].apply(lambda x: [nucleotide_to_int[nuc] for nuc in x])
    one_hot_sequences = np.array([to_categorical(seq, num_classes=num_classes) for seq in sequences])
    featurelist = ['n_a', 'n_c', 'n_t', 'n_g', 'mu_a', 'mu_c', 'mu_t', 'mu_g', 'moment_a_2', 'moment_c_2', 'moment_t_2', 'moment_g_2',
                   'moment_a_3', 'moment_c_3', 'moment_t_3', 'moment_g_3', 'moment_a_4', 'moment_c_4', 'moment_t_4', 'moment_g_4',
                   'moment_a_5', 'moment_c_5', 'moment_t_5', 'moment_g_5', 'moment_a_6', 'moment_c_6', 'moment_t_6', 'moment_g_6',
                   'moment_a_7', 'moment_c_7', 'moment_t_7', 'moment_g_7']
    features = df_build[featurelist].copy()
    X_train, X_temp, y_train, y_temp, mask_train, mask_temp = train_test_split(features.values, one_hot_sequences, df_build['maskpos'], test_size=0.25, random_state=123)
    X_val, X_test, y_val, y_test, mask_val, mask_test = train_test_split(X_temp, y_temp,mask_temp, test_size=0.2, random_state=123)
    # Reshape features for LSTM input
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(256, input_shape=(NV_Len, 1)))
    model.add(RepeatVector(seq_length))
    model.add(LSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(num_classes, activation='softmax')))
    
    # Use a lower learning rate and gradient clipping
    optimizer = Adam(learning_rate=lr, clipnorm=1.0)

    # Custom loss function
    def mask_loss(mask_train, weight):
        def custom_loss(y_true, y_pred):
            # Specify the weight of each loss
            cross_entropy_weight = (100-weight)/200
            count_penalty_weight = (100-weight)/200
            count_mask_weight    = weight/100
            #print(f'cross_entropy_weight:{cross_entropy_weight} - count_penalty_weight:{count_penalty_weight} - count_mask_weight:{count_mask_weight}')
            
            # Standard categorical crossentropy loss
            cross_entropy_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    
            # Convert y_pred from one-hot encoded to integer sequences
            y_pred_int = tf.argmax(y_pred, axis=-1)
    
            # Count predicted nucleotides
            count_A_pred = tf.reduce_sum(tf.cast(tf.equal(y_pred_int, 0), tf.float32), axis=1)
            count_C_pred = tf.reduce_sum(tf.cast(tf.equal(y_pred_int, 1), tf.float32), axis=1)
            count_G_pred = tf.reduce_sum(tf.cast(tf.equal(y_pred_int, 2), tf.float32), axis=1)
            count_T_pred = tf.reduce_sum(tf.cast(tf.equal(y_pred_int, 3), tf.float32), axis=1)
    
            # Convert y_pred from one-hot encoded to integer sequences
            y_true_int = tf.argmax(y_true, axis=-1)
    
            # Count predicted nucleotides
            count_A_true = tf.reduce_sum(tf.cast(tf.equal(y_true_int, 0), tf.float32), axis=1)
            count_C_true = tf.reduce_sum(tf.cast(tf.equal(y_true_int, 1), tf.float32), axis=1)
            count_G_true = tf.reduce_sum(tf.cast(tf.equal(y_true_int, 2), tf.float32), axis=1)
            count_T_true = tf.reduce_sum(tf.cast(tf.equal(y_true_int, 3), tf.float32), axis=1)
    
            # Compute the count penalty (mean squared error)
            count_penalty = tf.reduce_mean(tf.square(count_A_pred - count_A_true) +
                                           tf.square(count_C_pred - count_C_true) +
                                           tf.square(count_G_pred - count_G_true) +
                                           tf.square(count_T_pred - count_T_true))
    
            # Compute how many predicted wrong at the masked position
            y_true_list = y_true_int.numpy().tolist()
            y_pred_list = y_pred_int.numpy().tolist()
            count_mask_wrong = sum([ 1 for yt, yp, m in zip(y_true_list, y_pred_list, (mask_train+1).fillna(0)) if yt[int(m)-1] != yt[int(m)-1] and int(m) > 0 ])
           
            # Combine the cross-entropy loss and the count penalty
            total_loss = cross_entropy_weight*cross_entropy_loss + count_penalty_weight*(count_penalty/count_loss_ratio) + count_mask_weight*(count_mask_wrong/len(y_true_int))
    
            return total_loss
        return custom_loss

    datetimestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    # modeltype = 'lstm_method2'
    
    # setup log directory
    logdir = 'log'
    os.makedirs(logdir, exist_ok = True)
    
    # setup output directory
    outdir = 'output_folder'
    os.makedirs(outdir, exist_ok = True)
    
    # setup model directory
    modeldir = 'saved_model'
    os.makedirs(modeldir, exist_ok = True)
    # modeling history report
    reportmhrfile = f'{outdir}/modeling_history.csv'
    mhr_column = ['date', 'time', 'model', 'input', 'sample_size', 'training_size', 'test_size', 'sequence_length', 'NV_length',
                  'epoches', 'training_start', 'training_end', 'training_time', 'training_report', 'performance_report', 'log_file']
    # model log file
    logfile = f'{logdir}/{modeltype}_{datetimestamp}.log'
    orig_stdout = sys.stdout
    f = open(logfile, 'a')
    sys.stdout = f
    # model file
    modelfile = f'{modeldir}/{modeltype}_{datetimestamp}'
    outdir = 'output_folder'
    os.makedirs(outdir, exist_ok = True)
    
    thisdate = datetimestamp[:4] + '_' + datetimestamp[4:6] + '_' + datetimestamp[6:8]
    thistime = datetimestamp[8:10] + ':' + datetimestamp[10:12] + ':' + datetimestamp[12:14]
    inputfile = f'{infilename}_position{i}'


    # Compile the model using the custom loss function
    model.compile(optimizer=optimizer, loss=mask_loss(mask_train=mask_train, weight=targetweight), metrics=['accuracy'], run_eagerly=True)
    #model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Define the checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        filepath=modelfile,
        monitor='val_loss',  # Monitor validation loss
        save_best_only=True,  # Save only when the validation loss improves
        save_weights_only=False,  # Save the entire model, not just the weights
        mode='min',  # Save when the monitored metric is minimized
        verbose=1  # Print a message when the model is saved
    )

    # Callbacks for early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7)
    
    training_start = datetime.now()
    
    history = model.fit(
        X_train, y_train,
        epochs=epoch,
        batch_size=batch,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr, checkpoint_callback]
    )
    #custom_objects = {'custom_loss': custom_loss}
    #best_model = load_model('saved_model/500k10epoch.test', custom_objects=custom_objects)
    history_data = {
        "epoch": list(range(1, len(history.history['accuracy']) + 1)),
        "accuracy": history.history['accuracy'],
        "loss": history.history['loss'],
        "val_accuracy": history.history['val_accuracy'],
        "val_loss": history.history['val_loss'],
    }

    training_end = datetime.now()
    training_time = training_end - training_start

    # Create a DataFrame from the history data
    df_history = pd.DataFrame(history_data)

    # Output folder path
    output_folder = 'output_folder'

    # Generate the output file path with the timestamp
    output_file = os.path.join(output_folder, f"training_history_{datetimestamp}.csv")

    # Save DataFrame to a CSV file in the output folder
    df_history.to_csv(output_file, index=False)
    import matplotlib.pyplot as plt

    # Assuming 'history' contains the training history
    epochs = range(1, len(history.history['accuracy']) + 1)
    
    plt.figure(figsize=(8, 6))
    
    # Plot training accuracy
    plt.plot(epochs, history.history['accuracy'], 'g-*', label='Training Accuracy')
    
    # Plot validation accuracy
    plt.plot(epochs, history.history['val_accuracy'], 'b-o', label='Validation Accuracy')
    
    plt.title('Training Accuracy and Validation Accuracy Vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save the plot as an image
    plt.savefig(f"output_folder/accuracy_vs_epochs_{datetimestamp}.png")
    loss, accuracy = model.evaluate(X_test, y_test)
    predictions = model.predict(X_test)
    # Convert predictions and y_test from one-hot encoding to nucleotide sequences
    predicted_nucleotides = np.argmax(predictions, axis=-1)
    original_nucleotides = np.argmax(y_test, axis=-1)

    # Convert integers back to nucleotide characters
    predicted_sequences = [''.join([int_to_nucleotide[nuc] for nuc in seq]) for seq in predicted_nucleotides]
    original_sequences = [''.join([int_to_nucleotide[nuc] for nuc in seq]) for seq in original_nucleotides]
    # Calculate the error rate
    total_nucleotides = len(predicted_sequences) * seq_length

    # Calculate the error rate
    incorrect_nucleotides = sum(p != o for p_seq, o_seq in zip(predicted_sequences, original_sequences) for p, o in zip(p_seq, o_seq))
    error_rate = incorrect_nucleotides / total_nucleotides
    df_result = pd.DataFrame(zip(mask_test,predicted_sequences, original_sequences), columns = ['Position', 'Predicted', 'Original'])
    df_result = pd.DataFrame(zip((mask_test+1).fillna(0), predicted_sequences, original_sequences), columns=['Position', 'Predicted', 'Original'])
    
    df_result['check'] = df_result.apply(lambda x: [p == o for p, o in zip(x['Predicted'], x['Original'])], axis=1)
    
    df_result['mask_correct'] = df_result[['Position', 'check']].apply(lambda x: 0 if x['Position'] == 0 else (1 if x['check'][int(x['Position']-1)] else 0), axis=1)
    df_result['mask_wrong'] = np.where(df_result['Position'] == 0, 0, 1 - df_result['mask_correct'])
    
    df_result['other_correct'] = df_result[['mask_correct', 'check']].apply(lambda x: sum([1 if p == True else 0 for p in x['check']])-x['mask_correct'], axis=1)
    df_result['other_wrong'] = np.where(df_result['Position'] == 0, seq_length - df_result['other_correct'], seq_length - 1 - df_result['other_correct'])
    
    df_result_group = df_result.groupby('Position')[['mask_correct', 'mask_wrong', 'other_correct', 'other_wrong']].sum().reset_index()
    df_result_group['mask_accuracy'] = np.where(df_result_group['Position'] == 0, 0.0,
                                                df_result_group['mask_correct'] / (df_result_group['mask_correct'] + df_result_group['mask_wrong']))
    df_result_group['other_accuracy'] = df_result_group['other_correct'] / (df_result_group['other_correct'] + df_result_group['other_wrong'])
    df_result_group['num_seq'] = (df_result_group['mask_correct'] + df_result_group['mask_wrong'] +
                                  df_result_group['other_correct'] + df_result_group['other_wrong']) / seq_length
    df_result_group.to_csv('PerformanceByMaskPosition.csv', index=False)
    df_result_group['overall_accuracy'] = (df_result_group['mask_correct']+df_result_group['other_correct'])/(df_result_group['mask_correct']+df_result_group['mask_wrong']+df_result_group['other_correct']+df_result_group['other_wrong'])
    output_folder = 'output_folder'
    output_file = os.path.join(output_folder, f"mask_history_{datetimestamp}.csv")
    predictedfile = f"{output_folder}/{modeltype}_{datetimestamp}_prediction.csv"

    # Save DataFrame to a CSV file in the output folder
    df_result_group.to_csv(output_file, index=False)
    df_result.to_csv(predictedfile, index=False)

    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot mask accuracy, other accuracy, and overall accuracy on the primary y-axis
    ax1.plot(df_result_group['Position'], df_result_group['mask_accuracy'], 'g-o', label='Mask Accuracy')
    ax1.plot(df_result_group['Position'], df_result_group['other_accuracy'], 'b-s', label='Other Accuracy')
    ax1.plot(df_result_group['Position'], df_result_group['overall_accuracy'], 'r-^', label='Overall Accuracy')

    # Set the labels and legends for the primary y-axis
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='upper left')
    ax1.set_title('Comparison of Accuracies and Number of Sequences by Mask Position')
    
    # Save the plot as an image
    plt.savefig(f"output_folder/accuracies_vs_position_{datetimestamp}.png")

    # save modeling record
    r = {'date': thisdate, 'time': thistime, 'model': modeltype, 'input': inputfile, 
         'sample_size': df_build.shape[0], 'training_size': X_train.shape[0], 'test_size': X_test.shape[0], 
         'sequence_length': seq_length, 'NV_length': NV_Len, 'epoches': epoch, 
         'training_start': training_start, 'training_end': training_end, 'training_time': training_time, 
         'training_report': f"training_history_{datetimestamp}.csv", 
         'performance_report': f"mask_history_{datetimestamp}.csv", 
         'log_file': logfile}
    rpt = pd.DataFrame(r, index=[0])
    if os.path.exists(reportmhrfile):
        rpt.to_csv(reportmhrfile, mode='a', header=False, index=False)
    else:
        rpt.to_csv(reportmhrfile, mode='w', index=False)

    # Function to calculate average metrics and maximum mismatches
    def calculate_metrics(original_sequences, predicted_sequences):
        total_matches = 0
        total_mismatches = 0
        total_characters = 0
        max_mismatches = 0
        max_mismatch_record = None
    
        for orig_seq, pred_seq in zip(original_sequences, predicted_sequences):
            match_count = sum(1 for orig_char, pred_char in zip(orig_seq, pred_seq) if orig_char == pred_char)
            mismatch_count = len(orig_seq) - match_count
    
            total_matches += match_count
            total_mismatches += mismatch_count
            total_characters += len(orig_seq)
    
            if mismatch_count > max_mismatches:
                max_mismatches = mismatch_count
                max_mismatch_record = (orig_seq, pred_seq)
    
        average_matches = total_matches / len(original_sequences)
        average_mismatches = total_mismatches / len(original_sequences)
        accuracy = total_matches / total_characters
        mismatch_rate = total_mismatches / total_characters
    
        return {
            "average_matches": average_matches,
            "average_mismatches": average_mismatches,
            "accuracy": accuracy,
            "mismatch_rate": mismatch_rate,
            "max_mismatches": max_mismatches,
            "max_mismatch_record": max_mismatch_record
        }
    # Calculate and print the metrics
    metrics = calculate_metrics(original_sequences, predicted_sequences)
    
    print(f'Average Matches: {metrics["average_matches"]:.2f}')
    print(f'Average Mismatches: {metrics["average_mismatches"]:.2f}')
    print(f'Overall Accuracy: {metrics["accuracy"]:.4f}')
    print(f'Overall Mismatch Rate: {metrics["mismatch_rate"]:.4f}')
    print(f'Maximum Mismatches: {metrics["max_mismatches"]}')
    print(f'Original  sequence with maximum mismatches: {metrics["max_mismatch_record"][0]}')
    print(f'Predicted sequence with maximum mismatches: {metrics["max_mismatch_record"][1]}')
    
    
    sys.stdout = orig_stdout
    f.close()
