import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import time
from torchvision.io import read_image
import pandas as pd
from sklearn.model_selection import train_test_split
from model import CNN_network
from functions.load_machine_config import load_machine_config
from utils.data_loader import load_image_data
import seaborn as sn

config = load_machine_config()

def normalize_all_data(X, method='zscore'):
    """
    Apply global normalization to all data points collectively.
    
    Args:
        X (numpy.ndarray): Input data with shape [samples, 1, time_steps, features]
        method (str): Normalization method ('zscore' or 'minmax')
        
    Returns:
        numpy.ndarray: Normalized data
    """
    # Make a copy to avoid modifying the original data
    X_normalized = X.copy()
    
    if method == 'zscore':
        # Z-score normalization across all data points
        global_mean = np.mean(X_normalized)
        global_std = np.std(X_normalized)
        
        # Avoid division by zero
        if global_std == 0:
            global_std = 1.0
        
        # Apply normalization globally
        X_normalized = (X_normalized - global_mean) / global_std
        
    elif method == 'minmax':
        # Min-max normalization to [0, 1] range across all data points
        global_min = np.min(X_normalized)
        global_max = np.max(X_normalized)
        
        # Avoid division by zero
        if global_max == global_min:
            X_normalized = np.zeros_like(X_normalized)  # Set all to zero if no range
        else:
            # Apply normalization globally
            X_normalized = (X_normalized - global_min) / (global_max - global_min)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return X_normalized

def prepare_emo_samples(feature_data, emo_idx):
    """
    Prepare input samples for a specific emo.
    
    Args:
        feature_data (list): List of feature matrices for a emo
        emo_idx (int): Index of the emo
        
    Returns:
        tuple: Prepared data and labels
    """
    if not feature_data:
        print("Warning: No feature data provided")
        return np.array([]), np.array([])
    
    # Initialize with first two samples
    if len(feature_data) < 2:
        print("Warning: Not enough samples for emo")
        return np.array([]), np.array([])
    
    # Convert to proper format for CNN input
    # CNN expects input of shape [batch_size, channels, height, width]
    # Our data is [time_steps, features], so we need to reshape
    samples = []
    labels = []
    
    for i in range(len(feature_data)):
        # # Add a channel dimension (1 channel)
        # # Original: [41, 90] -> Reshaped: [1, 41, 90]
        # sample = np.expand_dims(feature_data[i], axis=0)
        samples.append(feature_data[i])
        labels.append(emo_idx)
    
    return np.array(samples), np.array(labels)

def combine_all_emo_data(all_emo_data, emo_indices):
    """
    Combine data from all emos into training and testing datasets.
    
    Args:
        all_emo_data (list): List of [feature_data, emotion] pairs
        emo_indices (dict): Dictionary mapping emotions to indices
        
    Returns:
        tuple: Combined X and Y data
    """
    if not all_emo_data:
        print("Error: No emo data provided")
        return np.array([]), np.array([])
    
    all_samples = []
    all_labels = []
    
    for feature_data, emotion in all_emo_data:
        emo_idx = emo_indices[emotion]
        samples, labels = prepare_emo_samples(feature_data, emo_idx)
        
        if samples.size > 0:
            all_samples.extend(samples)
            all_labels.extend(labels)
    
    return np.array(all_samples), np.array(all_labels)

work_directory = config["data_dir"] + "Emotion/"

# Configure dataset parameters
actuators = ["ur3e/joint/figures"]
usernames = ["u0", "u2", "u3", "u4", "u5", "u7", "u8", "u9", "u10", "u11"]
usernames_all_tasks = ["u2", "u7", "u8", "u9"]
emotions = ["a", "p", "s", "j", "n"]
tasks = ["lw"]
free_tasks = ["lw", "drink", "knock", "s", "star", "stir", "throw", "tri", "wave"]
ref_tasks = ["lw", "s", "star", "stir", "tri"]
postures = ["free", "ref"]
num_instances = 20 # situations that have no 20 instances will be skipped

# Create a mapping of emotions to indices
emo_indices = {emotion: idx for idx, emotion in enumerate(emotions)}

# Load data for all emos
print("Loading data...")

# For simplicity, using only first actuator and task
actuator = actuators[0]
task = tasks[0]
posture = postures[1] # 0 for 'free' 1 for 'ref'

all_user_correct = 0
all_user_total = 0
all_user_array = np.zeros((len(emotions), len(emotions)))

total_bin_confidences = []
total_bin_accuracies = []
total_bin_counts = []

for username in usernames:
    all_emo_data = []

    for emotion in emotions:
        feature_data = load_image_data(
            work_directory, actuator, username, emotion, task, posture, num_instances
        )
        all_emo_data.append([feature_data, emotion])

    # Combine data from all emos
    X, Y = combine_all_emo_data(all_emo_data, emo_indices)
    print(f"Data loaded with shape X: {X.shape}, Y: {Y.shape}")

    # Apply global normalization to all data points collectively
    # X = normalize_all_data(X, method='zscore')  # Use 'minmax' for [0, 1] range
    # print(f"Data normalized globally using z-score method")

    # Configure device (CPU or GPU)
    device = config["compdev"]
    print(f"Using device: {device}")

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

    # Convert to PyTorch tensors
    x_train = torch.from_numpy(x_train).float().to(device)
    x_test = torch.from_numpy(x_test).float().to(device)
    y_train = torch.from_numpy(y_train).long().to(device)
    y_test = torch.from_numpy(y_test).long().to(device)

    # Initialize CNN model
    input_size = 150  # time steps dimension
    cnn = CNN_network.CNNet(input_size, batch_size=8, num_class=len(emotions), epochs=100)
    cnn = cnn.to(device)
    print('Training device:', next(cnn.parameters()).device)

    # Prepare data loaders
    train_loader, test_loader = cnn.prepare_data_loaders(x_train, x_test, y_train, y_test)

    # Train model
    print("Starting training...")
    start_time = time.time()
    CNN_network.train_model(cnn, train_loader)  # Renamed from 'fit'
    train_time = time.time()
    print(f"Training time used: {train_time-start_time:.2f} seconds")


    target_names = ["Annoyance", "Pleasure", "Sadness", "Joy", "Neutral"]
    # target_names = ["Annoyance", "Joy", "Sadness", "Pleasure", "Neutral"]
    # Evaluate model
    print("Evaluating model...")
    with torch.no_grad():
        correct, total, array, bin_confidences, bin_accuracies, bin_counts = CNN_network.evaluate_model(cnn, test_loader, target_names)  # Renamed from 'eva'
    all_user_correct += correct
    all_user_total += total
    all_user_array = all_user_array + array
    total_bin_confidences.append(bin_confidences)
    total_bin_accuracies.append(bin_accuracies)
    total_bin_counts.append(bin_counts)
    test_time = time.time()
    print(f"Testing time used: {test_time-train_time:.2f} seconds")

    # Save model
    model_save_path = os.path.join(work_directory, f"segments/{actuator}/{task}_{posture}_cnn.pt")
    torch.save(cnn, model_save_path)
    print(f"Model saved to {model_save_path}")

    # plt.show()

    # Report user-level accuracy
    user_accuracy = 100 * correct / total if total > 0 else 0
    print(f'User: {username}, Correct: {correct}, User accuracy: {user_accuracy:.3f}%')

overall_accuracy = 100 * all_user_correct / all_user_total if all_user_total > 0 else 0
print(f'Overall Correct: {all_user_correct}, Overall accuracy: {overall_accuracy:.3f}%')

array_norm = np.around(all_user_array.astype('float') / np.sum(all_user_array, axis=1)[:, None], decimals=3)
df_cm_norm = pd.DataFrame(
        array_norm,
        index=target_names,
        columns=target_names
    )

# Generate a classification report with precision, recall, f1-score
classification_report = pd.DataFrame(columns=['Precision', 'Recall', 'F1-Score'])
for i, emotion in enumerate(target_names):
    true_positives = all_user_array[i, i]
    false_positives = np.sum(all_user_array[:, i]) - true_positives
    false_negatives = np.sum(all_user_array[i, :]) - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    classification_report.loc[emotion] = [round(precision, 3), round(recall, 3), round(f1_score, 3)]
print("Classification Report:")
print(classification_report)

print("Confusion Matrix (counts):")
print(df_cm_norm)

# Plot normalized confusion matrix (predicted on x, true on y)
plt.figure(figsize=(6, 4))
sn.heatmap(df_cm_norm, annot=True, cmap='Blues', fmt='.3f',
            xticklabels=df_cm_norm.columns, yticklabels=df_cm_norm.index)
plt.xlabel('Predicted Emotion')
plt.ylabel('True Emotion')
plt.tight_layout()

plt.title('Confusion Matrix')
plt.savefig(f"results/{task}_{posture}_cnn_confusion_matrix.png")

print("Total Bin Confidences:", total_bin_confidences)
print("Total Bin Accuracies:", total_bin_accuracies)
# average across users and show a shaded band
conf_avg = np.nanmean(np.array(total_bin_confidences), axis=0)
acc_avg = np.nanmean(np.array(total_bin_accuracies), axis=0)
print("Conf Avg:", conf_avg)
print("Acc Avg:", acc_avg)
count_sum = np.nansum(np.array(total_bin_counts), axis=0)
mask = ~np.isnan(count_sum)
conf_avg = conf_avg[mask]
acc_avg = acc_avg[mask]
plt.figure(figsize=(6, 4))
plt.plot(conf_avg, acc_avg, marker='o', label='Calibration Curve')
plt.fill_between(conf_avg, acc_avg - np.std(np.array(total_bin_accuracies), axis=0),
                 acc_avg + np.std(np.array(total_bin_accuracies), axis=0), color='b', alpha=0.2)
plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Perfect Calibration')
plt.xlabel('Confidence')
plt.ylabel('Accuracy')
# plt.title('Calibration Curve')

plt.show()