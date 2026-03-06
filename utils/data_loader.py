import os
import numpy as np
import pandas as pd
from torchvision.io import read_image

def load_feature_data(work_directory, sessionname, tablename, username, fingername, featurename, num_instances):
    feature_data = []
    for idx in range(1, num_instances + 1):
        file_path = os.path.join(work_directory, "segments", sessionname, tablename, username, fingername, featurename, f"touchscreen_featureVector_{idx}.csv")
        try:
            data = pd.read_csv(file_path, header=None).values
            data = np.nan_to_num(data, nan=0.0)
            feature_data.append(data)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error: {e}")
    return feature_data

def normalize_all_data(X, method='zscore'):
    X_normalized = X.copy()
    if method == 'zscore':
        mean, std = np.mean(X_normalized), np.std(X_normalized)
        std = 1.0 if std == 0 else std
        return (X_normalized - mean) / std
    elif method == 'minmax':
        min_, max_ = np.min(X_normalized), np.max(X_normalized)
        return (X_normalized - min_) / (max_ - min_) if max_ != min_ else np.zeros_like(X_normalized)
    elif method == 'none':
        return X_normalized
    else:
        raise ValueError("Unknown normalization method")

def prepare_user_verification_data(target_data, all_user_finger_data, target_key):
    genuine = [np.expand_dims(d, 0) for d in target_data]
    genuine_labels = np.ones(len(genuine))
    impostors = []
    for k, v in all_user_finger_data.items():
        if k != target_key:
            impostors.extend(np.expand_dims(d, 0) for d in v)
    impostor_sample = np.random.choice(len(impostors), len(genuine), replace=len(impostors) < len(genuine))
    impostor = [impostors[i] for i in impostor_sample] if isinstance(impostor_sample, np.ndarray) else impostors
    impostor_labels = np.zeros(len(impostor))
    X = np.array(genuine + impostor)
    y = np.concatenate([genuine_labels, impostor_labels])
    return X, y



def load_image_data(work_directory, actuator, username, emotion, task, posture, num_instances):
    """
    Load image from png files.
    No image-wise normalization is applied at this stage.
    
    Args:
        work_directory (str): Base directory path
        actuator (str): Name of the actuator
        emotion (str): Name of the emotion
        task (str): Name of the task
        posture (str): Name of the posture
        num_instances (int): Number of instances to load
        
    Returns:
        list: List of loaded image matrices
    """
    image_data = []
    
    for idx in range(1, num_instances+1):
        file_path = os.path.join(
            work_directory,
            "segments",
            actuator
        ) + "/" + username + "_" + emotion + "_" + task + "_" + posture + "_" + str(idx) + ".png"
        
        try:
            # Load image data (150 × 150 RGB)
            data = read_image(file_path)
            if data.shape[0] == 4:
                data = data[:3, :, :]  # Discard alpha channel
                # data = data.permute(1, 2, 0)  # Permute to (H, W, C) for displaying
            
            # # Replace NaN values with zeros
            # data = np.nan_to_num(data, nan=0.0)
            
            # Check for valid data
            if np.isfinite(data).all():
                image_data.append(data)
            else:
                print(f"Warning: Non-finite values found in {file_path}")
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                image_data.append(data)
                
        # except FileNotFoundError:
        #     print(f"Warning: File not found: {file_path}")
        #     continue
        except Exception as e:
            # print(f"Error loading file {file_path}: {e}")
            continue
            
    return image_data


def load_image_train_data_loso(work_directory, actuator, user, usernames, emotion, task, posture, num_instances):
    """
    Leave-one-subject-out (LOSO)
    Training data from all users except the target user.
    Test data from the target user.

    Load image from png files.
    No image-wise normalization is applied at this stage.
    
    Args:
        work_directory (str): Base directory path
        actuator (str): Name of the actuator
        emotion (str): Name of the emotion
        task (str): Name of the task
        posture (str): Name of the posture
        num_instances (int): Number of instances to load
        
    Returns:
        list: List of loaded image matrices
    """
    other_usernames = [ou for ou in usernames if ou != user]
    image_data = []

    # Add few-shot from target user to training set
    for idx in np.random.choice(range(0, int(num_instances/2)), 2, replace=False):
        file_path = os.path.join(
            work_directory,
            "segments",
            actuator
        ) + "/" + user + "_" + emotion + "_" + task + "_" + posture + "_" + str(idx) + ".png"
        try:
            # Load image data (150 × 150 RGB)
            data = read_image(file_path)
            if data.shape[0] == 4:
                data = data[:3, :, :]  # Discard alpha channel
                # data = data.permute(1, 2, 0)  # Permute to (H, W, C) for displaying
            
            # # Replace NaN values with zeros
            # data = np.nan_to_num(data, nan=0.0)
            
            # Check for valid data
            if np.isfinite(data).all():
                image_data.append(data)
            else:
                print(f"Warning: Non-finite values found in {file_path}")
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                image_data.append(data)
                
        # except FileNotFoundError:
        #     print(f"Warning: File not found: {file_path}")
        #     continue
        except Exception as e:
            # print(f"Error loading file {file_path}: {e}")
            continue
    
    for other_user in other_usernames:
        # for idx in range(1, num_instances+1):
        # random sample 2 instance from each other user
        sampled_indices = np.random.choice(range(0, int(num_instances/2)), 3, replace=False)
        for idx in sampled_indices:
            file_path = os.path.join(
                work_directory,
                "segments",
                actuator
            ) + "/" + other_user + "_" + emotion + "_" + task + "_" + posture + "_" + str(idx) + ".png"
            
            try:
                # Load image data (150 × 150 RGB)
                data = read_image(file_path)
                if data.shape[0] == 4:
                    data = data[:3, :, :]  # Discard alpha channel
                    # data = data.permute(1, 2, 0)  # Permute to (H, W, C) for displaying
                
                # # Replace NaN values with zeros
                # data = np.nan_to_num(data, nan=0.0)
                
                # Check for valid data
                if np.isfinite(data).all():
                    image_data.append(data)
                else:
                    print(f"Warning: Non-finite values found in {file_path}")
                    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                    image_data.append(data)
                    
            # except FileNotFoundError:
            #     print(f"Warning: File not found: {file_path}")
            #     continue
            except Exception as e:
                # print(f"Error loading file {file_path}: {e}")
                continue
                
    return image_data


def load_image_test_data_loso(work_directory, actuator, user, usernames, emotion, task, posture, num_instances):
    """
    Leave-one-subject-out (LOSO)
    Training data from all users except the target user.
    Test data from the target user.

    Load image from png files.
    No image-wise normalization is applied at this stage.
    
    Args:
        work_directory (str): Base directory path
        actuator (str): Name of the actuator
        emotion (str): Name of the emotion
        task (str): Name of the task
        posture (str): Name of the posture
        num_instances (int): Number of instances to load
        
    Returns:
        list: List of loaded image matrices
    """
    other_usernames = [ou for ou in usernames if ou != user]
    image_data = []

    # Add target user to training set
    for idx in range(int(num_instances/2), num_instances):
        file_path = os.path.join(
            work_directory,
            "segments",
            actuator
        ) + "/" + user + "_" + emotion + "_" + task + "_" + posture + "_" + str(idx) + ".png"
        try:
            # Load image data (150 × 150 RGB)
            data = read_image(file_path)
            if data.shape[0] == 4:
                data = data[:3, :, :]  # Discard alpha channel
                # data = data.permute(1, 2, 0)  # Permute to (H, W, C) for displaying
            
            # # Replace NaN values with zeros
            # data = np.nan_to_num(data, nan=0.0)
            
            # Check for valid data
            if np.isfinite(data).all():
                image_data.append(data)
            else:
                print(f"Warning: Non-finite values found in {file_path}")
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                image_data.append(data)
                
        # except FileNotFoundError:
        #     print(f"Warning: File not found: {file_path}")
        #     continue
        except Exception as e:
            # print(f"Error loading file {file_path}: {e}")
            continue
                
    # for other_user in other_usernames:
    #     # for idx in range(1, num_instances+1):
    #     # random sample 2 instance from each other user
    #     sampled_indices = np.random.choice(range(int(num_instances/2), num_instances), 3, replace=False)
    #     for idx in sampled_indices:
    #         file_path = os.path.join(
    #             work_directory,
    #             "segments",
    #             actuator
    #         ) + "/" + other_user + "_" + emotion + "_" + task + "_" + posture + "_" + str(idx) + ".png"
            
    #         try:
    #             # Load image data (150 × 150 RGB)
    #             data = read_image(file_path)
    #             if data.shape[0] == 4:
    #                 data = data[:3, :, :]  # Discard alpha channel
    #                 # data = data.permute(1, 2, 0)  # Permute to (H, W, C) for displaying
                
    #             # # Replace NaN values with zeros
    #             # data = np.nan_to_num(data, nan=0.0)
                
    #             # Check for valid data
    #             if np.isfinite(data).all():
    #                 image_data.append(data)
    #             else:
    #                 print(f"Warning: Non-finite values found in {file_path}")
    #                 data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    #                 image_data.append(data)
                    
    #         # except FileNotFoundError:
    #         #     print(f"Warning: File not found: {file_path}")
    #         #     continue
    #         except Exception as e:
    #             # print(f"Error loading file {file_path}: {e}")
    #             continue
                
    return image_data