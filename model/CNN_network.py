import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim 
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


class CNNet(nn.Module):
    def __init__(self, time_steps, batch_size, num_class, epochs):
        super(CNNet, self).__init__()
        self.test_in = []
        self.test_out = []
        self.batch_size = batch_size
        self.num_class = num_class
        self.epochs = epochs
        self.time_steps = time_steps
        self.feature_dim = 150
        
        # # Dynamically calculate the final feature size
        # time_after_conv1 = time_steps - 2  # kernel_size=3
        # time_after_pool1 = time_after_conv1 // 2  # max_pool size=2
        # time_after_conv2 = time_after_pool1 - 2  # kernel_size=3
        # time_after_pool2 = time_after_conv2 // 2  # max_pool size=2
        # time_after_conv3 = time_after_pool2 - 2  # kernel_size=3 
        # time_after_pool3 = time_after_conv3 // 2  # max_pool size=2
        
        # self.final_time_dim = max(1, time_after_pool3)  # Ensure at least 1
        
        # Define model layers - adapted for time series data
        # Input shape: [batch_size, 3, time_steps, feature_dim]
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), padding=(0, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(0, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(0, 1))
        
        # Calculate the flattened size for the fully connected layer
        # After 3 conv layers and 3 pooling layers
        # self.flattened_size = 128 * self.final_time_dim * self.feature_dim
        self.flattened_size = 128 * 4 * 5
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_class)
        
        # Prevent overfitting
        self.dropout = nn.Dropout(p=0.2)
        
        # Store test data
        self.test_in = None
        self.test_out = None

    def forward(self, x):
        # x shape: [batch_size, 1, time_steps, feature_dim]
        # Ensure input is properly shaped with channel dimension
        # if x.dim() == 3:
        #     # If input is [batch_size, time_steps, feature_dim]
        #     x = x.unsqueeze(1)
        
        # Apply convolutions
        x = F.max_pool2d(self.conv1(x), kernel_size=3) 
        x = F.relu(x)
        # x = self.dropout(x)
        
        x = F.max_pool2d(self.conv2(x), kernel_size=3) 
        x = F.relu(x)
        x = self.dropout(x)
        
        x = F.max_pool2d(self.conv3(x), kernel_size=3)
        x = F.relu(x)
        # x = self.dropout(x)
        
        # Flatten for fully connected layers
        # print(x.shape) [8, 128, 4, 5]
        x = x.view(-1, self.flattened_size)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x

    def prepare_data_loaders(self, x_train, x_test, y_train, y_test):
        """
        Prepare data loaders for training and testing.
        
        Args:
            x_train (torch.Tensor): Training input data
            x_test (torch.Tensor): Testing input data
            y_train (torch.Tensor): Training target data
            y_test (torch.Tensor): Testing target data
            
        Returns:
            tuple: Training and testing data loaders
        """
        print(f"{x_train.shape[0]} train samples")
        print(f"{x_test.shape[0]} test samples")

        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=True
        )
        
        # Store test data for evaluation
        self.test_in = x_test
        self.test_out = y_test
        
        return train_loader, test_loader

# Training function
def train_model(model, train_loader, learning_rate=1e-4, max_grad_norm=1.0):
    """
    Train the CNN model with gradient clipping and NaN detection.
    
    Args:
        model (CNNet): The CNN model to train
        train_loader (DataLoader): DataLoader with training data
        learning_rate (float, optional): Learning rate for optimizer
        max_grad_norm (float, optional): Maximum norm for gradient clipping
    """
    # Set model to training mode
    model.train()
    
    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-7)
    
    # Training loop
    for epoch in range(model.epochs):
        running_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            # Check for NaN in input
            if torch.isnan(x_batch).any():
                print(f"Warning: NaN detected in input batch {batch_idx}, replacing with zeros")
                x_batch = torch.nan_to_num(x_batch, nan=0.0)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(x_batch)
            
            # Calculate loss
            loss = criterion(outputs, y_batch)
            
            # Check if loss is NaN
            if torch.isnan(loss).any():
                print(f"Warning: NaN loss detected at epoch {epoch+1}, batch {batch_idx}")
                # Skip this batch
                continue
                
            # Backward pass
            loss.backward()
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Check for NaN gradients
            has_nan_grad = False
            for param in model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                print(f"Warning: NaN gradient detected at epoch {epoch+1}, batch {batch_idx}")
                optimizer.zero_grad()  # Clear the bad gradients
                continue
                
            # Update weights
            optimizer.step()
            
            # Calculate statistics
            predicted = torch.max(outputs.data, 1)[1]
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
            running_loss += loss.item()
            
        # Print epoch statistics
        if total > 0:  # Avoid division by zero
            if epoch + 1 < model.epochs:
                print(f'Epoch: {epoch+1}  Loss: {running_loss/len(train_loader):.6f}  Accuracy: {correct*100/total:.3f}%\r', end='')
            else:
                print(f'Epoch: {epoch+1}  Loss: {running_loss/len(train_loader):.6f}  Accuracy: {correct*100/total:.3f}%')
        else:
            if epoch + 1 < model.epochs:
                print(f'Epoch: {epoch+1}  Loss: {running_loss/len(train_loader) if len(train_loader) > 0 else 0:.6f}  Accuracy: 0.000%\r', end='')
            else:
                print(f'Epoch: {epoch+1}  Loss: {running_loss/len(train_loader) if len(train_loader) > 0 else 0:.6f}  Accuracy: 0.000%')

# Evaluation function
def evaluate_model(model, test_loader, usernames):
    """
    Evaluate the CNN model.
    
    Args:
        model (CNNet): The trained CNN model
        test_loader (DataLoader): DataLoader with test data
        usernames (list): List of usernames for display
    """
    # Set model to evaluation mode
    model.eval()
    
    correct = 0
    total = 0
    predict_label = []
    true_label = []
    predict_probs = []
    
    for test_inputs, test_labels in test_loader:
        # Forward pass
        outputs = model(test_inputs)
        # print("Output of CNN:", outputs)
        
        # Get predictions
        predicted = torch.max(outputs, 1)[1]
        
        # Store predictions and true labels
        predict_label.append(predicted)
        true_label.append(test_labels)
        
        # Calculate accuracy
        correct += (predicted == test_labels).sum().item()
        total += test_labels.size(0)

        # Prepare for a calibration curve
        # convert to probabilities
        predict_probs.append(F.softmax(outputs, dim=1))
    
    total_accuracy = 100 * correct / total if total > 0 else 0
    print(f'Correct: {correct}, Test accuracy: {total_accuracy:.3f}%')

    # Get all predictions and true labels
    true_l = model.test_out.to('cpu')
    with torch.no_grad():
        pred_l = torch.max(model(model.test_in), 1)[1].to('cpu')
    
    
    # Get unique labels
    unique_labels = sorted(torch.unique(true_l).tolist())

    
    # Generate target names for classification report
    target_names = [usernames[i] for i in unique_labels]
    
    # Display classification report
    print('Classification Report:')
    print(classification_report(true_l, pred_l, target_names=target_names))


    # Create confusion matrix
    array = confusion_matrix(true_l, pred_l)
    print('Confusion Matrix:')
    
    # Normalize confusion matrix for better visualization
    array_norm = np.around(array.astype('float') / np.sum(array, axis=1)[:, None], decimals=3)
    print(array)
    
    # Create dataframe for seaborn heatmap with predicted on x-axis (columns) and true on y-axis (index)
    df_cm_norm = pd.DataFrame(
        array_norm,
        index=[usernames[i] for i in unique_labels],   # true labels -> y-axis
        columns=[usernames[i] for i in unique_labels]  # predicted labels -> x-axis
    )
    
    # # Plot normalized confusion matrix (predicted on x, true on y)
    # plt.figure(figsize=(6, 4))
    # sn.heatmap(df_cm_norm, annot=True, cmap='Blues', fmt='.3f',
    #            xticklabels=df_cm_norm.columns, yticklabels=df_cm_norm.index)
    # plt.xlabel('Predicted Emotion')
    # plt.ylabel('True Emotion')
    # plt.tight_layout()


    # plt.title('Confusion Matrix')

    # Extract all probabilities and labels for calibration curve
    all_probs = torch.cat(predict_probs, dim=0).to('cpu')
    # Re-calculate by all output classes so sum is 1
    all_probs = all_probs / all_probs.sum(dim=1, keepdim=True)

    # print("All probs shape:", all_probs.shape)
    # print("All probs:", all_probs)
    all_labels = torch.cat(true_label, dim=0).to('cpu')
    # print("All labels shape:", all_labels.shape)
    # print("All labels:", all_labels)
    all_preds = torch.cat(predict_label, dim=0).to('cpu')
    true_class_probs = all_probs[torch.arange(len(all_labels)), all_labels]
    # print("True class probabilities:", true_class_probs)

    # Compute calibration bins
    num_bins = 5
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_edges[:-1]
    bin_uppers = bin_edges[1:]

    probs = true_class_probs.numpy()
    preds = all_preds.numpy()
    labels = all_labels.numpy()

    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.sum()
        if prop_in_bin > 0:
            # mean confidence of samples in this bin
            avg_conf = probs[in_bin].mean()
            # accuracy of samples in this bin
            acc = (preds[in_bin] == labels[in_bin]).mean()
            print("Bin ({:.2f}, {:.2f}]: Count = {}, Avg Conf = {:.3f}, Acc = {:.3f}".format(
                bin_lower, bin_upper, prop_in_bin, avg_conf, acc))
        else:
            print("No samples in bin ({:.2f}, {:.2f}]".format(bin_lower, bin_upper))
            avg_conf = np.nan
            acc = np.nan
        bin_confidences.append(avg_conf)
        bin_accuracies.append(acc)
        bin_counts.append(prop_in_bin)

    bin_confidences = np.array(bin_confidences)
    bin_accuracies = np.array(bin_accuracies)
    bin_counts = np.array(bin_counts)

    # Plot calibration curve
    # mask = ~np.isnan(bin_accuracies)
    # plt.figure(figsize=(6, 4))
    # # Perfectly calibrated line
    # plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    # # Calibration curve
    # plt.plot(bin_confidences[mask], bin_accuracies[mask], marker='o', label='CNN (5-class emotion)')

    # plt.xlabel('Predicted probability')
    # plt.ylabel('Empirical accuracy')
    # plt.title('Calibration Curve (Reliability Diagram)')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return correct, total, array, bin_confidences, bin_accuracies, bin_counts