#Original paper's implementation: kfold validation, downsampling trainng data to tackle class imbalance
import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt 
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from itertools import cycle


def create_model():
    # Define the CNN model according to the architecture described in the paper
    model = Sequential([
        # First Conv Block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(96, 96, 3), kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(3, 3)),
        Dropout(0.25),
        
        # Second Conv Block
        Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
        Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Third Conv Block
        Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Flattening and Fully Connected Layers
        Flatten(),
        Dense(1024, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        
        # Output Layer
        Dense(7, activation='softmax')  #We have 7 classes for classification
    ])

    # Compile the model
    opt= Adam(learning_rate= .001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # Print the model summary
    model.summary()

    return model 


def load_images_and_labels(data_dir, img_size):
    images = []
    labels = []
    class_names = os.listdir(data_dir)
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image = load_img(image_path, target_size=img_size)
            image = img_to_array(image) / 255.0  # Normalize pixel values
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)


def train_model(epochs, img_size, num_classes):
    train_data_dir = '../data/train_balanced_upsampled_1600' # modify if directory changes
    batch_size = 32
    k_folds = 10

    # Load data
    X, y = load_images_and_labels(train_data_dir, img_size)
    y = to_categorical(y, num_classes=num_classes)  # One-hot encoding

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=740)
    fold_no = 1
    accuracies = []

    # Callbacks
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Store metrics for all folds
    all_folds_train_loss = []
    all_folds_train_accuracy = []
    all_folds_val_loss = []
    all_folds_val_accuracy = []

    for train_index, val_index in kf.split(X):
        print(f"Training on fold {fold_no}...")

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Create a new model for each fold
        model = create_model()

        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1,
            callbacks=[lr_scheduler, early_stopping]
        )

        # Store metrics for each fold
        all_folds_train_loss.append(history.history['loss'])
        all_folds_train_accuracy.append(history.history['accuracy'])
        all_folds_val_loss.append(history.history['val_loss'])
        all_folds_val_accuracy.append(history.history['val_accuracy'])

        # Evaluate on validation data
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        print(f"Fold {fold_no} - Validation accuracy: {val_accuracy:.4f}")
        accuracies.append(val_accuracy)

        # Save the model for this fold
        model_path = f'../saved_models/kfold/saved_model_fold_{fold_no}.h5'
        model.save(model_path)
        print(f"Model for fold {fold_no} saved at {model_path}")

        fold_no += 1

    # Save metrics for each fold
    for fold_idx, (train_loss, train_acc, val_loss, val_acc) in enumerate(
        zip(all_folds_train_loss, all_folds_train_accuracy, all_folds_val_loss, all_folds_val_accuracy),
        start=1,
    ):
        np.save(f'../results/kfold/fold_{fold_idx}_train_loss.npy', train_loss)
        np.save(f'../results/kfold/fold_{fold_idx}_train_accuracy.npy', train_acc)
        np.save(f'../results/kfold/fold_{fold_idx}_val_loss.npy', val_loss)
        np.save(f'../results/kfold/fold_{fold_idx}_val_accuracy.npy', val_acc)

    # Calculate average performance across folds
    avg_val_loss = np.mean([np.mean(val_loss) for val_loss in all_folds_val_loss])
    avg_val_accuracy = np.mean(accuracies)

    print("\nFinal Cross-Validation Results:")
    print(f"Average Validation Loss: {avg_val_loss:.4f}")
    print(f"Average Validation Accuracy: {avg_val_accuracy:.4f}")

    return accuracies, all_folds_train_loss, all_folds_train_accuracy, all_folds_val_loss, all_folds_val_accuracy


# Parameters
epochs = 140
img_size = (96, 96)
num_classes = 7
k_folds = 10

# Perform k-fold cross-validation (Uncomment if you want to train the model)

# epochs= 140 
# img_size = (96, 96)
# num_classes = 7
# #Validation Accuracy
# accuracies= train_model(epochs, img_size, num_classes)
# average_accuracy = np.mean(accuracies)
# print(f"\nAverage validation accuracy across all folds: {average_accuracy:.4f}")


def load_and_plot_metrics(k_folds, epochs):
    # Lists to store all folds' metrics
    all_folds_train_loss = []
    all_folds_train_accuracy = []
    all_folds_val_loss = []
    all_folds_val_accuracy = []

    # Load metrics for each fold
    for fold_idx in range(1, k_folds + 1):
        train_loss = np.load(f'../results/kfold/fold_{fold_idx}_train_loss.npy', allow_pickle=True)
        train_accuracy = np.load(f'../results/kfold/fold_{fold_idx}_train_accuracy.npy', allow_pickle=True)
        val_loss = np.load(f'../results/kfold/fold_{fold_idx}_val_loss.npy', allow_pickle=True)
        val_accuracy = np.load(f'../results/kfold/fold_{fold_idx}_val_accuracy.npy', allow_pickle=True)
        
        all_folds_train_loss.append(train_loss)
        all_folds_train_accuracy.append(train_accuracy)
        all_folds_val_loss.append(val_loss)
        all_folds_val_accuracy.append(val_accuracy)

    # Pad metrics to align lengths (due to early stopping)
    max_epochs = max(len(loss) for loss in all_folds_train_loss)
    padded_train_loss = [np.pad(loss, (0, max_epochs - len(loss)), constant_values=np.nan) for loss in all_folds_train_loss]
    padded_train_accuracy = [np.pad(acc, (0, max_epochs - len(acc)), constant_values=np.nan) for acc in all_folds_train_accuracy]
    padded_val_loss = [np.pad(loss, (0, max_epochs - len(loss)), constant_values=np.nan) for loss in all_folds_val_loss]
    padded_val_accuracy = [np.pad(acc, (0, max_epochs - len(acc)), constant_values=np.nan) for acc in all_folds_val_accuracy]

    # Compute mean metrics across all folds, ignoring NaNs
    mean_train_loss = np.nanmean(padded_train_loss, axis=0)
    mean_train_accuracy = np.nanmean(padded_train_accuracy, axis=0)
    mean_val_loss = np.nanmean(padded_val_loss, axis=0)
    mean_val_accuracy = np.nanmean(padded_val_accuracy, axis=0)

    # Plot Mean Training and Validation Loss
    plt.figure(figsize=(4, 3))
    plt.plot(range(1, max_epochs + 1), mean_train_loss, label='Mean Training Loss', linewidth=2, color='red')
    plt.plot(range(1, max_epochs + 1), mean_val_loss, label='Mean Validation Loss', linewidth=2, color='blue')
    plt.xlabel('Epoch', fontsize=16, fontweight='bold')
    plt.ylabel('Loss', fontsize=16, fontweight='bold')
    plt.title('Loss vs. Epoch (Training and Validation)', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    #plt.grid(True)
    plt.tight_layout()
    #plt.show()

    # Plot Mean Training and Validation Accuracy
    plt.figure(figsize=(4, 3))
    plt.plot(range(1, max_epochs + 1), mean_train_accuracy, label='Mean Training Accuracy', linewidth=2, color='green')
    plt.plot(range(1, max_epochs + 1), mean_val_accuracy, label='Mean Validation Accuracy', linewidth=2, color='orange')
    plt.xlabel('Epoch', fontsize=16, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Accuracy vs. Epoch (Training and Validation)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=16)
    #plt.grid(True)
    plt.tight_layout()
    plt.show()


# Load metrics and plot
load_and_plot_metrics(k_folds, epochs)


def calculate_metrics(model, test_images, test_labels, class_names):
    y_pred = model.predict(test_images, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(test_labels, axis=1)
    
    # Classification report for individual class metrics
    report = classification_report(y_true_classes, y_pred_classes, target_names=class_names, output_dict=True)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true_classes, y_pred_classes, average='weighted')
    
    return precision, recall, f1_score, report

# Evaluate model on unseen test data
#test_data_dir = r'H:\skinLesion\data\test_independent' # modify if directory changes
test_data_dir = r'H:\ECE740_Project_software\data\test_balanced_upsampled_400'
class_names = sorted(os.listdir(test_data_dir))  # Assuming class directories represent class names
test_images, test_labels = load_images_and_labels(test_data_dir, img_size)
test_labels = to_categorical(test_labels, num_classes=num_classes)

test_accus = []
all_fold_metrics = []
individual_class_reports = []

for fold_no in range(1, 11):
    mname = 'saved_model_fold_' + str(fold_no) + '.h5'
    mpath = '../saved_models/kfold/' + mname
    model = load_model(mpath)
    loss, accu = model.evaluate(test_images, test_labels, verbose=0)
    print(f'Test accuracy fold {fold_no}: {accu}')
    test_accus.append(accu)
    
    # Calculate precision, recall, F1 score
    precision, recall, f1_score, report = calculate_metrics(model, test_images, test_labels, class_names)
    all_fold_metrics.append({'precision': precision, 'recall': recall, 'f1_score': f1_score})
    individual_class_reports.append(report)

# Compute average metrics across all folds
mean_test_accu_across_models= np.mean(test_accus)
average_metrics = {
    'precision': np.mean([metrics['precision'] for metrics in all_fold_metrics]),
    'recall': np.mean([metrics['recall'] for metrics in all_fold_metrics]),
    'f1_score': np.mean([metrics['f1_score'] for metrics in all_fold_metrics])
}

print (f'Average test accuracy across all 10 models: {mean_test_accu_across_models}')
print("\nAverage metrics across all folds:")
print(f"Precision: {average_metrics['precision']:.4f}")
print(f"Recall: {average_metrics['recall']:.4f}")
print(f"F1 Score: {average_metrics['f1_score']:.4f}")

# Print metrics for individual classes from one fold (e.g., last fold)
print("\nMetrics for individual classes (last fold):")
for class_name, metrics in individual_class_reports[-1].items():
    if isinstance(metrics, dict):  # Skip overall averages
        print(f"Class: {class_name}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1-score']:.4f}")
        print(f"  Support: {metrics['support']}\n")




def plot_auc_roc_curve_test(model, X_test, y_test, num_classes, class_names):
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Compute ROC curve and AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot all ROC curves
    plt.figure(figsize=(10, 6))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta'])
    
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlabel('False Positive Rate', fontsize=16, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=16, fontweight='bold')
    plt.title('ROC Curves for Test Data', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=18, prop={'weight': 'bold'})
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    #plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()


# Load the actual class names from the test directory
class_names = sorted(os.listdir(test_data_dir))

# Plot AUC ROC Curve for Test Data
plot_auc_roc_curve_test(model, test_images, test_labels, num_classes, class_names)



def plot_confusion_matrix(y_true, y_pred, class_names, normalize=False):
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format=".2f" if normalize else "d")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""), fontsize=16, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=14, fontweight='bold')
    plt.ylabel("True Label", fontsize=14, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()


# Evaluate model on unseen test data
#test_data_dir = r'H:\skinLesion\data\test_independent' # modify if directory changes
test_data_dir = r'H:\skinLesion\data\test_balanced_upsampled_400'
class_names = sorted(os.listdir(test_data_dir))  
test_images, test_labels = load_images_and_labels(test_data_dir, img_size)
test_labels_categorical = to_categorical(test_labels, num_classes=num_classes)

# Load the model for the last fold
fold_no = 10 
mname = f'saved_model_fold_{fold_no}.h5'
mpath = f'../saved_models/kfold/{mname}'
model = load_model(mpath)

# Evaluate the model
loss, accu = model.evaluate(test_images, test_labels_categorical, verbose=0)
print(f'Test accuracy for last fold (Fold {fold_no}): {accu}')

# Generate predictions for the test set
y_pred = np.argmax(model.predict(test_images, verbose=0), axis=1)
y_true = test_labels  # Non-categorical ground truth labels

# Plot the confusion matrix for the last fold
print(f"Confusion Matrix for Fold {fold_no}:")
plot_confusion_matrix(y_true, y_pred, class_names, normalize=True)

