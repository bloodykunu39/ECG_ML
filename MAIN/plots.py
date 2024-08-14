import matplotlib.pyplot as plt
import seaborn as sns


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt

# Set the style and color palette
sns.set_theme(context='notebook', style='whitegrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)









class accuracy_and_validation_plots:
    # Create subplots
    def __init__(self, epochs, train_loss, val_loss, train_accuracy, val_accuracy ):
        self.epochs = epochs
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.train_accuracy = train_accuracy
        self.val_accuracy = val_accuracy
    def plot_figure(self, savepath):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plotting Training and Validation Loss
        ax1.plot(self.epochs, self.train_loss, label='Training Loss', color='b', marker='o')
        ax1.plot(self.epochs, self.val_loss, label='Validation Loss', color='r', marker='o')
        ax1.set_title('Training and Validation Loss', fontsize=16)
        ax1.set_xlabel('Epochs', fontsize=14)
        ax1.set_ylabel('Loss', fontsize=14)
        ax1.legend(loc='upper right', fontsize=12)
        ax1.grid(True)

        # Plotting Training and Validation Accuracy
        ax2.plot(self.epochs, self.train_accuracy, label='Training Accuracy', color='b', marker='o')
        ax2.plot(self.epochs, self.val_accuracy, label='Validation Accuracy', color='r', marker='o')
        ax2.set_title('Training and Validation Accuracy', fontsize=16)
        ax2.set_xlabel('Epochs', fontsize=14)
        ax2.set_ylabel('Accuracy', fontsize=14)
        ax2.legend(loc='lower right', fontsize=12)
        ax2.grid(True)

        # Tight layout for better spacing
        plt.tight_layout()

        # Save the figure (optional)
        if savepath:
            plt.savefig(savepath, dpi=300)

        # Show the plot
        plt.show()





class ModelEvaluator:
    def __init__(self, model, X_train, y_train, X_test, y_test, class_names):
        """
        Initialize the ModelEvaluator.

        Parameters:
        - model: Trained model to be evaluated.
        - X_train: Training data features.
        - y_train: Training data labels.
        - X_test: Test data features.
        - y_test: Test data labels.
        - class_names: Optional list of class names for labeling. If None, uses default class indices.
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.class_names = class_names

    def predict(self):
        """
        Predict the labels and probabilities for the training and test datasets.
        """
        self.y_train_pred = self.model.predict(self.X_train)
        self.y_test_pred = self.model.predict(self.X_test)
        self.y_train_proba = self.model.predict_proba(self.X_train)
        self.y_test_proba = self.model.predict_proba(self.X_test)

    def compute_metrics(self):
        """
        Compute and return key evaluation metrics for both training and test datasets.
        """
        self.metrics = {
            'train': {
                'precision': precision_score(self.y_train, self.y_train_pred, average='weighted'),
                'recall': recall_score(self.y_train, self.y_train_pred, average='weighted'),
                'f1_score': f1_score(self.y_train, self.y_train_pred, average='weighted'),
                'accuracy': accuracy_score(self.y_train, self.y_train_pred)
            },
            'test': {
                'precision': precision_score(self.y_test, self.y_test_pred, average='weighted'),
                'recall': recall_score(self.y_test, self.y_test_pred, average='weighted'),
                'f1_score': f1_score(self.y_test, self.y_test_pred, average='weighted'),
                'accuracy': accuracy_score(self.y_test, self.y_test_pred)
            }
        }
        return self.metrics

    def print_classification_report(self):
        """
        Print a detailed classification report for the test dataset.
        """
        print("Classification Report for Test Data:")
        print(classification_report(self.y_test, self.y_test_pred, target_names=self.class_names))

    def plot_confusion_matrix(self, normalize=False, cmap='Blues', save_path=None):
        """
        Plot the confusion matrix for the test dataset.

        Parameters:
        - normalize: Whether to normalize the confusion matrix. Default is False.
        - cmap: Colormap for the confusion matrix plot. Default is 'Blues'.
        - save_path: Optional. File path to save the plot. If None, the plot is not saved.
        """
        cm = confusion_matrix(self.y_test, self.y_test_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
        else:
            title = 'Confusion Matrix'

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap, xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(title, fontsize=16)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        
        # Save the figure if a save path is provided
        if save_path:
            plt.savefig(save_path, dpi=300)
        
        plt.show()

    def plot_precision_recall_curves(self, save_path=None):
        """
        Plot the precision-recall curves for the test dataset.

        Parameters:
        - save_path: Optional. File path to save the plot. If None, the plot is not saved.
        """
        y_test_bin = label_binarize(self.y_test, classes=np.unique(self.y_test))
        plt.figure(figsize=(10, 8))
        sns.set(style="whitegrid")

        for i, class_name in enumerate(self.class_names):
            precision, recall, _ = precision_recall_curve(y_test_bin[:, i], self.y_test_proba[:, i])
            average_precision = average_precision_score(y_test_bin[:, i], self.y_test_proba[:, i])
            
            plt.plot(recall, precision, label=f'{class_name} (AP = {average_precision:.2f})')

        plt.title('Precision-Recall Curves', fontsize=16)
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.legend(loc='lower left', fontsize=12)
        plt.grid(True)

        # Save the figure if a save path is provided
        if save_path:
            plt.savefig(save_path, dpi=300)
        
        plt.show()

    def plot_roc_curves(self, save_path=None):
        """
        Plot the ROC curves for the test dataset.

        Parameters:
        - save_path: Optional. File path to save the plot. If None, the plot is not saved.
        """
        y_test_bin = label_binarize(self.y_test, classes=np.unique(self.y_test))
        plt.figure(figsize=(10, 8))
        sns.set(style="whitegrid")

        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], self.y_test_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.title('ROC Curves', fontsize=16)
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.legend(loc='lower right', fontsize=12)
        plt.grid(True)

        # Save the figure if a save path is provided
        if save_path:
            plt.savefig(save_path, dpi=300)
        
        plt.show()

    def evaluate(self):
        """
        Full evaluation: Predict labels, compute metrics, print classification report, plot confusion matrix,
        plot precision-recall curves, and plot ROC curves.
        """
        self.predict()
        metrics = self.compute_metrics()
        self.print_classification_report()
        self.plot_confusion_matrix()
        self.plot_precision_recall_curves()
        self.plot_roc_curves()
        return metrics

