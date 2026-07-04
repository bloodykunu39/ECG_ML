import matplotlib.pyplot as plt
import seaborn as sns
import torch

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





class Model_Evaluator:
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




import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelEvaluator:
    """
    A class to evaluate a PyTorch model and generate various evaluation plots and reports.
    """
    def __init__(self, model: torch.nn.Module, test_dataloader: torch.utils.data.DataLoader, label_names: list[str]):
        """
        Initialize the ModelEvaluator with a model, test dataloader, and label names.

        Args:
            model (torch.nn.Module): The PyTorch model to evaluate.
            test_dataloader (torch.utils.data.DataLoader): The dataloader for the test dataset.
            label_names (list[str]): The list of label names for the classes.
        """
        self.model = model
        self.test_dataloader = test_dataloader
        self.label_names = label_names
        self.model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for images, labels in self.test_dataloader:
                outputs = self.model(images.to(device))
                _, predicted = torch.max(outputs, 1)
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(labels.cpu().numpy())
        self.y_true = y_true
        self.y_pred = y_pred
        del self.model
        
    def confusion_matrix_plot(self) -> None:
        """
        Plot the confusion matrix.
        """
        cm = confusion_matrix(self.y_true, self.y_pred)
        cm = cm / cm.sum(axis=1)[:, np.newaxis]  # Normalize confusion matrix
        df_cm = pd.DataFrame(cm, index=self.label_names, columns=self.label_names)
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True, cmap='Blues')
        plt.xlabel('Prediction')
        plt.ylabel('Truth')
        plt.title('Confusion Matrix')
        plt.show()

    def classification_report_print(self) -> None:
        """
        Print the classification report.
        """
        print(classification_report(self.y_true, self.y_pred, target_names=self.label_names))

    def precision_recall_curve_plot(self, num_classes: int) -> None:
        """
        Plot the precision-recall curve for each class.

        Args:
            num_classes (int): The number of classes.
        """
        y_true = label_binarize(self.y_true, classes=range(num_classes))
        y_pred = label_binarize(self.y_pred, classes=range(num_classes))

        plt.figure(figsize=(10, 8))
        for i in range(num_classes):
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
            ap = average_precision_score(y_true[:, i], y_pred[:, i])
            plt.plot(recall, precision, label=f'Class {i} (AP = {ap:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.show()

    def roc_curve_plot(self, num_classes: int) -> None:
        """
        Plot the ROC curve for each class.

        Args:
            num_classes (int): The number of classes.
        """
        y_true = label_binarize(self.y_true, classes=range(num_classes))
        y_pred = label_binarize(self.y_pred, classes=range(num_classes))

        plt.figure(figsize=(10, 8))
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_precision_recall_curves(y_true, y_scores, class_names=["ST", "SB", "SR"], 
                                  colors=['blue', 'red', 'green'],
                                  figsize_micro=(8, 6), figsize_multi=(10, 8)):
    """
    Plot Precision-Recall curves for multi-class classification.
    
    Args:
        y_true: Array of true labels
        y_scores: Array of probability scores (shape: n_samples x n_classes)
        class_names: List of class names
        colors: List of colors for each class
        figsize_micro: Figure size for micro-averaged PR curve
        figsize_multi: Figure size for multi-class PR curves
    
    Returns:
        dict: Dictionary containing precision, recall, and average_precision for each class
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize
    import matplotlib.pyplot as plt
    import numpy as np
    
    n_classes = len(class_names)
    y_true = np.array(y_true)
    
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    # For each class, compute PR curve using probability scores
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            (y_true == i).astype(int),  # Binary true labels for class i
            y_scores[:, i]  # Probability scores for class i
        )
        average_precision[i] = average_precision_score(
            (y_true == i).astype(int), 
            y_scores[:, i]
        )
    
    # Micro-average: using probability scores
    y_true_binary = label_binarize(y_true, classes=list(range(n_classes)))
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true_binary.ravel(),
        y_scores.ravel()
    )
    average_precision["micro"] = average_precision_score(
        y_true_binary, 
        y_scores,
        average="micro"
    )
    
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))
    
    # Plot micro-average PR curve
    plt.figure(figsize=figsize_micro)
    plt.step(recall['micro'], precision['micro'], where='post', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
        .format(average_precision["micro"]))
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Plot individual class PR curves
    plt.figure(figsize=figsize_multi)
    for i, color, class_name in zip(range(n_classes), colors, class_names):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label=f'{class_name} (AP = {average_precision[i]:0.2f})')
    
    plt.plot(recall['micro'], precision['micro'], color='gold', lw=2, linestyle='--',
             label=f'Micro-average (AP = {average_precision["micro"]:0.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curves for Multi-class Classification')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return {"precision": precision, "recall": recall, "average_precision": average_precision}


def evaluate_model_with_cm(model, dataloader, device, class_names=["ST", "SB", "SR"], 
                           figsize=(10, 7), cmap='Blues', return_metrics=False):
    """
    Evaluate model and plot confusion matrix.
    
    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        class_names: List of class names for confusion matrix labels
        figsize: Figure size for the plot
        cmap: Colormap for heatmap
        return_metrics: If True, returns y_true, y_pred, y_scores
    
    Returns:
        If return_metrics=True: tuple of (y_true, y_pred, y_scores, cm)
        Otherwise: None (just plots)
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import pandas as pd
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt
    
    model.eval()
    y_pred = []
    y_true = []
    y_scores = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images.to(device))
            probabilities = F.softmax(outputs, dim=1)
            y_scores.extend(probabilities.cpu().numpy())
            
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    y_scores = np.array(y_scores)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Create and normalize confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    df_cm = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)
    plt.figure(figsize=figsize)
    sns.heatmap(df_cm, annot=True, cmap=cmap)
    plt.xlabel('Prediction')
    plt.ylabel('Truth')
    plt.title('Confusion Matrix')
    plt.show()
    
    if return_metrics:
        return y_true, y_pred, y_scores, cm
    
def plot_micro_averaged_pr_curve(y_true, y_scores, class_names=["ST", "SB", "SR"], 
                                  figsize=(8, 6)):
    """
    Plot micro-averaged Precision-Recall curve.
    
    Args:
        y_true: Array of true labels
        y_scores: Array of probability scores (shape: n_samples x n_classes)
        class_names: List of class names
        figsize: Figure size for the plot
    
    Returns:
        dict: Dictionary containing micro-averaged precision, recall, and average_precision
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize
    import matplotlib.pyplot as plt
    import numpy as np
    
    n_classes = len(class_names)
    y_true = np.array(y_true)
    
    # Micro-average: using probability scores
    y_true_binary = label_binarize(y_true, classes=list(range(n_classes)))
    precision_micro, recall_micro, _ = precision_recall_curve(
        y_true_binary.ravel(),
        y_scores.ravel()
    )
    average_precision_micro = average_precision_score(
        y_true_binary, 
        y_scores,
        average="micro"
    )
    
    print('Average precision score, micro-averaged over all classes: {0:0.4f}'
          .format(average_precision_micro))
    
    # Plot micro-average PR curve
    plt.figure(figsize=figsize)
    plt.step(recall_micro, precision_micro, where='post', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.4f}'
        .format(average_precision_micro))
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return {
        "precision": precision_micro, 
        "recall": recall_micro, 
        "average_precision": average_precision_micro
    }


def plot_per_class_pr_curves(y_true, y_scores, class_names=["ST", "SB", "SR"], 
                              colors=['blue', 'red', 'green'], figsize=(10, 8)):
    """
    Plot individual class Precision-Recall curves with micro-average.
    
    Args:
        y_true: Array of true labels
        y_scores: Array of probability scores (shape: n_samples x n_classes)
        class_names: List of class names
        colors: List of colors for each class
        figsize: Figure size for the plot
    
    Returns:
        dict: Dictionary containing precision, recall, and average_precision for each class and micro-average
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize
    import matplotlib.pyplot as plt
    import numpy as np
    
    n_classes = len(class_names)
    y_true = np.array(y_true)
    
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    # For each class, compute PR curve using probability scores
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            (y_true == i).astype(int),  # Binary true labels for class i
            y_scores[:, i]  # Probability scores for class i
        )
        average_precision[i] = average_precision_score(
            (y_true == i).astype(int), 
            y_scores[:, i]
        )
    
    # Micro-average: using probability scores
    y_true_binary = label_binarize(y_true, classes=list(range(n_classes)))
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true_binary.ravel(),
        y_scores.ravel()
    )
    average_precision["micro"] = average_precision_score(
        y_true_binary, 
        y_scores,
        average="micro"
    )
    
    # Plot individual class PR curves
    plt.figure(figsize=figsize)
    for i, color, class_name in zip(range(n_classes), colors, class_names):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label=f'{class_name} (AP = {average_precision[i]:0.4f})')
    
    plt.plot(recall['micro'], precision['micro'], color='gold', lw=2, linestyle='--',
             label=f'Micro-average (AP = {average_precision["micro"]:0.4f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curves for Multi-class Classification')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return {"precision": precision, "recall": recall, "average_precision": average_precision}


def plot_roc_curves(y_true, y_scores, class_names=["ST", "SB", "SR"], 
                    colors=['aqua', 'darkorange', 'cornflowerblue'], 
                    figsize=(10, 8)):
    """
    Plot ROC curves for multi-class classification with micro and macro averages.
    
    Args:
        y_true: Array of true labels
        y_scores: Array of probability scores (shape: n_samples x n_classes)
        class_names: List of class names
        colors: List of colors for each class
        figsize: Figure size for the plot
    
    Returns:
        dict: Dictionary containing fpr, tpr, and roc_auc for each class and averages
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    import matplotlib.pyplot as plt
    import numpy as np
    
    n_classes = len(class_names)
    
    # Binarize the true labels for multi-class ROC
    y_test = label_binarize(y_true, classes=list(range(n_classes)))
    
    # Use probability scores (not predictions!)
    y_score = y_scores  # This contains the probability scores from softmax
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Compute ROC curve and ROC area for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.figure(figsize=figsize)
    
    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"],
             label='Micro-average ROC curve (AUC = {0:0.4f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    # Plot macro-average ROC curve
    plt.plot(fpr["macro"], tpr["macro"],
             label='Macro-average ROC curve (AUC = {0:0.4f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    # Plot ROC curve for each class
    for i, color, class_name in zip(range(n_classes), colors, class_names):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {class_name} (AUC = {roc_auc[i]:0.4f})')
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves for Multi-class Classification')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Print AUC scores
    print("AUC Scores:")
    for i, class_name in enumerate(class_names):
        print(f"Class {class_name}: {roc_auc[i]:.4f}")
    print(f"Micro-average: {roc_auc['micro']:.4f}")
    print(f"Macro-average: {roc_auc['macro']:.4f}")
    
    return {"fpr": fpr, "tpr": tpr, "roc_auc": roc_auc}


def print_classification_report(y_true, y_pred, class_names=["ST", "SB", "SR"], digits=4):
    """
    Print classification report with precision, recall, f1-score for each class.
    
    Args:
        y_true: Array of true labels
        y_pred: Array of predicted labels
        class_names: List of class names
        digits: Number of decimal digits to display (default: 4)
    
    Returns:
        str: Classification report as string
    """
    from sklearn.metrics import classification_report
    import numpy as np
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    report = classification_report(y_true, y_pred, target_names=class_names, digits=digits)
    print(report)
    
    return report


def plot_micro_averaged_pr_curve(y_true, y_scores, class_names=["ST", "SB", "SR"], 
                                  figsize=(8, 6)):
    """
    Plot micro-averaged Precision-Recall curve.
    
    Args:
        y_true: Array of true labels
        y_scores: Array of probability scores (shape: n_samples x n_classes)
        class_names: List of class names
        figsize: Figure size for the plot
    
    Returns:
        dict: Dictionary containing micro-averaged precision, recall, and average_precision
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize
    import matplotlib.pyplot as plt
    import numpy as np
    
    n_classes = len(class_names)
    y_true = np.array(y_true)
    
    # Micro-average: using probability scores
    y_true_binary = label_binarize(y_true, classes=list(range(n_classes)))
    precision_micro, recall_micro, _ = precision_recall_curve(
        y_true_binary.ravel(),
        y_scores.ravel()
    )
    average_precision_micro = average_precision_score(
        y_true_binary, 
        y_scores,
        average="micro"
    )
    
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision_micro))
    
    # Plot micro-average PR curve
    plt.figure(figsize=figsize)
    plt.step(recall_micro, precision_micro, where='post', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
        .format(average_precision_micro))
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return {
        "precision": precision_micro, 
        "recall": recall_micro, 
        "average_precision": average_precision_micro
    }


def plot_per_class_pr_curves(y_true, y_scores, class_names=["ST", "SB", "SR"], 
                              colors=['blue', 'red', 'green'], figsize=(10, 8)):
    """
    Plot individual class Precision-Recall curves with micro-average.
    
    Args:
        y_true: Array of true labels
        y_scores: Array of probability scores (shape: n_samples x n_classes)
        class_names: List of class names
        colors: List of colors for each class
        figsize: Figure size for the plot
    
    Returns:
        dict: Dictionary containing precision, recall, and average_precision for each class and micro-average
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize
    import matplotlib.pyplot as plt
    import numpy as np
    
    n_classes = len(class_names)
    y_true = np.array(y_true)
    
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    # For each class, compute PR curve using probability scores
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            (y_true == i).astype(int),  # Binary true labels for class i
            y_scores[:, i]  # Probability scores for class i
        )
        average_precision[i] = average_precision_score(
            (y_true == i).astype(int), 
            y_scores[:, i]
        )
    
    # Micro-average: using probability scores
    y_true_binary = label_binarize(y_true, classes=list(range(n_classes)))
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true_binary.ravel(),
        y_scores.ravel()
    )
    average_precision["micro"] = average_precision_score(
        y_true_binary, 
        y_scores,
        average="micro"
    )
    
    # Plot individual class PR curves
    plt.figure(figsize=figsize)
    for i, color, class_name in zip(range(n_classes), colors, class_names):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label=f'{class_name} (AP = {average_precision[i]:0.2f})')
    
    plt.plot(recall['micro'], precision['micro'], color='gold', lw=2, linestyle='--',
             label=f'Micro-average (AP = {average_precision["micro"]:0.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curves for Multi-class Classification')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return {"precision": precision, "recall": recall, "average_precision": average_precision}



def evaluate_all(model, dataloader, device, class_names=["ST", "SB", "SR"],
                 colors_pr=['blue', 'red', 'green'],
                 colors_roc=['aqua', 'darkorange', 'cornflowerblue'],
                 plot_confusion_matrix=True,
                 plot_pr_micro=True,
                 plot_pr_per_class=True,
                 plot_roc=True,
                 print_report=True):
    """
    Comprehensive model evaluation with all plots and metrics.
    
    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        class_names: List of class names
        colors_pr: List of colors for PR curves
        colors_roc: List of colors for ROC curves
        plot_confusion_matrix: Whether to plot confusion matrix
        plot_pr_micro: Whether to plot micro-averaged PR curve
        plot_pr_per_class: Whether to plot per-class PR curves
        plot_roc: Whether to plot ROC curves
        print_report: Whether to print classification report
    
    Returns:
        dict: Dictionary containing all metrics (y_true, y_pred, y_scores, cm, pr_metrics, roc_metrics, report)
    """
    results = {}
    
    # Step 1: Evaluate model and get confusion matrix
    if plot_confusion_matrix:
        print("=" * 80)
        print("CONFUSION MATRIX")
        print("=" * 80)
        y_true, y_pred, y_scores, cm = evaluate_model_with_cm(
            model, dataloader, device, class_names=class_names, return_metrics=True
        )
        results['y_true'] = y_true
        results['y_pred'] = y_pred
        results['y_scores'] = y_scores
        results['confusion_matrix'] = cm
        print()
    else:
        # Still need to get predictions for other plots
        import torch
        import torch.nn.functional as F
        import numpy as np
        
        model.eval()
        y_pred = []
        y_true = []
        y_scores = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                outputs = model(images.to(device))
                probabilities = F.softmax(outputs, dim=1)
                y_scores.extend(probabilities.cpu().numpy())
                
                _, predicted = torch.max(outputs, 1)
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(labels.cpu().numpy())
        
        y_scores = np.array(y_scores)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        results['y_true'] = y_true
        results['y_pred'] = y_pred
        results['y_scores'] = y_scores
    
    # Step 2: Plot micro-averaged PR curve
    if plot_pr_micro:
        print("=" * 80)
        print("MICRO-AVERAGED PRECISION-RECALL CURVE")
        print("=" * 80)
        pr_micro = plot_micro_averaged_pr_curve(
            results['y_true'], results['y_scores'], class_names=class_names
        )
        results['pr_micro'] = pr_micro
        print()
    
    # Step 3: Plot per-class PR curves
    if plot_pr_per_class:
        print("=" * 80)
        print("PER-CLASS PRECISION-RECALL CURVES")
        print("=" * 80)
        pr_per_class = plot_per_class_pr_curves(
            results['y_true'], results['y_scores'], 
            class_names=class_names, colors=colors_pr
        )
        results['pr_per_class'] = pr_per_class
        print()
    
    # Step 4: Plot ROC curves
    if plot_roc:
        print("=" * 80)
        print("ROC CURVES")
        print("=" * 80)
        roc_metrics = plot_roc_curves(
            results['y_true'], results['y_scores'],
            class_names=class_names, colors=colors_roc
        )
        results['roc_metrics'] = roc_metrics
        print()
    
    # Step 5: Print classification report
    if print_report:
        print("=" * 80)
        print("CLASSIFICATION REPORT")
        print("=" * 80)
        report = print_classification_report(
            results['y_true'], results['y_pred'], class_names=class_names
        )
        results['classification_report'] = report
        print()
    
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    
    return results


# ---------------------------------------------------------------------------
# Multi-seed utilities  (added for 5-seed reproducibility)
# ---------------------------------------------------------------------------

def plot_learning_curves_multiseed(
    all_histories: list,
    metric: str = "val_acc",
    title: str = "Learning Curve (5 seeds)",
    save_path: str = None,
    figsize: tuple = (10, 5),
) -> None:
    """
    Plot mean ± std learning curves across multiple seeds.
    Handles variable-length histories (different early-stopping per seed)
    by NaN-padding shorter runs before computing mean/std.

    Parameters
    ----------
    all_histories : list of dict
        One dict per seed, each with keys:
        'train_loss', 'val_loss', 'train_acc', 'val_acc'
        (lists of per-epoch values — may differ in length).
    metric : str
        Which key to plot. One of 'train_loss', 'val_loss',
        'train_acc', 'val_acc'.
    title : str
        Plot title.
    save_path : str or None
        If given, saves the figure to this path.
    figsize : tuple
        Matplotlib figure size.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    raw = [h[metric] for h in all_histories]

    # Pad shorter runs with NaN so all rows have the same length
    max_len = max(len(r) for r in raw)
    padded  = np.full((len(raw), max_len), np.nan)
    for i, r in enumerate(raw):
        padded[i, :len(r)] = r

    # NaN-aware mean and std (ignores missing epochs)
    mean   = np.nanmean(padded, axis=0)
    std    = np.nanstd(padded,  axis=0)
    epochs = np.arange(1, max_len + 1)

    fig, ax = plt.subplots(figsize=figsize)

    # Individual seed traces (light) — plot only the non-NaN portion
    seed_labels = [40 + i for i in range(len(raw))]
    for i, (row, lbl) in enumerate(zip(padded, seed_labels)):
        valid = ~np.isnan(row)
        ax.plot(epochs[valid], row[valid], alpha=0.25, linewidth=1,
                label=f"seeds (individual)" if i == 0 else "_nolegend_",
                color="steelblue")

    # Mean ± std band (NaN propagation hides missing epochs cleanly)
    ax.plot(epochs, mean, linewidth=2.5, label="Mean", color="steelblue")
    ax.fill_between(
        epochs,
        np.where(np.isnan(mean - std), np.nan, mean - std),
        np.where(np.isnan(mean + std), np.nan, mean + std),
        alpha=0.25, color="steelblue", label="±1 std"
    )

    ylabel = "Accuracy (%)" if "acc" in metric else "Loss"
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


def summarize_seed_results(
    all_metrics: list,
    class_names: list = None,
    save_csv: str = None,
) -> dict:
    """
    Compute and print mean ± std of key metrics across seeds.

    Parameters
    ----------
    all_metrics : list of dict
        One dict per seed — the return value of evaluate_all().
        Each dict must contain 'y_true', 'y_pred', 'y_scores'.
    class_names : list of str or None
        Class labels (for per-class F1). Defaults to ['ST','SB','SR'].
    save_csv : str or None
        If given, saves a summary CSV to this path.

    Returns
    -------
    dict  with keys 'mean' and 'std', each mapping metric → value.
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score, average_precision_score,
        classification_report
    )
    from sklearn.preprocessing import label_binarize

    if class_names is None:
        class_names = ["ST", "SB", "SR"]

    n_classes = len(class_names)
    rows = []
    
    # Store classification_report dictionaries and AUCs for all seeds
    all_reports = []
    all_aucs = {c: [] for c in class_names}

    for s, res in enumerate(all_metrics):
        y_true   = np.array(res["y_true"])
        y_pred   = np.array(res["y_pred"])
        y_scores = np.array(res["y_scores"])

        acc      = accuracy_score(y_true, y_pred) * 100
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_micro = f1_score(y_true, y_pred, average="micro")

        y_bin    = label_binarize(y_true, classes=list(range(n_classes)))
        auc_mac  = roc_auc_score(y_bin, y_scores, average="macro", multi_class="ovr")
        ap_mic   = average_precision_score(y_bin, y_scores, average="micro")
        
        # Per-class AUCs
        for i, c_name in enumerate(class_names):
            all_aucs[c_name].append(roc_auc_score(y_bin[:, i], y_scores[:, i]))
        
        # Collect full classification report as dict
        rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        all_reports.append(rep)

        row = {
            "seed":       40 + s,
            "accuracy":   acc,
            "f1_macro":   f1_macro,
            "f1_micro":   f1_micro,
            "auc_macro":  auc_mac,
            "ap_micro":   ap_mic,
        }
        rows.append(row)

    df = pd.DataFrame(rows).set_index("seed")

    mean = df.mean()
    std  = df.std()

    # Pretty print Summary
    print("\n" + "=" * 100)
    print("  MULTI-SEED SUMMARY  (5 seeds: 40–44)")
    print("=" * 100)
    header = f"{'Metric':<20} {'Mean':>10} {'Std':>10}"
    print(header)
    print("-" * 42)
    fmt_map = {
        "accuracy":  ("{:.2f}%", "{:.2f}%"),
        "f1_macro":  ("{:.4f}",  "{:.4f}"),
        "f1_micro":  ("{:.4f}",  "{:.4f}"),
        "auc_macro": ("{:.4f}",  "{:.4f}"),
        "ap_micro":  ("{:.4f}",  "{:.4f}"),
    }
    for metric, (mfmt, sfmt) in fmt_map.items():
        print(f"  {metric:<18} {mfmt.format(mean[metric]):>10} "
              f"± {sfmt.format(std[metric]):<10}")
    print("-" * 100)
    
    # ── Print Full Averaged Classification Report ──
    print("  PER-CLASS CLASSIFICATION REPORT (Mean ± Std)")
    print("-" * 100)
    print(f"  {'Class':<15} {'Precision':<20} {'Recall':<20} {'F1-Score':<20} {'AUC':<20} {'Support'}")
    print("-" * 100)
    
    # Per-class rows
    for c_name in class_names:
        precs = [rep[c_name]['precision'] for rep in all_reports]
        recs  = [rep[c_name]['recall'] for rep in all_reports]
        f1s   = [rep[c_name]['f1-score'] for rep in all_reports]
        aucs  = all_aucs[c_name]
        support = all_reports[0][c_name]['support'] # Support is same across seeds
        
        p_str = f"{np.mean(precs):.4f} ± {np.std(precs):.4f}"
        r_str = f"{np.mean(recs):.4f} ± {np.std(recs):.4f}"
        f_str = f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}"
        a_str = f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f}"
        print(f"  {c_name:<15} {p_str:<20} {r_str:<20} {f_str:<20} {a_str:<20} {support}")
        
    print()
    
    # Averages rows
    # Accuracy is a single number in the dict, not precision/recall/f1
    accs = [rep['accuracy'] for rep in all_reports]
    total_support = all_reports[0]['macro avg']['support']
    print(f"  {'accuracy':<77} {np.mean(accs):.4f} ± {np.std(accs):.4f}  {total_support}")
    
    for avg_type in ['macro avg', 'weighted avg']:
        precs = [rep[avg_type]['precision'] for rep in all_reports]
        recs  = [rep[avg_type]['recall'] for rep in all_reports]
        f1s   = [rep[avg_type]['f1-score'] for rep in all_reports]
        
        # Calculate macro AUC manually across seeds
        if avg_type == 'macro avg':
            aucs = [np.mean([all_aucs[c][i] for c in class_names]) for i in range(len(all_reports))]
        else:
            # Weighted AUC
            aucs = [np.average([all_aucs[c][i] for c in class_names], weights=[all_reports[0][c]['support'] for c in class_names]) for i in range(len(all_reports))]
            
        p_str = f"{np.mean(precs):.4f} ± {np.std(precs):.4f}"
        r_str = f"{np.mean(recs):.4f} ± {np.std(recs):.4f}"
        f_str = f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}"
        a_str = f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f}"
        print(f"  {avg_type:<15} {p_str:<20} {r_str:<20} {f_str:<20} {a_str:<20} {total_support}")
        
    print("=" * 100 + "\n")

    if save_csv:
        df.loc["mean"] = mean
        df.loc["std"]  = std
        df.to_csv(save_csv)
        print(f"Summary CSV saved: {save_csv}")

    return {"mean": mean.to_dict(), "std": std.to_dict(), "per_seed": df}


def evaluate_multiseed_plots(
    all_metrics: list,
    class_names: list = None,
    colors_pr: list = ['blue', 'red', 'green'],
    colors_roc: list = ['aqua', 'darkorange', 'cornflowerblue'],
    save_dir: str = None,
    namkaran: str = "multiseed"
):
    """
    Generate averaged plots across multiple seeds:
    1. Averaged Confusion Matrix with standard deviation.
    2. Averaged ROC curves with shaded standard deviation region.
    3. Averaged PR curves with shaded standard deviation region.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (
        confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
    )
    from sklearn.preprocessing import label_binarize
    import os

    if class_names is None:
        class_names = ["ST", "SB", "SR"]
    
    n_classes = len(class_names)
    n_seeds = len(all_metrics)
    
    # ---------------------------------------------------------
    # 1. Averaged Confusion Matrix (Normalized to Proportions)
    # ---------------------------------------------------------
    all_cms = []
    for res in all_metrics:
        if 'confusion_matrix' in res:
            cm = np.array(res['confusion_matrix'])
        else:
            cm = confusion_matrix(res['y_true'], res['y_pred'])
        # Normalize to proportions (0.0 to 1.0) based on True Labels (rows)
        cm_prop = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        all_cms.append(cm_prop)
            
    all_cms = np.array(all_cms)
    mean_cm = np.mean(all_cms, axis=0)
    std_cm = np.std(all_cms, axis=0)
    
    plt.figure(figsize=(8, 6))
    
    annot = np.empty_like(mean_cm, dtype=object)
    for i in range(n_classes):
        for j in range(n_classes):
            annot[i, j] = f"{mean_cm[i, j]:.3f}\n± {std_cm[i, j]:.3f}"
            
    sns.heatmap(mean_cm, annot=annot, fmt="", cmap="Blues", cbar=True,
                xticklabels=class_names, yticklabels=class_names, 
                vmin=0, vmax=1.0) # Lock color scale to 0.0 - 1.0
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Averaged Confusion Matrix (5 Seeds, Proportions)')
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{namkaran}_multiseed_cm.png"), dpi=150, bbox_inches='tight')
    plt.show()

    # ---------------------------------------------------------
    # 2. Averaged ROC Curves
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    
    mean_fpr = np.linspace(0, 1, 100)
    
    # A. Per-class ROC
    per_class_auc_strings = []
    for i, color in zip(range(n_classes), colors_roc):
        tprs = []
        aucs = []
        for res in all_metrics:
            y_bin = label_binarize(res['y_true'], classes=list(range(n_classes)))
            fpr, tpr, _ = roc_curve(y_bin[:, i], res['y_scores'][:, i])
            
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc(fpr, tpr))
            
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs, axis=0)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        
        per_class_auc_strings.append(f"Class {class_names[i]}: {mean_auc:.4f} ± {std_auc:.4f}")
        
        plt.plot(mean_fpr, mean_tpr, color=color,
                 label=f'{class_names[i]} (AUC = {mean_auc:.4f} ± {std_auc:.4f})',
                 lw=2, alpha=0.8)
        
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=0.2)

    # B. Micro-average ROC
    tprs_micro = []
    aucs_micro = []
    for res in all_metrics:
        y_bin = label_binarize(res['y_true'], classes=list(range(n_classes)))
        fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), res['y_scores'].ravel())
        interp_tpr = np.interp(mean_fpr, fpr_micro, tpr_micro)
        interp_tpr[0] = 0.0
        tprs_micro.append(interp_tpr)
        aucs_micro.append(auc(fpr_micro, tpr_micro))
        
    mean_tpr_micro = np.mean(tprs_micro, axis=0)
    mean_tpr_micro[-1] = 1.0
    mean_auc_micro = np.mean(aucs_micro)
    std_auc_micro = np.std(aucs_micro)
    
    plt.plot(mean_fpr, mean_tpr_micro, color='deeppink', linestyle=':',
             label=f'Micro-average (AUC = {mean_auc_micro:.4f} ± {std_auc_micro:.4f})',
             lw=2, alpha=0.8)

    # C. Macro-average ROC
    tprs_macro = []
    aucs_macro = []
    for res in all_metrics:
        y_bin = label_binarize(res['y_true'], classes=list(range(n_classes)))
        seed_macro_tpr = np.zeros_like(mean_fpr)
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], res['y_scores'][:, i])
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            seed_macro_tpr += interp_tpr
        seed_macro_tpr /= n_classes
        tprs_macro.append(seed_macro_tpr)
        aucs_macro.append(auc(mean_fpr, seed_macro_tpr))
        
    mean_tpr_macro = np.mean(tprs_macro, axis=0)
    mean_tpr_macro[-1] = 1.0
    mean_auc_macro = np.mean(aucs_macro)
    std_auc_macro = np.std(aucs_macro)
    
    plt.plot(mean_fpr, mean_tpr_macro, color='navy', linestyle=':',
             label=f'Macro-average (AUC = {mean_auc_macro:.4f} ± {std_auc_macro:.4f})',
             lw=2, alpha=0.8)

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Averaged ROC Curves (5 Seeds)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{namkaran}_multiseed_roc.png"), dpi=150, bbox_inches='tight')
    plt.show()

    # Print AUC scores
    print("AUC Scores (Mean ± Std):")
    print("-" * 30)
    for auc_str in per_class_auc_strings:
        print(auc_str)
    print(f"Micro-average: {mean_auc_micro:.4f} ± {std_auc_micro:.4f}")
    print(f"Macro-average: {mean_auc_macro:.4f} ± {std_auc_macro:.4f}")
    print("-" * 30 + "\n")

    # ---------------------------------------------------------
    # 3. Averaged PR Curves
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    
    mean_recall = np.linspace(0, 1, 100)
    
    # A. Per-class PR
    for i, color in zip(range(n_classes), colors_pr):
        precisions = []
        aps = []
        for res in all_metrics:
            y_bin = label_binarize(res['y_true'], classes=list(range(n_classes)))
            precision, recall, _ = precision_recall_curve(y_bin[:, i], res['y_scores'][:, i])
            
            interp_prec = np.interp(mean_recall, recall[::-1], precision[::-1])
            precisions.append(interp_prec)
            aps.append(average_precision_score(y_bin[:, i], res['y_scores'][:, i]))
            
        mean_prec = np.mean(precisions, axis=0)
        std_prec = np.std(precisions, axis=0)
        mean_ap = np.mean(aps)
        std_ap = np.std(aps)
        
        plt.plot(mean_recall, mean_prec, color=color,
                 label=f'{class_names[i]} (AP = {mean_ap:.4f} ± {std_ap:.4f})',
                 lw=2, alpha=0.8)
        
        prec_upper = np.minimum(mean_prec + std_prec, 1)
        prec_lower = np.maximum(mean_prec - std_prec, 0)
        plt.fill_between(mean_recall, prec_lower, prec_upper, color=color, alpha=0.2)

    # B. Micro-average PR
    precs_micro = []
    aps_micro = []
    for res in all_metrics:
        y_bin = label_binarize(res['y_true'], classes=list(range(n_classes)))
        precision_micro, recall_micro, _ = precision_recall_curve(y_bin.ravel(), res['y_scores'].ravel())
        interp_prec = np.interp(mean_recall, recall_micro[::-1], precision_micro[::-1])
        precs_micro.append(interp_prec)
        aps_micro.append(average_precision_score(y_bin.ravel(), res['y_scores'].ravel()))
        
    mean_prec_micro = np.mean(precs_micro, axis=0)
    mean_ap_micro = np.mean(aps_micro)
    std_ap_micro = np.std(aps_micro)
    
    plt.plot(mean_recall, mean_prec_micro, color='gold', linestyle='--',
             label=f'Micro-average (AP = {mean_ap_micro:.4f} ± {std_ap_micro:.4f})',
             lw=2, alpha=0.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Averaged PR Curves (5 Seeds)')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{namkaran}_multiseed_pr.png"), dpi=150, bbox_inches='tight')
    plt.show()