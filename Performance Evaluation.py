#Calculate evaluation indicators
import os
import numpy as np
import cv2


def read_mask(file_path, target_size=None):
    """读取掩膜图像并调整大小"""
    mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if target_size:
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    return mask


def calculate_metrics(prediction, ground_truth):
    """Calculate multiple evaluation indicators"""
    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    tp = np.logical_and(prediction, ground_truth).sum()  # True Positive
    fp = np.logical_and(prediction, np.logical_not(ground_truth)).sum()  # False Positive
    fn = np.logical_and(np.logical_not(prediction), ground_truth).sum()  # False Negative
    tn = np.logical_and(np.logical_not(prediction), np.logical_not(ground_truth)).sum()  # True Negative

    dice = (2 * tp) / (2 * tp + fp + fn)
    jaccard = tp / (tp + fp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)

    return tp,fp,fn,tn,dice, jaccard, accuracy, sensitivity, specificity, precision


def evaluate_masks(pred_folder, gt_folder):
    """Evaluate all masks in the folder"""
    gt_files = sorted(os.listdir(gt_folder))

    tp_sum = []
    fp_sum = []
    fn_sum = []
    tn_sum = []

    dice_scores = []
    jaccard_scores = []
    accuracies = []
    sensitivities = []
    specificities = []
    precisions = []
    num = 0
    for gt_file in gt_files:
        gt_path = os.path.join(gt_folder, gt_file)
        pred_path = os.path.join(pred_folder, gt_file)
        num +=1
        if not os.path.exists(pred_path):
            print(f"Prediction file does not exist：{pred_path}")
            continue

        gt_mask = read_mask(gt_path)
        pred_mask = read_mask(pred_path, target_size=gt_mask.shape[::-1])  # 确保尺寸一致

        tp,fp,fn,tn,dice, jaccard, accuracy, sensitivity, specificity, precision = calculate_metrics(pred_mask, gt_mask)

        tp_sum.append(tp)
        fp_sum.append(fp)
        fn_sum.append(fn)
        tn_sum.append(tn)

        dice_scores.append(dice)
        jaccard_scores.append(jaccard)
        accuracies.append(accuracy)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        precisions.append(precision)

    metrics = {
        'TP': np.sum(tp_sum),
        'FP': np.sum(fp_sum),
        'FN': np.sum(fn_sum),
        'TN': np.sum(tn_sum),
        'Dice': np.mean(dice_scores),
        'Jaccard': np.mean(jaccard_scores),
        'Accuracy': np.mean(accuracies),
        'Sensitivity': np.mean(sensitivities),
        'Specificity': np.mean(specificities),
        'Precision': np.mean(precisions)
    }
    print("Total number of images：", num)
    return metrics


# folder path
pred_folder = "dataset/test_result"
gt_folder = "dataset/masks_test"
# Calculate evaluation indicators
metrics = evaluate_masks(pred_folder, gt_folder)
print("Evaluation indicators：")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")


