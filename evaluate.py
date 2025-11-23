import os
import numpy as np
import nibabel as nib


def compute(mask1, mask2, num_classes):
    dice_scores = []
    iou_scores = []
    recall_scores = []

    for class_id in range(1, num_classes + 1):
        mask1_class = (mask1 == class_id).astype(np.uint8)
        mask2_class = (mask2 == class_id).astype(np.uint8)

        # Dice
        intersection = np.sum(mask1_class * mask2_class)
        denominator = np.sum(mask1_class) + np.sum(mask2_class)
        if denominator == 0:
            dice = 1.0 if intersection == 0 else 0.0
        else:
            dice = (2.0 * intersection) / denominator
        dice_scores.append(dice)

        # IoU
        union = np.sum(np.logical_or(mask1_class, mask2_class))
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        iou_scores.append(iou)

        # Recall
        tp = intersection
        fn = np.sum(mask1_class) - tp
        if mask1_class.sum() == 0:
            if mask2_class.sum() == 0:
                recall = 1.0
            else:
                recall = 0.0
        else:
            recall = tp / (tp + fn + 1e-8)
        recall_scores.append(recall)

    return dice_scores, iou_scores, recall_scores


def main(folder1, folder2, num_classes=3):
    file_names = [f for f in os.listdir(folder1) if f.endswith('.nii.gz')]

    total_dice = np.zeros(num_classes)
    total_iou = np.zeros(num_classes)
    total_recall = np.zeros(num_classes)
    count = 0

    results = {}

    for file_name in file_names:
        path1 = os.path.join(folder1, file_name)
        path2 = os.path.join(folder2, file_name)

        if not os.path.exists(path1) or not os.path.exists(path2):
            print(f"Warning: File {file_name} not found in one of the folders.")
            continue

        try:
            img1 = nib.load(path1)
            img2 = nib.load(path2)
        except Exception as e:
            print(f"Error loading file {file_name}: {e}")
            continue

        data1 = img1.get_fdata()
        data2 = img2.get_fdata()

        if data1.shape != data2.shape:
            print(f"Warning: Shape mismatch for file {file_name}. Skipping.")
            continue

        dice, iou, recall = compute(data1, data2, num_classes)

        results[file_name] = {'Dice': dice, 'IoU': iou, 'Recall': recall}

        total_dice += np.array(dice)
        total_iou += np.array(iou)
        total_recall += np.array(recall)
        count += 1

        print(f"File: {file_name}")
        for class_id in range(1, num_classes + 1):
            print(f"  Class {class_id} - Dice: {dice[class_id - 1]:.4f}, IoU: {iou[class_id - 1]:.4f}, Recall: {recall[class_id - 1]:.4f}")
        print("-" * 50)

    if count > 0:
        avg_dice = total_dice / count
        avg_iou = total_iou / count
        avg_recall = total_recall / count

        print("\nOverall Statistics:")
        for class_id in range(1, num_classes + 1):
            print(
                f"  Class {class_id} - Average Dice: {avg_dice[class_id - 1]:.4f}, "
                f"Average IoU: {avg_iou[class_id - 1]:.4f}, "
                f"Average Recall: {avg_recall[class_id - 1]:.4f}"
            )
    else:
        print("No valid files found.")

    return results, avg_dice, avg_iou, avg_recall

if __name__ == "__main__":
    folder1 = r'C:\Users\Administrator\Desktop\nnUNet_master\DATASET\nnUNet_raw\Dataset102_quan\labelsTs'
    folder2 = r'C:\Users\Administrator\Desktop\nnUNet_master\DATASET\nnUNet_raw\Dataset102_quan\Regoutput'
    num_classes = 3  # 根据你的标签类别数量设置

    results, avg_dice, avg_iou, avg_recall = main(folder1, folder2, num_classes)

### 原版
# File: ATM_729.nii.gz
#   Class 1 - Dice: 0.8930, IoU: 0.8067, Recall: 0.9414
#   Class 2 - Dice: 0.8149, IoU: 0.6876, Recall: 0.8271
#   Class 3 - Dice: 0.3248, IoU: 0.1939, Recall: 0.2878
# --------------------------------------------------
# File: ATM_731.nii.gz
#   Class 1 - Dice: 0.8951, IoU: 0.8102, Recall: 0.9161
#   Class 2 - Dice: 0.8203, IoU: 0.6954, Recall: 0.9242
#   Class 3 - Dice: 0.4238, IoU: 0.2689, Recall: 0.4088
# --------------------------------------------------
# File: ATM_732.nii.gz
#   Class 1 - Dice: 0.8944, IoU: 0.8089, Recall: 0.9497
#   Class 2 - Dice: 0.8856, IoU: 0.7947, Recall: 0.8288
#   Class 3 - Dice: 0.4592, IoU: 0.2981, Recall: 0.3941
# --------------------------------------------------
# File: ATM_734.nii.gz
#   Class 1 - Dice: 0.8998, IoU: 0.8178, Recall: 0.9223
#   Class 2 - Dice: 0.9091, IoU: 0.8334, Recall: 0.9122
#   Class 3 - Dice: 0.5460, IoU: 0.3755, Recall: 0.4996
# --------------------------------------------------

# Overall Statistics:
#   Class 1 - Average Dice: 0.8956, Average IoU: 0.8109, Average Recall: 0.9324
#   Class 2 - Average Dice: 0.8575, Average IoU: 0.7528, Average Recall: 0.8731
#   Class 3 - Average Dice: 0.4385, Average IoU: 0.2841, Average Recall: 0.3976


### 回归
#   Class 1 - Dice: 0.8780, IoU: 0.7825, Recall: 0.9413
#   Class 2 - Dice: 0.8265, IoU: 0.7043, Recall: 0.8135
#   Class 3 - Dice: 0.3232, IoU: 0.1927, Recall: 0.2884
# --------------------------------------------------
# File: ATM_731.nii.gz
#   Class 1 - Dice: 0.8967, IoU: 0.8127, Recall: 0.9122
#   Class 2 - Dice: 0.8324, IoU: 0.7130, Recall: 0.9089
#   Class 3 - Dice: 0.4142, IoU: 0.2612, Recall: 0.3950
# --------------------------------------------------
# File: ATM_732.nii.gz
#   Class 1 - Dice: 0.8916, IoU: 0.8044, Recall: 0.9465
#   Class 2 - Dice: 0.8854, IoU: 0.7944, Recall: 0.8236
#   Class 3 - Dice: 0.4511, IoU: 0.2913, Recall: 0.3973
# --------------------------------------------------
# File: ATM_734.nii.gz
#   Class 1 - Dice: 0.8943, IoU: 0.8087, Recall: 0.9151
#   Class 2 - Dice: 0.9024, IoU: 0.8222, Recall: 0.9021
#   Class 3 - Dice: 0.5329, IoU: 0.3633, Recall: 0.4922
# --------------------------------------------------

# Overall Statistics:
#   Class 1 - Average Dice: 0.8901, Average IoU: 0.8021, Average Recall: 0.9288
#   Class 2 - Average Dice: 0.8617, Average IoU: 0.7585, Average Recall: 0.8620
#   Class 3 - Average Dice: 0.4304, Average IoU: 0.2771, Average Recall: 0.3932