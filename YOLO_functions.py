import os
import cv2
import matplotlib.pyplot as plt

def load_images_by_class(df_target, classes, base_path):
    """
    Loads one representative image per class from the given dataframe.

    Args:
        df_target: Pandas DataFrame containing object detection info.
        classes: List of class IDs to process.
        base_path: Directory where image files are stored.

    Returns:
        images: Dictionary of class_id -> RGB image.
        boxes: Dictionary of class_id -> [xmin, xmax, ymin, ymax]
    """
    boxes = {}
    images = {}

    for class_id in classes:
        print(f"Processing class {class_id}...")
        target_rows = df_target[df_target['class_id'] == class_id]
        print(f"Number of rows for class {class_id}: {len(target_rows)}")

        if target_rows.empty:
            print(f"No data found for class {class_id}. Skipping.")
            continue

        image_found = False
        for _, row in target_rows.iterrows():
            file_name = row['frame']
            image_path = os.path.join(base_path, file_name)

            if not os.path.exists(image_path):
                continue

            print(f"Found image {image_path} for class {class_id}.")

            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image {image_path}. Trying next row...")
                continue
            else:
                image_found = True
                break

        if image_found:
            print(f"Image found AND read for class {class_id}: {image_path}")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images[class_id] = image_rgb
            boxes[class_id] = [
                int(row['xmin']),
                int(row['xmax']),
                int(row['ymin']),
                int(row['ymax'])
            ]

    return images, boxes


def extract_yolo_predictions(results, filename, yolo_to_custom):
    """
    Extracts YOLO predictions from results and maps them to custom class IDs.

    Args:
        results: YOLO model inference results (expects results[0].boxes).
        filename: Name of the current frame/image.
        yolo_to_custom: Dictionary mapping YOLO class IDs to custom class IDs.

    Returns:
        List of predictions: [frame, xmin, xmax, ymin, ymax, custom_class_id]
    """
    predictions_data = []

    for box in results[0].boxes:
        frame = filename

        center_x, center_y, width, height = box.xywh[0].int().tolist()

        xmin = int(center_x - width / 2)
        ymin = int(center_y - height / 2)
        xmax = int(center_x + width / 2)
        ymax = int(center_y + height / 2)

        class_id = box.cls.int().item()
        custom_class_id = yolo_to_custom.get(class_id, -1)

        predictions_data.append([frame, xmin, xmax, ymin, ymax, custom_class_id])
    
    return predictions_data


def compute_iou(boxA, boxB):
    """
    Computes Intersection over Union (IoU) between two bounding boxes.

    Args:
        boxA: [xmin, ymin, xmax, ymax]
        boxB: [xmin, ymin, xmax, ymax]

    Returns:
        IoU value (float)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) != 0 else 0
    return iou


def calculate_iou_for_predictions(predictions_df, ground_truth_df):
    """
    Matches predictions to ground truth boxes and computes IoUs.

    Args:
        predictions_df: DataFrame of predicted boxes.
        ground_truth_df: DataFrame of ground truth boxes.

    Returns:
        iou_info: List of [pred_index, gt_index, pred_class, gt_class, IoU]
    """
    iou_info = []

    for i, pred in predictions_df.iterrows():
        frame = pred['frame']
        pred_box = [pred['xmin'], pred['ymin'], pred['xmax'], pred['ymax']]
        pred_class = pred['class_id']

        gt_boxes = ground_truth_df[ground_truth_df['frame'] == frame]

        for j, gt in gt_boxes.iterrows():
            gt_box = [gt['xmin'], gt['ymin'], gt['xmax'], gt['ymax']]
            gt_class = gt['class_id']

            iou = compute_iou(pred_box, gt_box)

            iou_info.append([i, j, pred_class, gt_class, iou])

    return iou_info