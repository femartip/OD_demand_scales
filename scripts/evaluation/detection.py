import json
import sys
import os

def check_inclusion(pred_box,gt_box):
    return pred_box[0] >= gt_box[0] and pred_box[1] >= gt_box[1] and pred_box[2] <= gt_box[2] and pred_box[3] <= gt_box[3]

def bb_intersection_over_union(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))

        if interArea == 0:
            return 0

        boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou


def evaluate_detection(data, predefined):
    threshold_iou = 0.5
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    label_mismatch = 0

    prediction_detections = data['predictions']['detections']

    if predefined:
        detections = data['ground_truth']
    else:
        detections = data['detections']

    if detections == None:
        return {
            'true_positives': 0,
            'false_positives': len(prediction_detections),
            'false_negatives': 0,
            'label_mismatch': 0
        }
    else:
        ground_truth_detections = detections['detections']

    
    
    boxes_covered = []
    
    for pred in prediction_detections:
        pred_box = pred['bounding_box']
        pred_adapted = [ pred_box[0], pred_box[1], pred_box[0]+pred_box[2], pred_box[1]+pred_box[3] ]
        matched = False
        max_iou = 0
        max_index = 0
        
        for i, gt in enumerate(ground_truth_detections):
            gt_box = gt['bounding_box']
            crowd = gt['iscrowd'] == 1 if dataset=="coco-2017" else 0

            gt_adapted  = [ gt_box[0], gt_box[1], gt_box[0]+gt_box[2], gt_box[1]+gt_box[3] ]
            
            if crowd and check_inclusion(pred_adapted,gt_adapted):
                matched = True
                max_iou = 1
                max_index = i
                break
            else:
                iou = bb_intersection_over_union(pred_adapted, gt_adapted)

            
            if iou >= max_iou:
                max_iou = iou
                max_index = i

        if max_iou >= threshold_iou:
            matched = True 
            
        
        if matched:
            if max_index not in boxes_covered :
                true_positives += 1
                boxes_covered.append(max_index)
                if ground_truth_detections[max_index]["label"] != pred["label"]:
                    label_mismatch += 1
            elif crowd:
                true_positives += 1
            else:
                false_positives += 1
        else:

            false_positives += 1

    
    false_negatives = len(ground_truth_detections) - len(boxes_covered)
    
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'label_mismatch': label_mismatch
    }

def list_files_in_directory(directory_path):
    file_names = []
    
    for file_name in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, file_name)):
            file_names.append(file_name)
    
    return file_names

def save_json_to_folder(data, folder_path, file_name):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    file_path = os.path.join(folder_path, file_name)
    
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
    print(f"JSON file saved at {file_path}")

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Usage: python script_name.py <dataset>")
        sys.exit(1)

    dataset = str(sys.argv[1])

    PREDEFINED_DATASETS = ["coco-2017","voc-2007"]
    
    predefined = dataset in PREDEFINED_DATASETS

    directory_path = f'./results_{dataset}'

    files_list = list_files_in_directory(directory_path)
    
    for file in files_list:

        f = open(directory_path +"/"+ file)
        data = json.load(f)

        for i in range(len(data["samples"])):
            results = evaluate_detection(data["samples"][i],predefined)
            data["samples"][i]['detection_tp'] = results['true_positives']
            data["samples"][i]['detection_fp'] = results['false_positives']
            data["samples"][i]['detection_fn'] = results['false_negatives']
 
        save_json_to_folder(data, f'./detections_{dataset}', file)
