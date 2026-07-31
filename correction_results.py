import json
import sys
import os

def correct_results(data):
    matched_predictions = set()  
    
    # Correct label mismatches between VOC and COCO
    for gt in data["ground_truth"]["detections"]:
        for idx, pred in enumerate(data["predictions"]["detections"]):
            
            if idx in matched_predictions:
                continue
            
            if (
                (
                    (gt["label"] == "aeroplane" and pred["label"] == "airplane") or
                    (gt["label"] == "sofa" and pred["label"] == "couch") or
                    (gt["label"] == "diningtable" and pred["label"] == "dining table") or
                    (gt["label"] == "motorbike" and pred["label"] == "motorcycle") or
                    (gt["label"] == "pottedplant" and pred["label"] == "potted plant") or
                    (gt["label"] == "tvmonitor" and pred["label"] == "tv")
                ) 
            ):
                
                matched_predictions.add(idx)
                
                pred["eval_coco"] = "tp"
                data["eval_coco_tp"] += 1
                data["eval_coco_fp"] -= 1
                data["eval_coco_fn"] -= 1
                break  
        
    
    return {
        'true_positives': data["eval_coco_tp"],
        'false_positives': data["eval_coco_fp"],
        'false_negatives': data["eval_coco_fn"]
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

        f = open(f'results_{dataset}/' + file)
        data = json.load(f)

        for i in range(len(data["samples"])):
            results = correct_results(data["samples"][i])
            data["samples"][i]['eval_coco_tp'] = results['true_positives']
            data["samples"][i]['eval_coco_fp'] = results['false_positives']
            data["samples"][i]['eval_coco_fn'] = results['false_negatives']
 
        save_json_to_folder(data, f'./correction_{dataset}', file)
