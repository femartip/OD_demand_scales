import json
import sys
import os
import matplotlib.pyplot as plt


def get_classes(data, term, gt_name):
    class_dict = {}
    print(len(data))
    
    for sample in data:
        if sample[gt_name] != None:
            for gt in sample[gt_name]["detections"]:
                if gt[term] in class_dict:
                    class_dict[gt[term]] +=1
                else:
                    class_dict[gt[term]] = 1
        
    return class_dict



def list_files_in_directory(directory_path):
    file_names = []
    
    for file_name in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, file_name)):
            file_names.append(file_name)
    
    return file_names


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Usage: python script_name.py <dataset>")
        sys.exit(1)

    dataset = str(sys.argv[1])

    if dataset == "coco-2017":
        term = "supercategory"
    elif dataset == "voc-2007"or dataset=="driving":
        term = "label"

    PREDEFINED_DATASETS = ["coco-2017","voc-2007"]
    
    predefined = dataset in PREDEFINED_DATASETS

    if predefined:
        gt_name = "ground_truth"
    else:
        gt_name = "detections"

    directory_path = f'./results_{dataset}'
    files_list = list_files_in_directory(directory_path)
    
    for file in [files_list[0]]:
        print(file)

        f = open(f'results_{dataset}/' + file)
        data = json.load(f)

        results = get_classes(data["samples"], term, gt_name)

    sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    labels = list(sorted_results.keys())
    values = list(sorted_results.values())

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color = "#31005c")
    plt.xticks(rotation=90)  
    plt.xlabel("Classes")
    plt.ylabel("Frequency")
    plt.tight_layout()

    plt.savefig(f"histogram_{dataset}.pdf")
