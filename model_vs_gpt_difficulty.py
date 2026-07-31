import pandas as pd
import re
import matplotlib.pyplot as plt
import sys

# Configuration
if len(sys.argv) < 3:
    print("Usage: python script_name.py <version> <task>")
    sys.exit(1)


version = str(sys.argv[1])

task = str(sys.argv[2])


df = pd.read_csv("detection_difficulty.csv",dtype={'image_id': object}, usecols= ["image_id","accuracy","accuracy_detection"])

if task == "localization" or task == "detection":
    gpt_diff = pd.read_csv(f"v{version}_{task}_fewshot_dataset_gpt_difficulty.csv", dtype={'image_id': object})
else:
    gpt_diff = pd.read_csv(f"v{version}_fewshot_dataset_gpt_difficulty.csv", dtype={'image_id': object})


gpt_diff = gpt_diff[gpt_diff['level'] != 'error']


try:
    gpt_diff["level"] = gpt_diff["level"].apply(lambda x: int(re.search(r'\d+', x).group()))
except TypeError:
    pass

level_counts = gpt_diff["level"].value_counts().sort_index()
print(level_counts)

level_counts.plot(kind='bar', color = "#31005c")

plt.xticks(rotation=0)
plt.xlabel('Level')
plt.ylabel('Count')
plt.title('Localisation Level Distribution')

plt.savefig(f"v{version}_{task}_level_distribution.pdf")


grouped = df.groupby('image_id', as_index=False).mean()
print(grouped)

final = pd.merge(grouped, gpt_diff, on='image_id')
final = final.dropna()
final = final[(final['level'] >= 1) & (final['level'] <= 4)]


print(final)

plt.figure(figsize=(9, 6))

avg_trend = final.groupby('level')['accuracy'].mean().reset_index()

avg_trend['level'] = avg_trend['level'].astype(str)


print(avg_trend)

plt.plot(avg_trend['level'], avg_trend['accuracy'], color='#FF6859', linewidth=2, label='Object Detection Accuracy',marker='o')


df = pd.read_csv("detection_difficulty.csv",dtype={'image_id': object}, usecols= ["image_id","accuracy_detection"])

grouped = df.groupby('image_id', as_index=False).mean()
final = pd.merge(grouped, gpt_diff, on='image_id')
final = final.dropna()
final = final[(final['level'] >= 1) & (final['level'] <= 4)]

avg_trend = final.groupby('level')['accuracy_detection'].mean().reset_index()

avg_trend['level'] = avg_trend['level'].astype(str)

plt.plot(avg_trend['level'], avg_trend['accuracy_detection'], color='skyblue', linewidth=2, label='Localization Accuracy',marker='o')



plt.xlabel('Level')
plt.ylabel('Accuracy')
plt.legend()
if task == "localization" or task == "detection":
    plt.savefig(f"v{version}_{task}_fewshot_detection_scatterplot.pdf")
else:
    plt.savefig(f"v{version}_fewshot_detection_scatterplot.pdf")

plt.show()