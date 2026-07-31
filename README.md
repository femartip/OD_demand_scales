# Vision Demand Scales to Estimate Object Detection Capabilities

# Current Implementation
## Methodology 

Based on OECD vision capability scale:
1. Dynamic versus static input
2. Sensor type and resolution
3. Mobile versus static camera
4. Real versus staged or synthetic environment
5. Visual task
6. Environmental complexity
7. Environment type
8. Lighting and weather
9. Real-time requirements
10. Visual-only versus multimodal input

The work combines it into a single image-demand scale. Where it is rephrased to focus con the capability required for the instance, focusing on image quality, where aplications are removed. 

Files data/prompts/few_shot_prompt_1dim_rubric.txt and data/prompts/one_shot_prompt_1dim_rubric.txt contain the prompts used, data/prompts/images contains the example images for the few shot prompt. 

Same rubric used for two tasks:
- Localisation, the prompt defines the task as determining object positions using tight rectangular bounding boxes.
- Object detection, it defines the task as localisation plus assigning each box to one of the available categories.

Pipeline:
1. Insert the task definition into the modified rubric.
2. Provide either the rubric alone or the rubric plus few-shot examples.
3. Submit the image to GPT-4V.
4. Parse the output into an integer from 1 to 5.
5. Group images according to their predicted demand level.
6. Measure the performance of several object detectors inside each group.
7. Plot detector performance against demand level as a model characteristic curve.