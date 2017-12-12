# Rune Auto-activation System
Course work for MM 803 - Image Processing (Fall 2017), Group Project @ University of Alberta, Canada

You can find the project description at https://github.com/NunchakusLei/rune_auto_activation_system/blob/master/project_description.pdf

And, the project proposal at https://github.com/NunchakusLei/rune_auto_activation_system/blob/master/project_proposal.pdf



# Component List
- **manual_labelling.py**: A script help us to label the input video manually. [User Manual](#manual-labelling-user-manual)
- **common_func.py**: A set of functions that commonly use across the whole system, such as the normalization operation for image displaying.
- **mnist_deep.py**: A CNN handwritten digit recognizer implementation from Tensorflow: https://www.tensorflow.org/get_started/mnist/pros
- **mnist_deep_estimator.py**: A module perform the CNN to recognize the handwritten digit.
- **preprocess.py**: A module perform image processing techniques as preprocessing for grid searching and handwritten digit recognition. This module also contains the process of applying competition constraints to filter out outliner cell detections.
- **prediction_selection.py**: A module perform final adjusting to reduce error make by the CNN handwritten digit recognizer.
- **rune_activator.py**: The Rune Auto-activation System. [User Manual](#rune-auto-activation-system-manual)
- **rune_eval.py**: A module to evaluate the accuracy of the Rune Auto-activation System. [User Manual](#evaluation-manual)



# Rune Auto-activation System Manual

### Execution (How to run)
```bash
python rune_activator.py [-h] (-f INPUT_FILE_PATH | -c) (-v | -i)
```

optional arguments:

  -h, --help            show this help message and exit

required arguments:

  -f INPUT_FILE_PATH, --file INPUT_FILE_PATH
 
                        User input argument for the image source file path.

  -c, --camera          Feed with a webcam.

  -v, --video           Input source type indicated as video.

  -i, --image           Input source type indicated as image.
  

For example,
```bash
python rune_activator.py -f data/Competition2017_buff.mpeg -v
```
The command above will execute the Rune Auto-activation System feeding a video file from path ```data/Competition2017_buff.mpeg```.

Or,
For example,
```bash
python rune_activator.py -f data/Competition2017_buff.mpeg -v
```

### Key Command List
- **```p```**: Pause/Resume playing the testing video.
- **```s```**: Save current frame as image for further analysis.



# Evaluation Manual
### Execution (How to run)



# Manual Labelling User Manual
**Warning**, This script is developed specific for this project. And, due to the time limit, some feature might not be perfect.

### Execution
The first step to use this manual labelling script is to execute it. You should able to run the manual labelling script using command following,

```bash
python manual_labelling.py -f <labeling_file_path>
```

### How to label
Once the script is running, you should see the first frame from the video file have been displayed. New, if you move your mouse onto the frame, then **click and drag** your mouse to draw a bounding box. This bounding box will be one of the label. For grid area label, drawing a bounding box (green color) will be enough. But, a ground truth of the handwritten number in cells will need to be **enter through keyboard**.

Specifically, the process of labelling each frame break down into two major parts. First, you will need to label the grid area by dragging a bounding box around the grid area. Secondly, you will need to label each cell by dragging a bounding box around each cell one by one, and then type the ground truth number for the handwritten number in that cell using keyboard.

### Label
Each frame have two type of labels, one for grid area label and another one for cells. The **grid area label** will be a single bounding box (**green color**). The **label for cells** (**purple color**) includes a bounding box for each cell and a ground truth for the handwritten number in the cell.

This manual labelling script will help you to label those two type of labels in a video file. The final label will be store in file named ```<video_file_name>.rune_label```.

### Labelling Mode
There are two mode for labelling, "grid" and "cell". The defualt mode will be "grid" when you start labelling a new frame. Once you label the grid area, the mode will switch to "cell" automatically. You can switch back to "grid" mode by pressing ```c``` key.

### Key Command List
- **```n```   ```<Right-Arrow>```**: Go to next frame
- **```l```   ```<Left-Arrow>```**: Go to last frame
- **```c```**: Change labelling mode
- **```r```**: Erase the label in current frame
- **```DEL```**: Delete last stroke of label
- **```ESC```**: Exit the script and save label to file



# References
- https://www.tensorflow.org/get_started/mnist/pros
