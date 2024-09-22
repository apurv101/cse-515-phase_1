# Task 1: Video Feature Extraction Using R3D-18 Model

This project implements **Task 1**, which involves visualizing a video (optionally) and extracting neural network-based feature descriptors from specific layers of the pre-trained **R3D-18** architecture.

## 1. Overview

The script provided extracts feature vectors from three layers of the **R3D-18** model:
- **Layer 3**: The output is a 256 × 8 × 14 × 14 tensor, which is reduced to a 512-dimensional vector by averaging each 4 × 14 × 14 subtensor.
- **Layer 4**: The output is a 512 × 4 × 7 × 7 tensor, which is reduced to a 512-dimensional vector by averaging each 4 × 7 × 7 subtensor.
- **AvgPool Layer**: The output is a 512-dimensional vector.

You can also choose to visualize the video being processed before extracting features by enabling the `--visualize` option.

## 2. Requirements

Ensure you have the following dependencies installed:
- **Python 3.7+**
- **PyTorch** (for model loading and tensor operations)
- **Torchvision** (for pre-trained video models)
- **OpenCV** (for video visualization)
- **NumPy** (for numerical operations)
- **PyAV** (for video decoding) – this library is required for reading video files efficiently.
- **FFmpeg** – PyAV requires FFmpeg to handle various video formats.

You can install the required Python packages using the following command:
```bash
pip install torch torchvision opencv-python numpy av
```

Additionally, ensure that **FFmpeg** is installed on your system:
- **Ubuntu**:
  ```bash
  sudo apt update
  sudo apt install ffmpeg
  ```
- **MacOS** (using Homebrew):
  ```bash
  brew install ffmpeg
  ```

## 3. How to Run the Script

### 3.1. Command Line Arguments
The script accepts the following arguments:
- `--video`: The path to the video file you wish to process.
- `--visualize`: Optionally display the video as it is being processed. This flag is optional, and if omitted, no video will be shown.

### 3.2. Running the Script
Once you have saved the file as `task_1.py`, you can run the script from the terminal by specifying the video file path.

#### Example 1: Run the script without video visualization
```bash
python task_1.py --video ./target_videos/video1.avi
```

#### Example 2: Run the script with video visualization
```bash
python task_1.py --video ./target_videos/video1.avi --visualize
```

### 3.3. Expected Output
After running the script, you will see the following:
1. **If `--visualize` is enabled**: The video will be displayed frame-by-frame using OpenCV.
2. **Feature Extraction**: The script will print three 512-dimensional feature vectors:
   - Feature vector from **Layer 3**.
   - Feature vector from **Layer 4**.
   - Feature vector from **AvgPool Layer**.

The output will look something like this:
```
Feature vector from layer3 (512-dimensional):
tensor([...])

Feature vector from layer4 (512-dimensional):
tensor([...])

Feature vector from avgpool (512-dimensional):
tensor([...])
```

## 4. File Structure

```
.
├── task_1.py          # Python script to run Task 1
└── target_videos/     # Directory containing the video files
```

Make sure to modify the `--video` argument to point to the correct path of the video file.

## 5. Theory and Explanation of Key Decisions

1. **Sampling Frames**: 
   - We sample 16 evenly spaced frames from the video to capture temporal information while reducing computational cost.
   
2. **Permuting Tensor Dimensions**:
   - The video is loaded in the format `(T, H, W, C)` but needs to be permuted to `(T, C, H, W)` to match the input format required by the R3D-18 model.

3. **Hooks for Layer Outputs**:
   - Hooks are attached to `layer3`, `layer4`, and `avgpool` to extract intermediate outputs during the forward pass. This is necessary to capture the feature descriptors from these layers without altering the original model.

4. **Averaging Subtensors**:
   - For `layer3` and `layer4`, we group channels and average them, along with averaging over the spatial dimensions. This ensures the reduction of high-dimensional tensors into the required 512-dimensional feature vectors.



## 6. Troubleshooting

- If the video doesn't load correctly, ensure the file format is supported by OpenCV (e.g., `.avi`, `.mp4`).
- If you encounter errors regarding PyTorch or torchvision not being installed, make sure you have installed the required libraries using `pip`.

## 7. Conclusion

This script implements Task 1 by extracting feature descriptors from a pre-trained **R3D-18** model. The extracted features can be used for various video analysis tasks such as classification, retrieval, or similarity matching.
