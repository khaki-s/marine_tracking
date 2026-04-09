# Marine Tracking

Marine Tracking is a comprehensive toolkit for tracking and analyzing the movement behavior of marine organisms (crabs, mussels, and shrimps). This project utilizes computer vision and machine learning techniques to extract motion trajectories of marine organisms from video data and analyzes them in conjunction with environmental data (such as tides) to study behavioral patterns.

## Project Structure

```text
marine_tracking/
|-- track/
|   |-- crab/
|   |   |-- code/
|   |   `-- distance_results/
|   |-- mussel/
|   |   |-- code/
|   |   |-- mussel/
|   |   |-- out/
|   |   `-- results/
|   `-- shrimp/
|       |-- code/
|       |-- distance/
|       |-- figure/
|       |-- R/
|       `-- tide/
|-- eval/
|   |-- crab/
|   |   |-- code/
|   |   `-- result/
|   |-- mussel/
|   |   |-- evaluate.py
|   |   |-- merge.py
|   |   `-- out/
|   `-- shrimp/
|       |-- code/
|       `-- results/
|-- utils/
|-- ultralytics/
|-- requirements.txt
|-- setup.py
`-- README.md
```

## Features

### 1. Multi-species Tracking

- **Crab Tracking**: Uses YOLOv8 and DeepSort algorithms to track crab movement, calculating displacement and velocity.
- **Mussel Monitoring**: Detects position changes in mussels and analyzes their movement patterns.
- **Shrimp Trajectories**: Plots movement trajectories of shrimps and analyzes their behavioral patterns.

### 2. Data Analysis

- **Tide Correlation Analysis**: Correlates marine organism movement with tidal data to study the influence of environmental factors on behavior.
- **Time Series Analysis**: Uses R language for time series analysis, including Simplex projection and S-map nonlinear analysis.
- **Spatial Distribution Visualization**: Generates heatmaps and trajectory plots to visually represent spatial distribution and movement patterns.

### 3. Image Processing

- **Video Processing**: Extracts frames from videos for object detection and tracking.
- **OCR Time Extraction**: Extracts timestamp information from video frames.
- **Image Enhancement**: Provides image enhancement functionality to improve video quality in low-light conditions.

## Technology Stack

- **Computer Vision**: YOLOv8 object detection, DeepSort multi-object tracking
- **Data Analysis**: Python (NumPy, SciPy, Pandas) and R language
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: PyTorch, Ultralytics

## Installation Guide

### Requirements

- Python 3.9.20
- R 4.4.3
- ultralytics-8.3.27 you can find it on: https://github.com/ultralytics/ultralytics
- CUDA-enabled GPU

### Installation Steps

1. Clone the repository:

```bash
git clone https://github.com/yourusername/marine_tracking.git
cd marine_tracking
```

2. Install Python dependencies:

```bash
pip install -e .
# or
pip install -r requirements.txt
```

3. Install R packages (for advanced data analysis):

```R
install.packages(c("rEDM", "ggplot2", "dplyr"))
```

## Usage Guide

### Crab Tracking

```bash
python track/crab/code/yolo_deepsort_dir_v4.py
```

This script processes video files in the specified directory, tracks crab movement, and generates output videos with trajectory and velocity information.

### Mussel Monitoring

```bash
python track/mussel/code/track.py
```

This script analyzes mussel position changes between consecutive image frames, marking moving and stationary individuals.

### Shrimp Trajectory Plotting

```bash
python track/shrimp/code/paint_trajectory_dir.py
```

This script processes shrimp videos, plots their movement trajectories, and saves the results in PDF format.

### Data Analysis

R scripts located in the `shrimp/R/` directory are used for advanced time series analysis:

- `simplex+s-map.R`: Uses Simplex projection and S-map for nonlinear time series analysis
- `ECCM.R`: Performs convergent cross mapping analysis

## Evaluation

### Crab Tracker Comparison

Scripts in `eval/crab/code/` generate MOT/TrackEval-style result files for different trackers:

- `track_deepsort.py`
- `track_bytetrack.py`
- `track_botsort.py`

Ground-truth reference file:

- `eval/crab/result/gt.txt`

### Mussel Statistical Evaluation

```bash
python eval/mussel/merge.py
python eval/mussel/evaluate.py
```

`evaluate.py` reports metrics such as Spearman correlation, RMSE, MAE, and limits of agreement.

### Shrimp Evaluation Scripts

Evaluation and tracker-output scripts are located in `eval/shrimp/code/`, with outputs under `eval/shrimp/results/`.

## Data Formats

### Input Data

- **Video Files**: Supports common video formats (e.g., .mp4, .mkv)
- **Image Sequences**: JPEG images arranged in chronological order
- **Tidal Data**: Tidal height data in CSV format

### Output Data

- **Trajectory Videos**: Video files with tracking markers
- **Displacement Data**: Displacement and velocity data in CSV format
- **Visualization Charts**: Trajectory and analysis charts in PDF format

## Models

The project uses pre-trained YOLOv8 models for object detection:

- `yolov8n.pt`: Lightweight model suitable for resource-constrained environments
- `yolov8x.pt`: Large model providing higher detection accuracy

## Case Studies
Video data for this study were sourced from the EMSO-Azores deep-sea observation network, specifically from the Lucky Strike hydrothermal vent field (37°N, 32°W) located on the Mid-Atlantic Ridge. you can see it on: https://www.emso-fr.org/Azores

The project contains data analysis for multiple years:

- 2016-2017
- 2017-2018
- 2018-2019
- 2019-2020
- 2020-2021

Each period's dataset includes distance results for crabs, which can be used to compare changes in marine organism behavior across different periods.

## Model Weights and Research Papers

The model weights for all three species and related research papers are publicly available on Figshare:
[https://doi.org/10.6084/m9.figshare.29931491.v5](https://doi.org/10.6084/m9.figshare.29931491.v5)

## Contact

For questions or collaborations, please contact: chujingyi@stu.ouc.edu.cn


