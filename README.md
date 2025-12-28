
This repository contains the implementation of framework for autonomous vehicles (AVs). The framework integrates transformer-based perception with symbolic reasoning to achieve real-time, explainable behavior analysis. The system processes multi-modal sensory data from AVs (e.g., camera, LiDAR, GPS, telemetry) and outputs behavior classifications with human-readable explanations.

Basic Requirements:
python
torch
numpy
scikit-learn
matplotlib


datasets used (CARLA-Sim-Traffic, nuScenes)

4. Hardware and Software Setup

Hardware:

NVIDIA RTX 3090 GPU

AMD Ryzen Threadripper 3970X CPU (32 cores)

128 GB DDR4 RAM

Ubuntu 22.04 LTS

Software:

Python 3.8 or above

PyTorch 2.0

CUDA 11.8

NumPy, SciPy

SHAP 0.41

ROS2 (for real-time deployment)

Docker (for containerized deployment)

5. Setup Datasets

We used the following datasets for our experiments:

CARLA-Sim-Traffic (Simulated Dataset):

Description: A dataset from the CARLA simulator with multi-modal sensory data (RGB, LiDAR, GPS, Telemetry) for autonomous vehicle training.

Download Link: CARLA GitHub

Use the provided scripts in the datasets/ directory to prepare the CARLA dataset.

nuScenes (Real-World Dataset):

Description: A multi-modal dataset capturing the behavior of real-world autonomous vehicles.

Download Link: nuScenes Dataset

You can download the dataset by following the instructions on the nuScenes website.

Custom Multi-Modal Dataset:

Description: A custom dataset to emulate real-world AV sensory streams and contextual information.

Access: The dataset is available upon request. Please contact the authors or check the supplementary material for more details.


Citations

If you use or reference this work, please cite the following paper:

Muhammad Usman Arshad, Yicong Li, Zhou Kuanjiu. Neuro-Symbolic Monitoring Framework for Autonomous Vehicles. Neuro-Symbolic AI, 2025.

Contact Information

For further questions, please contact:

Muhammad Usman Arshad: usmanarshad5050@outlook.com
