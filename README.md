# Pulmo Vision – Lung Cancer Detection

Pulmo Vision is a lung cancer prediction system featuring a deep learning backend and an interactive frontend interface for analyzing lung MRI images and displaying Grad-CAM class activation mapping heatmaps to assist in clinical visualization.

## Features

- **Transfer Learning Model**: Employs EfficientNetB0 pre-trained on ImageNet to classify MRI scans.
- **Explainable AI (Grad-CAM)**: Visualizes the regions of the MRI scans contributing to the model's prediction.
- **FastAPI Server**: High-performance backend servers for model execution and image prediction.
- **Interactive UI**: Upload MRI scans, run prediction, and view output results & heatmaps instantly.
