Smartphone PPG–Based Anemia Detection
Using Photoplethysmography (PPG) + Machine Learning + Streamlit GUI

A complete end-to-end system to detect anemia using only smartphone fingertip videos.

 Overview

This project implements a machine-learning pipeline that predicts anemia by analyzing a fingertip video captured using a smartphone camera.
The system extracts PPG (photoplethysmography) signals, computes physiological features, and classifies anemia using a Random Forest model.

It combines two biomedical datasets:

1️ Mendeley Hemoglobin PPG Dataset

Used to train anemia patterns (manual features).

2️ BIDMC PPG Waveform Dataset (MIT-BIH)

Used to validate model performance + generate model-compatible waveform features.

The final model is integrated into a Streamlit web application for real-time anemia inference.

  Goals

Extract PPG signals from smartphone videos

Filter and preprocess raw PPG

Compute HR, HRV, RR intervals, amplitude metrics

Train Random Forest classifier for anemia detection

Predict anemia probability from any fingertip video

Display signal, peaks, features, and health guidance

Build a deployable Streamlit dashboard
# smartphone-ppg-anemia
Smartphone Camera and Flashlight real-time fingertip video processing  based Anemia Detection 

