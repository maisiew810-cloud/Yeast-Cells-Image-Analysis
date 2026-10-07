# Yeast Cell Segmentation and Morphological Analysis
## Overview

This project aims to automate the analysis of Saccharomyces cerevisiae (budding yeast) cell morphology using brightfield microscopy images provided in PNG or TIFF format. The image analysis pipeline performs automated cell segmentation by identifying cell boundaries, filling internal holes, and applying Gaussian filtering and adaptive Otsu thresholding to generate accurate cell masks.
Following segmentation, the program presents each detected cell for user annotation, allowing researchers to classify cells into different morphological categories (e.g., single, small-budded, or large-budded). These annotations are used to quantify the distribution of cells across different cell morphologies and stages of the cell cycle (e.g., single = G0, small-budded=G1, large-budded= G2 & M phase).
For each analyzed cell, the pipeline automatically extracts morphological measurements and exports the results as a CSV file. Researchers can review and edit these annotations if necessary before using the curated dataset to train machine learning models capable of automatically classifying yeast cell morphologies and performing downstream statistical analyses.
The overall goal of this project is to reduce the time required for manual cell annotation, improve the consistency and reproducibility of morphological analysis.

Features
-	Reads PNG and TIFF filed images
-	Converts RGB images to grayscale
-	Applies Gaussian filtering to reduce image noise
-	Applies Otsu thresholding to convert to binary image
-	Fills holes within segmented cells
-	Removes small artifacts and blobs
-	Labels segmented cells
-	Extracts quantitative morphology measurements
-	Optional interactive cell classification
-	Saves results as CSV files
-	Generates cell segmentation, labeled-image visualizations, and feature datasets.
-	Uses generated datasets to train Random Forest, SVC, and KMeans models for cell morphology classification. 
-	Performs Grid Search for hyperparameter optimization. 
-	Evaluates model accuracy in identifying different cell morphologies. 
-	Conducts Dunn’s test to assess statistical differences in cell morphology features between cell types. 
-	Generates statistical visualizations, including KDE plots, histograms, PCA, t-SNE, and UMap to examine the distributions of cell morphology features and identify potential classification thresholds.

## Model Usage & Imports
-	Python 3.10+
-	Matplotlib
-	NumPy
-	Pandas
-	Skimage
-	Seaborn
-	Sklearn
-	Scipy
-	Torch
-	Scikit_Posthocs
-	Use this to install Scikit_Posthocs: pip install scikit-posthocs
Project Workflow
Input Raw Image -> Grayscale -> Gaussian Blur -> Local Otsu Thresholding -> Filling Holes -> Remove Artifacts -> Segment cell boundaries -> Cell labeling -> Region Property Extraction -> Data Output -> Import Data -> Train Models -> Predict Accuracy -> Conduct Dunn’s Test -> Generate KDE Plot

## How to Install
-	Prepare a package for users to install which is easier
How to Run
Mode 1 -> Morphometrics & Statistical Graphs [done]
 Performs cell morphometric analysis and generates statistical visualizations. 
Mode 2 -> Nucleolus-to-Cytosol Grayscale Signal Intensity Ratio [in progress]
 Calculates and analyzes the grayscale signal intensity ratio between the nucleolus and cytosol based off the region of interest (ROI).
Mode 3 -> Natural Language Processing Model [in progress]
 Uses a natural language processing model to interpret statistical graphs and convert the results into verbal descriptions.

Extracted Measurements
-	Cell Number
-	Cell Evaluated
-	Cell Type
-	Perimeter
-	Area
-	Perimeter Crofton
-	Eccentricity
-	Convex Area
-	Equivalent Diameter
-	Mean Intensity
-	Circularity
-	Image File

# References
https://scikit-image.org/docs/0.25.x/auto_examples/segmentation/plot_thresholding.html
https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_remove_objects.html
https://scikit-image.org/docs/stable/auto_examples/filters/plot_tophat.html
https://youtu.be/u3nG5_EjfM0?si=aTfsg2IInkP9g9Vu
https://forum.image.sc/t/regionprop-proposes-too-many-bounding-boxes/29898/3
https://medium.com/@michael71314/how-to-use-ocr-bounding-boxes-c00303bc11c4
https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
https://scikit-
https://medium.com/@abhishekjainindore24/gaussian-noise-in-machine-learning-aab693a10170
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
https://www.geeksforgeeks.org/machine-learning/k-means-clustering-introduction/
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
https://www.youtube.com/watch?v=uqiJah_kRxU&t=187s

# Acknowledgements
This project was developed in collaboration with Professor Xyrus Maurer-Alcalá, who provided technical guidance and support throughout the development of the image analysis pipeline. The microscopy images and biological expertise used in this project were provided by Professor Oliver Kerscher's Yeast Genetics Lab at the College of William & Mary.
This work was conducted as part of an undergraduate research project supported and funded through the Charles Center Summer Research Grant and the Mangum Fellowship.[
