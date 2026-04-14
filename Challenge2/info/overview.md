# Overview

## BCI Challenge

### What is a brain-computer interface?

A Brain-Computer Interface (BCI) is a system that allows a person to interact with a machine without any physical interaction. It works by extracting features from neuro-physiological signals (e.g., the power spectral densities on certain frequency bands) and assigning them to different classes. These classes may be associated to cognitive states, sensory responses, etc., and the features are chosen so that they are discriminative for each class. We call a paradigm the set of cognitive tasks that a subject is asked to perform when using a BCI system; different paradigms activate different brain mechanisms and yield different signal features that may be used later as features for classification.

In this competition, you will be working with data that follows the motor imagery paradigm. In this paradigm, a subject is asked to imagine movement, e.g lifting his hands, feet or tongue when a visual cue is displayed on a screen. The fact of voluntary imagining such movement produces Mu-waves in the motor cortex that may then be identified by a classifier algorithm. The laterality of the imagined movement (e.g., lifting the left-hand or right-hand) is reflected in the laterality of the production of Mu-waves, with different EEG spatial patterns being observed for each class of imagined movement. BCI systems using the motor imagery (MI) paradigm can be traced back to the 90s and are still often used in practice. Since the discriminant markers of the recorded EEG are related to oscillations in the Mu-band, most classifiers in the literature use the power spectral density of the signals in each electrode as features.

### The challenge

You are provided with data from six different subjects, each with a train and test partition. The data in X_train and X_test contain EEG signals collected from the scalp of each subject on several trials, while y_train informs the class of each trial, either left-hand or right-hand.

Your objective: build a predictive model in Python that learns the relationship between EEG signals and motor imagery task for each trial, and accurately predicts the trials in the test set (i.e. y_test).

### Why Does This Matter?

Motor imagery BCIs have direct applications in assistive technology and rehabilitation. They allow people with severe motor disabilities - such as those caused by stroke, ALS, or spinal cord injury - to control external devices (robotic arms, wheelchairs, communication boards) using only their thoughts. Beyond assistive use, MI-BCIs are also studied in the context of motor rehabilitation, where neurofeedback can help stroke patients retrain their motor cortex by reinforcing the neural patterns associated with movement imagination.

Accurate classification of motor imagery is thus a core problem in clinical and applied neuroscience. Improving decoding accuracy directly translates to more reliable and usable systems for real patients.

### Suggested Approaches

This is fundamentally a classification problem. You are encouraged to explore various statistical and machine learning methods, such as:

- Band-power features + linear classifiers: extract log-variance or power spectral density (PSD) in the Mu (8-12 Hz) and Beta (13-30 Hz) bands per electrode, then apply Linear Discriminant Analysis (LDA) or logistic regression.
- Common Spatial Patterns (CSP): a classical BCI method that learns spatial filters maximizing the variance ratio between two classes; the filtered signal power serves as features for a downstream classifier.
- Riemannian geometry: represent each trial as a covariance matrix and classify directly in the space of symmetric positive-definite matrices using Riemannian distance-based methods (e.g., Minimum Distance to Mean, MDM).
- Regularized covariance + SVM: estimate shrinkage covariance matrices per trial and feed them (vectorized) into a kernel SVM.
- Deep learning: convolutional neural networks (e.g., EEGNet, ShallowConvNet) that learn spatial and temporal filters end-to-end from raw or filtered EEG.

Note that EEG data is highly subject-specific. Simple but well-tuned per-subject pipelines often outperform complex models that do not account for inter-subject variability.
