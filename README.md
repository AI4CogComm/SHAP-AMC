# SHAP-AMC
Ues SHAP, an XAI approach, to find the negative information in attacked wireless signals, which helps improve the modulation classfication accuracy.

# CAMC
Source codes of the article: 

P. Dong,J. Wang,S. Gao,F. Zhou and Q. Wu, "Explainable Deep Learning Based Adversarial Defense for Automatic Modulation Classification," IEEE Internet of Things Journal, to be published. 

Please cite this paper when using the codes.

# Instructions

Reading the below sections in the written order will help better understand all the codes.

 ## Python

**rml128.py**
This code is used for data preprocessing, including converting IQ signals to AP format, normalization,
and other tasks. This code loads the training and validation datasets for training the 
automatic modulation classification (AMC) network, and uses the test dataset to simulate signal
data in real-world transmission scenarios.

**rml128_tiny.py**
This code is used for data preprocessing, including converting IQ signals to AP format, normalization,
and other tasks.
This code loads the training and validation datasets for adversarial fine-tuning of the AMC network.
The labeled test dataset is used for SHAP analysis to identify negative information introduced by adversarial attacks.

**rml128_dtiny.py**
This code is used for data preprocessing, including converting IQ signals to AP format, normalization,
and other tasks.
This code is used to construct the dataset for training the attack detection network.

**mltools.py**
This code is a tool for plotting the confusion matrix of signal data collected from real transmission scenarios.

**mltools_tiny.py**
This code is a tool for plotting the confusion matrix of signal data used for SHAP analysis.

**carlini_wagner_l2.py**
This code implements a modified version of the C&W attack based on the 
cleverhans library, tailored for generating adversarial perturbations on signal data.

**utils.py**
This code provides auxiliary support for the modified C&W attack,
facilitating its application to signal-based inputs.

**model**
This folder contains the pre-trained AMC network model.

**detectmian.py**
This script defines the architecture of the attack detection network,
constructs the adversarial dataset, and trains the attack detection network.

**SHAP_AFT.ipynb**
This code is the core implementation of the SHAP-AFT algorithm. It includes generation of adversarial signal samples for real transmission,
creation of adversarial samples for SHAP analysis and fine-tuning,
adversarial detection via the detection network, SHAP-based negative information mining,
and negative information removal followed by AMC network fine-tuning.
Due to limitations of the SHAP library, the code must be executed within a Jupyter Notebook environment.



# Environment
These models are implemented in Keras, and the environment setting is:

-   Python 3.9.0
-   TensorFlow-gpu 2.7.0
