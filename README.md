
# Lab 3: Machine Learning Model Training and Evaluation

This repo contains both a adaboost implementation as well as a decision tree learning model. 
Data preprocessing is given as well. These models are made to work with text and text substrings.

Each example should be a string formatted like so "label | value"
The lab3.py file is able to parse the string and turn each string into an example instance.

## Repository Contents

- **`data_prep.py`**: Script for preparing and preprocessing the dataset.
- **`lab3.py`**: Main script for training and evaluating the machine learning models.
- **`features.txt`**: File listing the features used in the models.
- **`train.dat`**: Training dataset.
- **`test.dat`**: Testing dataset.
- **`real_test.dat`**: Additional dataset that provides explicetly unlabeled data.
- **`best.model`**: Serialized file of the trained model with the best performance.
- **`wiki_text.txt`**: Used when parsing data from wikipedia articles into examples for the training of the model.
- **`xor.model`**: Serialized file of a model trained on XOR data.
- **`xorFeatures.txt`**: Features used for the XOR model.
- **`xorLabel.dat`**: Labels corresponding to the XOR dataset.
- **`xorNoLabel.dat`**: XOR dataset without labels.

## Prerequisites

Ensure you have the following installed:

- Python 3.x

## Usage

1. **Data Preparation**:
   - If you wish to create examples for the model to train on type
     '''
     python3 data_prep.py <label> <text.txt>
     '''
     Into the terminal, "label" is the label you wish to give all examples
     text.txt will be cut into 15 word segments and has the labels applied to the start of every line.

2. **Model Training and Evaluation**:
   - If you wish to train the model type:
     '''
     python3 lab3.py train <examples> <features> <hypothesisOut> <learning-type>
     '''
     Where <examples> is a file with preprocesses examples (read above to learn how to run file),
     <features> is a txt file with feature you want to the model to train on seperated by newlines.
     **Note: Each feature will be searched through each example as a substring**
     <hypothesisOut> is the filepath that you want to write your pre-train modeled to be encoded into for later use.
     **Note: Be careful as this can overwrite previous models if not checked beforehand**
     <learning-type> is the type of model you want to use for your predictions, the only 2 accepted values are "dt" (decision tree) or "ada" (adaboost algorithm)

3. **Testing**:
   - If you wish to use your trained model to make predicitions on text use it as so:
     '''
     python3 lab3.py predict <examples> <features> <hypothesis>
     '''
     <examples> is your unlabelled data, 15 word strings
     <features> and un-needed arguement but was in the write-up for the school assignment. You can pass any file here as it will not be read
     <hypothesis> must be a valid file path to a encoded model from the lab3.py file.

## Notes

- The `xor.model`, `xorFeatures.txt`, `xorLabel.dat`, and `xorNoLabel.dat` files pertain to experiments with the XOR problem, which is a classic test for machine learning algorithms.
- Ensure that the datasets (`train.dat`, `test.dat`, `real_test.dat`) are in the correct format expected by the scripts.
