
# Lab 3: Machine Learning Model Training and Evaluation

This repository contains the code and data for training and evaluating machine learning models, focusing on data preparation, feature extraction, and model training.

## Repository Contents

- **`data_prep.py`**: Script for preparing and preprocessing the dataset.
- **`lab3.py`**: Main script for training and evaluating the machine learning models.
- **`features.txt`**: File listing the features used in the models.
- **`train.dat`**: Training dataset.
- **`test.dat`**: Testing dataset.
- **`real_test.dat`**: Additional dataset for real-world testing.
- **`best.model`**: Serialized file of the trained model with the best performance.
- **`wiki_text.txt`**: Supplementary text data, possibly for natural language processing tasks.
- **`xor.model`**: Serialized file of a model trained on XOR data.
- **`xorFeatures.txt`**: Features used for the XOR model.
- **`xorLabel.dat`**: Labels corresponding to the XOR dataset.
- **`xorNoLabel.dat`**: XOR dataset without labels.

## Prerequisites

Ensure you have the following installed:

- Python 3.x
- Required Python packages (listed in `requirements.txt` if available)

## Usage

1. **Data Preparation**:
   - Run `data_prep.py` to preprocess the datasets. This script will clean the data and extract necessary features.

   ```bash
   python data_prep.py
   ```

2. **Model Training and Evaluation**:
   - Execute `lab3.py` to train the machine learning models using the prepared data.

   ```bash
   python lab3.py
   ```

   - The script will output performance metrics and save the trained model as `best.model`.

3. **Testing**:
   - Use the `test.dat` dataset to evaluate the model's performance.
   - For real-world testing scenarios, utilize `real_test.dat`.

## Notes

- The `xor.model`, `xorFeatures.txt`, `xorLabel.dat`, and `xorNoLabel.dat` files pertain to experiments with the XOR problem, which is a classic test for machine learning algorithms.
- Ensure that the datasets (`train.dat`, `test.dat`, `real_test.dat`) are in the correct format expected by the scripts.

## Contributing

Contributions are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
