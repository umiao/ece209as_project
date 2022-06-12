# ece209as_project
This is repository template for UCLA ECE209AS projects.

Use the folders as follows:

* doc/ for website content
* software/ for code used in your project
* data/ for data data used in your project

You may add additional folders as necessary.


**Classification.py Walking Through:**

To do the classification task, you need to train the RNN model then do the classification.
You can find the classification file under software/classification.py

There are several parameters that you might need to change before running the code.

root_dir is the dir where you save your model and log.

original_dataset_path is where you store original data.

imputed_dataset_path is where you store the imputation result.

feature_num is the number of features, you need to adjust according to imputation results.

saved_model_path is the path that you saved yout model.

test_mode is whether or not you are testing.



To train the RNN model, set test_mode to false; To do the classification, set test_mode to True.


