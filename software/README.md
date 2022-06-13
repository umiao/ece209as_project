**Google Drive and Colab**

We use the google drive to store the data and codes and run some .ipynb code files with colab.

[209AS project](https://drive.google.com/drive/folders/1-8JF-hG9DmKItDK5O7lk4-30raSYFWeZ?usp=sharing)

**Imputation and Classification models and codes reference:**

- [SAITS: Du, Wenjie, David Côté, and Yan Liu. "SAITS: Self-Attention-based Imputation for Time Series." arXiv preprint arXiv:2202.08516 (2022).](https://arxiv.org/abs/2202.08516)
- [BRITS: Cao, Wei, et al. "Brits: Bidirectional recurrent imputation for time series." Advances in neural information processing systems 31 (2018).](https://proceedings.neurips.cc/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html)
- [M-RNN: Yoon, Jinsung, William R. Zame, and Mihaela van der Schaar. "Estimating missing data in temporal data streams using multi-directional recurrent neural networks." IEEE Transactions on Biomedical Engineering 66.5 (2018): 1477-1490.](https://ieeexplore.ieee.org/abstract/document/8485748)


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
