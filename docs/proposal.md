# Project Proposal

## 1. Motivation & Objective

Time series are ubiquitous in real-world applications. Unfortunately, many unexpected accidents will cause missing values of data, such as irregular observations, software crash, communication outage, energy availability, power management, privacy and other human factors and so on. However, missing data often provide rich information and some missing rate can reach 90% in some datasets, which makes the data difficult to be utilized and exploited. This also hurts the downstream applications such as traditional classification or regression, sequential data integration and forecasting tasks, thus raising the demand for data imputation. 

In this project, we will compare the performance of different typical methods under different conditions of missing values using different types of datasets, also we will try to apply the self-attention mechanism to this problem and compare the performance to the traditional methods and other deep learning methods such as RNN, GRU and GAN.

## 2. State of the Art & Its Limitations

At first, we want to introduce some traditional methods:

* **Deletion methods** directly omit the missing data and perform analysis only on the observed data. However, this will not cause a good performance if the missing rate is high and inadequate samples are kept, which will also make the data incomplete and not suitable for downstream applications.
* **Neighbor based methods** impute the missing value from neighbors by clustering methods like KNN or DBSCAN. They first find the nearest neighbors of the missing values through other attributes and the update the missing values with the mean value of these neighbors.
* **Constraint based methods** discover the rules in dataset and take advantages of these rules to impute. These methods work when the data is highly continuous or satisfies certain patterns. However, multivariable time series in the real world are not usually satisfied with such rules.
* **Regression based methods** learn a regression model for predicting the missing values based on nearest neighbors and historical data, which rely a lot on the relativity and stability of the time series.
* **Statistical based methods** use statistical models such as simply taking the mean or median values to impute missing data. These methods relies on the whole dataset including both the historical and future data which is different from regression based methods.
* **Matrix-Factorization baed methods** try to apply the Matrix Factorization (MF) algorithm to impute the missing values with MF and reconstruction to find the correlations among the data.
* **Expectation-Maximization based methods** follow a two-stage strategy consisting of the E step and the M step which iteratively imputes the misssing values with the statistical model parameters and then updates the statistical model parameters to maximize the possibility of the distribution of the filled data.
* **Multi-Layer Perceptron based methods** use fully connected networks to predict the missing values by minimizaing the loss function.

The above traditional methods rarely take the temporal relations among the observations and treat the time series as normal structured data, thus losing the information from the time data. So far, deep learning based methods have been applied to multivatiable time series imputation and show positive progress in imputing the missing data. Most of them adopt or combine the idea of RNN, GRU and GAN:

* **GRU-D<sup>[1]</sup> (GRU)** is proposed as one of the early attempts to impute time series with deep learning models. It is also the first research to exploit that, RNN can model multivariable time series with the informativeness from the time series since former works attempted to impute missing values with RNN by concatenating timestamps and raw data, which means they regard timestamps as one attribute of raw data. It first the concept of time lag. It also adopts the gated recurrent unit to generate missing values called GRU-D and proposes the concept of decay rate.
* **M-RNN<sup>[2]</sup> & BRITS<sup>[3]</sup> (Bidirectional RNN)** both impute missing values according to hidden states from bidirectional RNN. However, M-RNN treats missing values as constants, while BRITS treats missing values as variables of the RNN graph. Furthermore, BRITS takes correlations among feathers into consideration while M-RNN doesn’t.
* **GRU-I<sup>[4]</sup> (GRU+GAN)** follows the structure of GRU-D with the removal of the input decay. So, there is no innovation in the RNN part as well as the decay rate. The GAN structure is made up of a generator (G) and a discriminator (D). Both G and D are based on GRU-I, and it takes lots of time to train the model to get the data imputed. However, this model is not practical since the accuracy of the generative model seems not stable with a random noise input. And it also makes the model hard to converge.
* **E<sup>2</sup>GAN<sup>[5]</sup> (GRU+GAN, Auto-Encoder Enhanced)** adopts an auto-encoder structure based on GRU-I to form the generator instead of taking a random noise vector as inputs like GRU-I, though this tackles the difficulty of training the model.
* **NAOMI<sup>[6]</sup> (RNN+GAN, Bidirectional Enhanced)** proposes a non-autoregressive model which conditions both previous values but also future values just like BRITS. However, in NAOMI, time gaps are ignored, and the data is injected into the RNN model without timestamps. It suggests the model is not aware of irregular time series although we can still take them as input by removing their timestamps directly.

Finally we introduce some self-attention based methods:

* **CDSA<sup>[8]</sup>** applies cross-dimensional self-attention jointly from three dimensions (time, location, and measurement) to impute missing values in spatiotemporal datasets. While it is specifically designed for spatiotemporal data rather than general time series.
* **NRTSI<sup>[9]</sup>** treats time series as a set of tuples (time, data). Such a design makes NRTSI applicable to irregularly sampled time series. However, its algorithm design consists of two nested loops, which weaken the advantage of self-attention that is parallelly computational and make the model process slowly.

## 3. Novelty & Rationale

What is new in your approach and why do you think it will be successful?

## 4. Potential Impact

If the project is successful, what difference will it make, both technically and broadly?

## 5. Challenges

Whether the dataset can be processed correctly, whether those previous classical models can be implemented correctly, and whether the performance under the same dataset will be significantly different from the past. Whether the design of the different missing patterns is reasonable and whether a reasonable and meaningful downstream task can be found for each different dataset to evaluate the data imputation performance.

## 6. Requirements for Success

First, we have to classify the datasets we will use, the patterns of missing data we will realize and the downstream tasks we will evaluate, which means we need to be able to read papers, collect information and work together well. Then we have to properly implement the different methods and compare their performance in a comprehensive and fair manner, which means we need to have good task design and programming skills.

## 7. Metrics of Success

Use classification/prediction to evaluate the models. The main metric is AUC score.

## 8. Execution Plan

The key task is specified in the 'Requirements for Success' part, we need to compare the performance of different typical methods under different conditions of missing values using different types of datasets, also we will try to apply the self-attention mechanism to this problem and compare the performance to the traditional methods and other deep learning methods such as RNN, GRU and GAN. 

In the preparation phase, the three of us divided the work to complete the dataset, the missing data patterns and the downstream tasks respectively. We then plan to pre-process the dataset, followed by several experiments, such as the performance impact of applying different patterns of missing data, the difference between the naturally missing dataset and the non-missing dataset after missing processing, and a comparison of the performance of each model in different downstream tasks.

## 9. Related Work


### 9.a. Papers

* **GRU-D<sup>[1]</sup>:** "Recurrent neural networks for multivariate time series with missing values."
* **M-RNN<sup>[2]</sup>:** "Estimating missing data in temporal data streams using multi-directional recurrent neural networks."
* **BRITS<sup>[3]</sup>:** "Brits: Bidirectional recurrent imputation for time series."
* **GRU-I<sup>[4]</sup>:** "Multivariate time series imputation with generative adversarial networks."
* **E<sup>2</sup>GAN<sup>[5]</sup>:** "E2gan: End-to-end generative adversarial network for multivariate time series imputation."
* **NAOMI<sup>[6]</sup>:** "NAOMI: Non-autoregressive multiresolution sequence imputation."

### 9.b. Datasets

We have listed all used datasets from the above papers and picked typical three of them:

* **PhysioNet Challenge 2012 dataset (PhysioNet)<sup>[13]</sup>:** The PhysioNet 2012 challenge dataset contains 12,000 multivariate clinical time-series samples collected from patients in ICU (Intensive Care Unit). Each sample is recorded during the first 48 hours after admission to the ICU. Depending on the status of patients, there are up to 37 time-series variables measured, for instance, temperature, heart rate, blood pressure. Measurements might be collected at regular intervals (hourly or daily), and also may be recorded at irregular intervals (only collected as required). Not all variables are available in all samples. Note that this dataset is very sparse and has 80% missing values in total. 
* **Beijing Multi-Site Air Quality<sup>[14]</sup>:** This air-quality dataset [54] includes hourly air pollutants data from 12 monitoring sites in Beijing. Data is collected from 2013/03/01 to 2017/02/28 (48 months in total). For each monitoring site, there are 11 continuous time series variables measured (e.g. PM2.5, PM10, SO2). We aggregate variables from 12 sites together so this dataset has 132 features. There are a total of 1.6% missing values in this dataset.
* **Electricity Load Diagrams<sup>[15]</sup>:** This is another widely-used public dataset from UCI [55]. It contains electricity consumption data (in kWh) collected from 370 clients every 15 minutes and has no missing data. The period of this dataset is from 2011/01/01 to 2014/12/31 (48 months in total). Similar to processing Air-Quality,

### 9.c. Software

* **Python**

## 10. References

List references correspondign to citations in your text above. For papers please include full citation and URL. For datasets and software include name and URL.

[1] Che, Zhengping, et al. "Recurrent neural networks for multivariate time series with missing values." Scientific reports 8.1 (2018): 1-12.

[2] Yoon, Jinsung, William R. Zame, and Mihaela van der Schaar. "Estimating missing data in temporal data streams using multi-directional recurrent neural networks." IEEE Transactions on Biomedical Engineering 66.5 (2018): 1477-1490.

[3] Cao, Wei, et al. "Brits: Bidirectional recurrent imputation for time series." Advances in neural information processing systems 31 (2018).

[4] Luo, Yonghong, et al. "Multivariate time series imputation with generative adversarial networks." Advances in neural information processing systems 31 (2018).

[5] Luo, Yonghong, et al. "E2gan: End-to-end generative adversarial network for multivariate time series imputation." Proceedings of the 28th international joint conference on artificial intelligence. AAAI Press, 2019.

[6] Liu, Yukai, et al. "NAOMI: Non-autoregressive multiresolution sequence imputation." Advances in neural information processing systems 32 (2019).

[7] Du, Wenjie, David Côté, and Yan Liu. "SAITS: Self-Attention-based Imputation for Time Series." arXiv preprint arXiv:2202.08516 (2022).

[8] Ma, Jiawei, et al. "CDSA: cross-dimensional self-attention for multivariate, geo-tagged time series imputation." arXiv preprint arXiv:1905.09904 (2019).

[9] Shan, Siyuan, Yang Li, and Junier B. Oliva. "NRTSI: Non-Recurrent Time Series Imputation." arXiv preprint arXiv:2102.03340 (2021).

[10] Lipton, Zachary C., David C. Kale, and Randall Wetzel. "Modeling missing data in clinical time series with rnns." Machine Learning for Healthcare 56 (2016).

[11] Du, Wenjie, David Côté, and Yan Liu. "SAITS: Self-Attention-based Imputation for Time Series." arXiv preprint arXiv:2202.08516 (2022).

[12] Zerveas, George, et al. "A transformer-based framework for multivariate time series representation learning." Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2021.

[13] A. Goldberger, L. Amaral, L. Glass, Jeffrey M. Hausdorff, P. Ivanov, R. Mark, J. Mietus, G. Moody, C. Peng, and H. Stanley. Physiobank, physiotoolkit, and physionet: components of a new research resource for complex physiologic signals. Circulation, 101 23:E215–20, 2000.

[14] Shuyi Zhang, Bin Guo, Anlan Dong, Jing He, Ziping Xu, and S. Chen. Cautionary tales on air-quality improvement in beijing. Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 473, 2017.

[15] Dheeru Dua and Casey Graff. UCI machine learning repository, 2017.


