# Project Proposal

## 1. Motivation & Objective

Time series are ubiquitous in real-world applications. Unfortunately, many unexpected accidents will cause missing values of data, such as irregular observations, software crash, communication outage, energy availability, power management, privacy and other human factors and so on. However, missing data often provide rich information and some missing rate can reach 90% in some datasets, which makes the data difficult to be utilized and exploited. This also hurts the downstream applications such as traditional classification or regression, sequential data integration and forecasting tasks, thus raising the demand for data imputation. 

In this project, we will try to apply the self-attention mechanism to this problem and compare the performance to the traditional methods and other deep learning methods such as RNN, GRU and GAN.

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

* **GRU-D** is proposed as one of the early attempts to impute time series with deep learning models. It is also the first research to exploit that, RNN can model multivariable time series with the informativeness from the time series since former works attempted to impute missing values with RNN by concatenating timestamps and raw data, which means they regard timestamps as one attribute of raw data. It first the concept of time lag. It also adopts the gated recurrent unit to generate missing values called GRU-D and proposes the concept of decay rate.
* **M-RNN & BRITS** both impute missing values according to hidden states from bidirectional RNN. However, M-RNN treats missing values as constants, while BRITS treats missing values as variables of the RNN graph. Furthermore, BRITS takes correlations among feathers into consideration while M-RNN doesn’t.
* **GRU-I** follows the structure of GRU-D with the removal of the input decay. So, there is no innovation in the RNN part as well as the decay rate. The GAN structure is made up of a generator (G) and a discriminator (D). Both G and D are based on GRU-I, and it takes lots of time to train the model to get the data imputed. However, this model is not practical since the accuracy of the generative model seems not stable with a random noise input. And it also makes the model hard to converge.
* **E<sup>2</sup>GAN** adopts an auto-encoder structure based on GRU-I to form the generator instead of taking a random noise vector as inputs like GRU-I, though this tackles the difficulty of training the model.


## 3. Novelty & Rationale

What is new in your approach and why do you think it will be successful?

## 4. Potential Impact

If the project is successful, what difference will it make, both technically and broadly?

## 5. Challenges

What are the challenges and risks?

## 6. Requirements for Success

What skills and resources are necessary to perform the project?

## 7. Metrics of Success

What are metrics by which you would check for success?

## 8. Execution Plan

Describe the key tasks in executing your project, and in case of team project describe how will you partition the tasks.

## 9. Related Work

### 9.a. Papers

List the key papers that you have identified relating to your project idea, and describe how they related to your project. Provide references (with full citation in the References section below).

### 9.b. Datasets

List datasets that you have identified and plan to use. Provide references (with full citation in the References section below).

### 9.c. Software

List softwate that you have identified and plan to use. Provide references (with full citation in the References section below).

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


