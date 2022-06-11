# Table of Contents
* Abstract
* [Introduction](#1-introduction)
* [Related Work](#2-related-work)
* [Technical Approach](#3-technical-approach)
* [Evaluation and Results](#4-evaluation-and-results)
* [Discussion and Conclusions](#5-discussion-and-conclusions)
* [References](#6-references)

# Abstract

In this project, we developed a self-adaptive serial data imputation framework. We designed and evaluated encoder structures which are cacpable of extracting the pattern of serial data whose distribution is unknown.
At the same time, we maintain a dataset with serial data from multiple domains, extracting each data's feature and assign them into non-overlapping subsets with cluster algorithm.
Whenever we encounter sequence with missing values, we would extract the pattern of the current sequence and compare with the cluster centers to pick the most relevant clusters. 
Then, selected subsets would be merged into one training set for learning of missing data imputation model.
We proved that the inner pattern of known serial data is sufficient of imputation model training, regardless of its actual meaning.
Our experimental results showed that our framework even beated imputation model trained with data having identical distribution as the testing set, without domain knowledge and significantly improved the performance of corresponding downstream task.

# 1. Introduction



## Motivation & Objective:

With the development of Internet and Internet of Things, large amounts of data is produced every moment with high speed and the processing and utilization of serial data has become an important and hot topic. 

Due to the nature of high speed networks and sensors, errors widely occur during the data collection and transference process and result in missingness of data. In this case, the integrity of data is damaged and brings negative effect to the downstream tasks. In some other cases, the missingness of data is natural and is due to uneven sampling (i.e., the interval between sampling varies and introduces missingness at certain times).
In both of the two scenarios, imputation of missing data is believed to have positive influence on the data processing and analysis.

During the pursuing of more accurate data imputation methods, learning based methods have become the mainstream. However, such methods also come with shortcomings like requirement of high volume data and high training cost and latency.
In fact, the shortage of training data is an especially critical challenge, as it is impossible for us to know the true values of the values which are already missing, and in some cases, such true values never exist. 
Also, we may face the so called **cold start** issue, which requires us to start inference and imputation immediately without available training data. 

In order to solve such challenges, we decide to propose a meta-learning framework for serial datat imputation. 
Meta-Learning is also known as learning-to-learning, which means that utilizing former knowledge to solve new tasks and is widely used in Few-shot and Zero-shot learning (machine learning with very few or zero training sample).
In fact, we believe that although the training data for imputation task in specific domain and field can be very rare, we still have plenty of serial data after all.
All we need to do is to overlook the meaning of data and focus on its dynamics and trend.
That is to say, for an incoming serial data whose distribution is unknown, we can still match and extract similar data from other independent datasets and use such data to help the learning of imputation methods.

We believe that in this manner, we would be able to solve the lack of training data and improve the performance of downstream tasks, after imputing the missing values with our meta-learning framework. 



## State of the Art & Its Limitations:

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

* **GRU-D<sup>[1]</sup> (GRU)** is proposed as one of the early attempts to impute time series with deep learning models. It is also the first research to exploit that, RNN can model multivariable time series with the informativeness from the time series since former works attempted to impute missing values with RNN by concatenating timestamps and raw data, which means they regard timestamps as one attribute of raw data. It first proposes the concept of time lag. It also adopts the gated recurrent unit to generate missing values called GRU-D and proposes the concept of decay rate.
* **M-RNN<sup>[2]</sup> & BRITS<sup>[3]</sup> (Bidirectional RNN)** both impute missing values according to hidden states from bidirectional RNN. However, M-RNN treats missing values as constants, while BRITS treats missing values as variables of the RNN graph. Furthermore, BRITS takes correlations among feathers into consideration while M-RNN doesn’t.
* **GRU-I<sup>[4]</sup> (GRU+GAN)** follows the structure of GRU-D with the removal of the input decay. So, there is no innovation in the RNN part as well as the decay rate. The GAN structure is made up of a generator (G) and a discriminator (D). Both G and D are based on GRU-I, and it takes lots of time to train the model to get the data imputed. However, this model is not practical since the accuracy of the generative model seems not stable with a random noise input. And it also makes the model hard to converge.
* **E<sup>2</sup>GAN<sup>[5]</sup> (GRU+GAN, Auto-Encoder Enhanced)** adopts an auto-encoder structure based on GRU-I to form the generator instead of taking a random noise vector as inputs like GRU-I, though this tackles the difficulty of training the model.
* **NAOMI<sup>[6]</sup> (RNN+GAN, Bidirectional Enhanced)** proposes a non-autoregressive model which conditions both previous values but also future values just like BRITS. However, in NAOMI, time gaps are ignored, and the data is injected into the RNN model without timestamps. It suggests the model is not aware of irregular time series although we can still take them as input by removing their timestamps directly.

Finally we introduce some self-attention based methods:

* **CDSA<sup>[8]</sup>** applies cross-dimensional self-attention jointly from three dimensions (time, location, and measurement) to impute missing values in spatiotemporal datasets. While it is specifically designed for spatiotemporal data rather than general time series.
* **NRTSI<sup>[9]</sup>** treats time series as a set of tuples (time, data). Such a design makes NRTSI applicable to irregularly sampled time series. However, its algorithm design consists of two nested loops, which weaken the advantage of self-attention that is parallelly computational and make the model process slowly.

### Limits

All these above SOTA models we mentioned don't consider the case of the lack of training data. For example, PhysioNet has a total of 80% missing values and is very sparse which means it's hard for us to find a true ground truth to train our imputation models. This will cause the bad performance of classification using the imputed data. All these methods need a plenty of training data and can't use other datasets to help.

## Novelty & Rationale:


## Potential Impact:

The most important impact is that we proposed an idea to solve the problem of lack of training data if the original data has too many missing values. We can provide a potential direction of figuring out the cold-start issue.

Then if our idea works, that means we don't need to know the actual meaning of the serial data any more instead we only need to focusing on the dynamic trend of all the data and find the similar as the training data which will change the traditional methods a lot.

Another potential impact lies in the contribution to the open-source community. With this project , we hope to make up the lack of missing data imputation opensource framework and toolkits, which can definitely benefits the related researches.

## Challenges:

Whether the dataset can be processed correctly, whether those previous classical models can be implemented correctly, and whether the performance under the same dataset will be significantly different from the past. Whether the design of the different missing patterns is reasonable and whether a reasonable and meaningful downstream task can be found for each different dataset to evaluate the data imputation performance.

## Requirements for Success:

First, we have to classify the datasets we will use, the patterns of missing data we will realize and the downstream tasks we will evaluate, which means we need to be able to read papers, collect information and work together well. Then we have to properly design an encoding, clustering and voting algorithm to realize our ideas.

## Metrics of Success:

Use classification/prediction to evaluate the models. The main metric is AUC score.

# 2. Related Work

### 2.a. Papers

* **M-RNN<sup>[2]</sup>:** "Estimating missing data in temporal data streams using multi-directional recurrent neural networks."
* **BRITS<sup>[3]</sup>:** "Brits: Bidirectional recurrent imputation for time series."

### 2.b. Datasets

We have listed all used datasets from the above papers and picked typical three of them:

* **PhysioNet Challenge 2012 dataset (PhysioNet)<sup>[11]</sup>:** The PhysioNet 2012 challenge dataset contains 12,000 multivariate clinical time-series samples collected from patients in ICU (Intensive Care Unit). Each sample is recorded during the first 48 hours after admission to the ICU. Depending on the status of patients, there are up to 37 time-series variables measured, for instance, temperature, heart rate, blood pressure. Measurements might be collected at regular intervals (hourly or daily), and also may be recorded at irregular intervals (only collected as required). Not all variables are available in all samples. Note that this dataset is very sparse and has 80% missing values in total. 
* **Beijing Multi-Site Air Quality<sup>[12]</sup>:** This air-quality dataset includes hourly air pollutants data from 12 monitoring sites in Beijing. Data is collected from 2013/03/01 to 2017/02/28 (48 months in total). For each monitoring site, there are 11 continuous time series variables measured (e.g. PM2.5, PM10, SO2). We aggregate variables from 12 sites together so this dataset has 132 features. There are a total of 1.6% missing values in this dataset.
* **Electricity Load Diagrams<sup>[13]</sup>:** This is another widely-used public dataset from UCI [55]. It contains electricity consumption data (in kWh) collected from 370 clients every 15 minutes and has no missing data. The period of this dataset is from 2011/01/01 to 2014/12/31 (48 months in total). Similar to processing Air-Quality,

### 2.c. Software

* **Python**

# 3. Technical Approach

# 4. Evaluation and Results

# 5. Discussion and Conclusions

## Future Work
Aside from the framework and experimental design discussed above, there are also serval interesting topics to be discussed.
1. **Generalization**: In our report, the meta-learning's 'pool' of data only contains two independent datasets, *i.e.*, Beijing Air Quality and Electricity Usage datasets.
The influence of adding more datasets into this 'pool' is to be discovered. Also, more datasets should be applied to test the generalization ability of this framework.

    At the same time, we processed the serial data by dividing it into fixed-length sequence so that encoder models based on CNN, RNN, Transformer and linear model are all applicable.
    The missing data is filled with median of the entire sequence as the encoder requires dense sequence.
    However, if the serial data is very sparse (and intervals between non-missing values are very large), the above assumption would not work.
    In this case, input described by quasi **run-length encoding** may be required.
    
2. **Model Design**:
    For now, the idea of encoder is to change the input sequence into a low-dimensional vector representative which is able to recover the original input sequence.
    Such design only assures the vector representative contains all the information of input, but it may not be the best way to discriminate similar sequences.
    We may design a metric based learning, and maybe we can manually label some of the data and use learning based method to estimate the distance of two sequences (rather than the trival **Euclidean Distance**).    
    With this manner, we can quantize the similarity between sequences, rather than based on a heuristic method (*e.g.*, clustering like KNN).

# 6. References

[1] Che, Zhengping, et al. "Recurrent neural networks for multivariate time series with missing values." Scientific reports 8.1 (2018): 1-12.

[2] Yoon, Jinsung, William R. Zame, and Mihaela van der Schaar. "Estimating missing data in temporal data streams using multi-directional recurrent neural networks." IEEE Transactions on Biomedical Engineering 66.5 (2018): 1477-1490.

[3] Cao, Wei, et al. "Brits: Bidirectional recurrent imputation for time series." Advances in neural information processing systems 31 (2018).

[4] Luo, Yonghong, et al. "Multivariate time series imputation with generative adversarial networks." Advances in neural information processing systems 31 (2018).

[5] Luo, Yonghong, et al. "E2gan: End-to-end generative adversarial network for multivariate time series imputation." Proceedings of the 28th international joint conference on artificial intelligence. AAAI Press, 2019.

[6] Liu, Yukai, et al. "NAOMI: Non-autoregressive multiresolution sequence imputation." Advances in neural information processing systems 32 (2019).

[7] Du, Wenjie, David Côté, and Yan Liu. "SAITS: Self-Attention-based Imputation for Time Series." arXiv preprint arXiv:2202.08516 (2022).

[8] Shan, Siyuan, Yang Li, and Junier B. Oliva. "NRTSI: Non-Recurrent Time Series Imputation." arXiv preprint arXiv:2102.03340 (2021).

[9] Lipton, Zachary C., David C. Kale, and Randall Wetzel. "Modeling missing data in clinical time series with rnns." Machine Learning for Healthcare 56 (2016).

[10] Zerveas, George, et al. "A transformer-based framework for multivariate time series representation learning." Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2021.

[11] A. Goldberger, L. Amaral, L. Glass, Jeffrey M. Hausdorff, P. Ivanov, R. Mark, J. Mietus, G. Moody, C. Peng, and H. Stanley. Physiobank, physiotoolkit, and physionet: components of a new research resource for complex physiologic signals. Circulation, 101 23:E215–20, 2000.

[12] Shuyi Zhang, Bin Guo, Anlan Dong, Jing He, Ziping Xu, and S. Chen. Cautionary tales on air-quality improvement in beijing. Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 473, 2017.

[13] Dheeru Dua and Casey Graff. UCI machine learning repository, 2017.

[14] [Li, Yuebiao, Zhiheng Li, and Li Li]”Missing traffic data: comparison of imputation methods.” IET Intelligent Transport Systems 8.1 (2014): 51-57.

[15] [Ahmed, Mohammed S., and Allen R. Cook]Analysis of freeway traffic time-series data by using Box-Jenkins techniques. No. 722. 1979.

[16] [Ueda, Naonori, et al.] “Split and merge EM algorithm for improving Gaussian mixture density estimates.” Journal of VLSI signal processing systems for signal, image and video technology 26.1 (2000): 133-140.

[17] [Troyanskaya, Olga, et al.] “Missing value estimation methods for DNA microarrays.” Bioinformatics 17.6 (2001): 520-525.

[18] [Kim, Hyunsoo, Gene H. Golub, and Haesun Park.] “Missing value estimation for DNA microarray gene expression data: local least squares imputation.” Bioinformatics 21.2 (2005): 187-198.

[19] [Ni, D., Leonard II, J.D.] “Markov chain Monte Carlo multiple imputation using Bayesian networks for incomplete intelligent transportation systems data”, Transp. Res. Rec., 2005, 1935, (1), pp. 57–67

[20] [Gilks, W.R., Richardson, S., Spiegelhalter, D.J.]”Markov chain Monte Carlo in practice” (Chapman & Hall, London, 1996)

[21] [Tipping, M.E., Bishop, C.M.]”Mixtures of probabilistic principalcomponent analyzers”, Neural Comput., 1999, 11, (2), pp. 443–482

[22] [Lima, Juan-Fernando, Patricia Ortega-Chasi, and Marcos Orellana Cordero]”A novel approach to detect missing values patterns in time series data.” Conference on Information Technologies and Communication of Ecuador. Springer, Cham, 2019.

[23] [Dong, Y., Peng, C.Y.J.]Principled missing data methods for researchers. Springer-Plus 2(1), 222 (2013).
