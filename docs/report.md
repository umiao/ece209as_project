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



## Motivation & Objective: What are you trying to do and why? (plain English without jargon)

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



## State of the Art & Its Limitations: How is it done today, and what are the limits of current practice?

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

## Novelty & Rationale: What is new in your approach and why do you think it will be successful?
## Potential Impact: If the project is successful, what difference will it make, both technically and broadly?
## Challenges: What are the challenges and risks?
## Requirements for Success: What skills and resources are necessary to perform the project?
## Metrics of Success: What are metrics by which you would check for success?

# 2. Related Work

# 3. Technical Approach

# 4. Evaluation and Results

# 5. Discussion and Conclusions

# 6. References
