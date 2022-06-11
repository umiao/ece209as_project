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
