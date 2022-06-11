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



## Motivation & Objective: What are you trying to do and why? 

With the development of Internet and Internet of Things, large amounts of data is produced every moment with high speed and the processing and utilization of serial data has become an important and hot topic. 

Due to the nature of high speed networks and sensors, errors widely occur during the data collection and transference process and result in missingness of data. In this case, the integrity of data is damaged and brings negative effect to the downstream tasks. In some other cases, the missingness of data is natural and is due to uneven sampling (i.e., the interval between sampling varies and introduces missingness at certain times).
In both of the two scenarios, imputation of missing data is believed to have positive influence on the data processing and analysis.

During the pursuing of more accurate data imputation methods, learning based methods have become the mainstream. However, such methods also come with shortcomings like requirement of high volume data and high training cost and latency.
In fact, the shortage of training data is an especially critical challenge, as it is impossible for us to know the true values of the values which are already missing, and in some cases, such true values never exist. 
Also, we may face the so called **cold start** issue, which requires us to start inference and imputation immediately without available training data. 
**Curse of dimension** points out that irrelevant and low-quality training data would impair the model's performance while increasing the overhead of training.
Thus, it is also important to adopt training data with high-quality and relevance.

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
