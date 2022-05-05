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
* **

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
