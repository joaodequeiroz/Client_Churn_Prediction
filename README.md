# TopBank Company - A Churn Prediction Project

---

## Table of Contents
- [1. Introduction](#1-introduction)
- [2. The Business Challenge](#2-business-challenge) 
- [3. Project Development](#3-project-development)
- [8. Next Steps](#8-next-steps)
- [9. Lessons Learned](#9-lessons-learned)
- [10. Conclusion](#10-conclusion)
- [References](#references)

# 1. Introduction
## 1.1 TopBank Company

TopBank is a major banking services company. It operates mainly in European countries offering financial products, from bank accounts to investments, also some types of insurance and investment product.
The company's business model is of the service type, that is, it sells banking services to its customers through physical branches and an online portal.
The company's main product is a bank account, in which the customer can deposit his salary, make withdrawals, deposits and transfer to other accounts. This bank account has no cost to the customer and has a 12-month term, that is, the customer needs to renew the contract for this account to continue using it for the next 12 months.
According to TopBank's Analytics team, each customer who has this bank account returns a monetary value of 15% of their estimated salary, if this is less than the average, and 20% if this salary is greater than the average, during the current period of your account. This amount is calculated annually.

## 1.2 Churn

Generally speaking, Churn is a metric that indicates the number of customers who canceled their contract or stopped buying your product in a given period of time. For example, customers who canceled the service contract or after its expiration, did not renew, are considered churn customers.

Source: [https://sejaumdatascientist.com/predicao-de-churn/](https://sejaumdatascientist.com/predicao-de-churn/)

## 1.3 Project Development Methodology

The project was developed based on the CRISP-DS (Cross-Industry Standard Process - Data Science, a.k.a. CRISP-DM) project management method, with the following steps:

**Step 01. Data Description**

**Step 02. Feature Engineering**

**Step 03. Data Filtering**

**Step 04. Exploratory Data Analysis**

**Step 05. Data Preparation**

**Step 06. Feature Selection**

**Step 07. Machine Learning Modelling**

**Step 08. Hyperparameter Fine Tunning**

**Step 09. Convert Model Performance to Business Values**

**Step 10. Deploy Model to Production**

# 2. Business Challenge

## 2.1 Problem

In recent months, the Analytics team of TopBank realized that the rate of customers cancelling their accounts and leaving the bank, the **churn** rate, reached unprecedented numbers in the company. Concerned about the increase in this rate, the team planned an action plan to reduce the customer evasion rate.

## 2.2 Goal

Create an action plan, with the objective of reducing customer evasion, that is, preventing the customer from canceling their contract and not renew it for another 12 months.

## 2.3 Deliverables

1. Deliver to TopBottom's CEO a model in production, which will receive a customer base via API and will return this same “scored” base, that is, one more column with the probability that each customer will go into churn.
2. Model's performance and results report with the following topics:
    - What's the company's current churn rate?
    - How the churn rate varies per month?
    - What's the model's performance in labelling the clients as churns?
    - What's the company's revenue, if the company avoids the customers to get into churn through the developed model?
3. Possible measure: discount coupon or other financial incentive.
    - Which customers could receive an incentive and at what cost, in order to maximize the ROI (Return on investment)? - The sum of incentives shall not exceed $ 10,000.00.
    
# 3. Project Development

## 3.1 STEP 01 - Data Description

### 3.1.1 Dataset

The Dataset is available on Kaggle: [https://www.kaggle.com/mervetorkan/churndataset](https://www.kaggle.com/mervetorkan/churndataset)

### 3.1.2 Data Dimension

![](img/data_types.png)

Number of rows: 10,000

Number of columns: 14

Missing values: none

Types of data: float64 (2), int64 (9) and object (3).

### 3.1.3 Features Description

**RowNumber** - The number of the row.

**CustomerID** - Customer's unique identifier.

**Surname** - Customer's surname.

**CreditScore** - Customer's credit score for the consumer market.

**Geography** - The country where the customer lives.

**Gender** - Customer's gender.

**Age** - Customer's age.

**Tenure** - Number of years that the customer was active.

**Balance** - The amount that the customer has in the bank account.

**NumOfProducts** - The number of products bought by the customer.

**HasCrCard** - Flag that indicates if the customer has a credit card.

**IsActiveMember** - Flag that indicates if the customer has done a bank activity in the last 12 months.

**EstimateSalary** - Estimate customer's monthly income.

**Exited** - Flag that indicates if the customer is in Churn.

The dataset was split into train and test sets with a ratio of 80/20. The split was made in the beginning of the project, before any data manipulation / transformation.

### 3.1.4 Descriptive Statistic

![](img/data_stats.png)

### 3.1.4.1 Numerical Attributes

**1. Credit Score**

- Credit score ranges from 350 up to 850. Mean = 652.5288 and Median = 652.

**2. Age**

- Age ranges from 18 up to 92. Mean = 38.9218. Median = 37.

**3. Tenure** 

- Tenure ranges from 0 to 10. Mean = 5.0128. Median = 5. According the histogram, the number of customers with tenure equal to zero and 10 is lower than the other tenure values.

**4. Balance**

- Balance ranges from zero up to 250,898.09**.** Mean = 76,485.88929. Median = 97,198.54000.

**5. Number of Products**

- Number of products ranges from 1 to 4. Mean = 1.5302. Median = 1.

**6. Has Credit Card**

- if the value is 0, the customer has no credit card and if the value is 1, the customer has credit card.
- Mean = 0.7055, that means that 70.55% of the customers have credit card.

**7. Is Active Member**

- if the value is 0, the customer is not active and if the value is 1, the customer is active.
- Mean = 0.5151, that means that 51.51% of the customers are active members and have used a bank service in the past 12 months.

**8. Estimated Salary**

- Estimated salary ranges from 11.58 up to 199,992.48. Mean = 100,090.23988. Median = 100,193.915.

**9. Exited** 

- if the value is 0, the customer is not in churn and if the value is 1, the customer is in churn.
- Mean = 0.2037, that means that **churn rate is 20.37%**

### 3.1.4.2 Categorical Attributes

**9. Surname**

- There are 2,932 different surnames and the most common is 'Smith' with 32 appearances.

**10. Geography**

- More than 50% of the customers come from France, the rest is almost equally divided by Germany and Spain.

**11. Gender**

- 54.57% of the customers are male and 45.43% are female.

## 3.2 STEP 2 - Feature Engineering

### 3.2.1 Mind Map
![](img/Client_Churn_Prediction.png)

