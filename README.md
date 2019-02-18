
## Designing a recommender system and predicting customer sentiment using data science techniques

### Summary

Within this e-commerce data science project, there are 3 sub-projects:
1. predicting missing rating for a product based on customer's past purchases and past ratings for other products (Amazon's Clothing, Shoes and Jewelry dataset)
2. come up with product recommendater system for customers based on their purchase history and product reviews (Amazon's home and kitchen dataset)
3. perform sentiment analysis on any given product for a given customer (Amazon's home and kitchen dataset)

I used Amazon DataSet from http://snap.stanford.edu/data/web-Amazon.html. All the three sub-projects have separate jupiter notebooks (written in python).

### Project_1: PREDICT MISSING RATING (for a product using FunkSVD & ALS models)

#### Summary:

I mainly focused on building latent factor models such as FunkSVD and ALS. Latent factor models are an alternative approach that tries to explain the ratings by characterizing both items and users on, say, 20 to 100 factors inferred from the ratings patterns. In a sense, such factors comprise a computerized alternative to the aforementioned human created song genes. For, Clothing, shoes, and Jewelry, the discovered factors might measure obvious dimensions such as color, design and brand. For users, each factor measures how much the user likes products that score high on the corresponding product factor. Some of the most successful realizations of latent factor models are based on matrix factorization. In its basic form, matrix factorization characterizes both items and users by vectors of factors inferred from item rating patterns. High correspondence between item and user factors leads to a recommendation. Of course, matrix factorization is simply a mathematical tool for playing around with matrices, and is therefore applicable in many scenarios where on would like to find out something hidden under the data. These methods have become popular in recent years by combining good scalability with predictive accuracy. In addition, they offer much flexibility for modeling various real-life situations.

#### Funk SVD Model Framework:

There are lot of missing rating values in user's ratings dataset. One of the advantages of FunkSVD model is we do not have to fill the missing ratings with mean/zero values even though it bacame very dense matrix. After that I have done matrix factorization along with regularization to come up with meaningful latent factors. 

Placeholer for minimizing equation

<todo: paste equation image here>

#### ALS Model Framework:

ALS model decomposes ratings matrix into two matrices eg: P and Q. ALS rechnique rotate between fixing the Q's and fixing the P's. When all P's are fixed, the system rcomputes the Q's by solving a least-squares problem, and vice versa. This ensures that each step decreases the cost function equation until convergence. While SGD is easier and faster than ALS, ALS has at least two favorable cases. The first is when the system computes each Q independently of the other item factors and computes each P independently of the other user factors. This gives rise to potentially massive parallalization of the algorithm. The second case is for systems centered in implicit data. Because the training set cannot be considered sparse, looping over each single training case as gradient descent does would not be practical. ALS can efficiently handle such cases. 

### Project_2: COLLABORATIVE FILTERING RECOMMENDER SYSTEM (K - nearest neighbors algorithm)

Goal: Build an item-based collaborative filtering system based on K - nearest neighbors to find the three most similar products. I used Amazon Home and Kitchen reviews dataset from http://snap.stanford.edu/data/web-Amazon.html. Also, build K neighbors classifier model to predict overall review rating based on text reviews. 

### Project_3: SENTIMENT ANALYSIS (for a given product and a given customer)

#### Goals:

    1. Perform sentiment analysis using the below techniques:

        a. Logistic regression with TFIDF vectorizer
        b. Logistic regression with TFIDF vectorizer and n-grams
        c. SVM classifier with TFIDF vectorizer and n-grams
        d. Naive Bayes with TFIDF vectorizer and n-grams

I noticed that sklearn's svm.SVC() classifier is extremely slow. 
    Support Vector Machines are powerful tools, but their compute and storage requirements increase rapidly with the number of training vectors. The core of an SVM is quadratic programming problem (QP), separating support vectors from the rest of the training data. Also, note that for the linear case, the algorithm used in LinearSVC by the liblinear implementation is much more efficient than its libsvm-based SVC counterpart and can scale almost linearly to millions of samples and/or features. SVM - training with nonlinear-kernels, which is default in sklearn's SVC, is complexity-wise approximately: 0(n_samples^2 * n_features). This applies to to the SMO-algorithm used within libsvm, which is the core-solver in sklearn for this type of problem. This changes much when no kernels are used and one uses sklearn.svm.LinearSVC (based on liblinear) or sklearn.linear_model.SGDClassifier.


### Algorithms used in this project

#### Predict missing rating
* FunkSVD
* ALS

#### Recommender System
* K - Nearest Neighbors (Collaborative filtering recommender system)

#### Sentiment Analysis
* Logistic Regression
* SVM
* Naive Bayes

### References

* Matrix Factorization: http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/
* Recommendation Systems: http://infolab.stanford.edu/~ullman/mmds/ch9.pdf
* http://snap.stanford.edu/data/web-Amazon.html
* Text Processing: https://www.analyticsvidhya.com/blog/2018/02/the-different-methods-deal-text-data-predictive-python/