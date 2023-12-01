# Machine Learning Problem Sets

In this practical session, you are going to use Linear Discriminant Analysis and Quadratic Discriminant Analysis in the context of __cervical cancer detection__, and further perform anomaly detection on a chemical process

## Problem 1: Cervical Cancer Detection

### Problem Description

The original data is available [in this link](https://mde-lab.aegean.gr/index.php/downloads/). This benchmark consists of pap smear images, illustrating cells. Each cell was then labeled by a physician into one of 7 categories,

| Category | Cell Type                                    | Label (binary) | Cell Count | Subtotal |
|----------|----------------------------------------------|----------------|------------|----------|
| Normal   | Superficial Squamous Epithelial              | 1 (0)          | 74         |          |
| Normal   | Intermediate Squamous Epithelial             | 2 (0)          | 70         |          |
| Normal   | Columnar Epithelial                          | 3 (0)          | 98         | 242      |
| Abnormal | Mild Squamous non-keratinizing dysplasia     | 4 (1)          | 182        |          |
| Abnormal | Moderate Squamous non-keratinizing dysplasia | 5 (1)          | 146        |          |
| Abnormal | Severe Squamous non-keratinizing dysplasia   | 6 (1)          | 197        |          |
| Abnormal | Squamous Cell carcinoma in situ intermediate | 7 (1)          | 150        | 675      |

as a result, there are two problems:

- Binary classification, in which one wants to classify whether an image is normal or abnormal
- Multi-Class classification, in which one wants to classify the cell type.

The dataset is discribed in various publications, among which,

> Jantzen, J., Norup, J., Dounias, G., & Bjerregaard, B. (2005). Pap-smear benchmark data for pattern classification. Nature inspired smart information systems (NiSIS 2005), 1-9.

In this problem set, you will use pre-extracted texture features based on [Local Binary Patterns](https://en.wikipedia.org/wiki/Local_binary_patterns). Here's an example of the extracted features:

![](FeatureExtraction.png)

- On the 1st row, a sample of the original is shown.
- On the 2nd row, the texture features are shown.
- On the 3rd row, a histogram over these features is computed. The final dataset is composed of these histograms.

__Note.__ In this problem, you have two classification problems. One with labels $\set{1, \cdots, 7}$, and a second one with labels $\set{0, 1}$ (cancerous or not). In the dataframe, you have the multi-class classification labels. As a consequence, you need to create 2 $y$ variables (with multi-class and binary labels), so as to test LDA and QDA on both problems.

### Solution Description

In this part of the problem set, you will program 2 classes,

- ```LinearDiscriminantAnalaysis```, which will implement the LDA method seen in class.
- ```QuadaraticDiscriminantAnalaysis```, which will implement the QDA method seen in class.

Your classes must implement three methods,

1. ```__init__(self, reg)```, where $reg$ is the regularization parameter described in class.
2. ```fit(self, X, y)```, which learns the LDA/QDA models from data
3. ```predict(self, x)```, which takes a matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$ and outputs a vector of predictions $\mathbf{y} \in \set{0,\cdots,C}$.
4. ```predict_proba(self, x)``` which has the same behavior of the last method, except that it predicts the probability of each instance to belong to a given class. In this sense, it returns a matrix of probabilities $\mathbf{P} \in \mathbb{R}^{n \times C}$.

### Conceptual Questions

1. Test your models on the binary and multi-class problems. Which performances do you get?
2. Compare your results with a simple Nearest Neighbors classifier (take the code from previous lectures) under the Euclidean and cosine metrics. How well do the methods perform?
3. Compute a matrix $\mathbf{M} \in \mathbb{R}^{C \times C}$ in which $M_{ij}$ corresponds to the __number of samples from class $i$__ predicted as __a sample from class $j$__. This matrix is known as __confusion matrix__. Based on it, which classes the classifiers fails the most? With which other class does it confuses the samples?
