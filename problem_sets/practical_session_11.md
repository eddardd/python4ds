# Gaussian Mixture Models on MNIST

The goal of this practical session is to exercise linear algebra and statistics with NumPy in a larger scale. Remember, in Lecture 12, we covered GMMs and applied it in a 2D dataset. With real data (high-dimensional and large number of samples), some of the choices we made are not stable, such as explicitly calculating the determinant and the inverse of covariance matrix. This happens, for instance, when the sample covariance matrix is singular.

MNIST is a standard benchmark in machine learning. It contains 60000 samples of 28 x 28 images containing handwritten digits. In this practical session, we are going to use it to illustrate the difficulty of the Expectation Maximization Algorithm in high dimensions.

## Step 1: Loading the dataset

Use the following snippet for loading the data:

```python
import torch
import torchvision
dataset = torchvision.datasets.MNIST(root='./.tmp', train=True, download=True,
                                     transform=torchvision.transforms.Compose([
                                         torchvision.transforms.Resize((16, 16)),
                                         torchvision.transforms.PILToTensor()
                                     ]))
loader = torch.utils.data.DataLoader(dataset, batch_size=100)

i, X, y = 0, [], []

for _x, _y in loader:
    i += 1
    X.append(_x.squeeze())
    y.append(_y)
    if i == 50:
        break
X = torch.cat(X, dim=0).numpy()
y = torch.cat(y, dim=0).numpy()

X = X.astype(np.float64) / 255
```

As is, $\mathbf{X}$ will be an array of shape $(60000, 256)$.

__Note.__ If you're on Google Colab, there's no need to install Pytorch.

## Problem 2: Gaussian Maximum Likelihood Estimation

Assuming that data follows a Gaussian distribution, compute the MLE of the mean and covariance, i.e.,

$$\hat{\mu} = \dfrac{1}{n}\sum_{i=1}^{n}\mathbf{x}_{i}$$

and

$$\hat{\Sigma} = \dfrac{1}{n-1}\sum_{i=1}^{n}(\mathbf{x}_{i}-\mu)(\mathbf{x}_{i}-\mu)^{T}$$

### Conceptual Questions

- Visualize the mean and covariance matrix.

- Try to sample new examples from $\mathcal{N}(\mathbf{x}|\hat{\mu}, \hat{\Sigma})$. Do they resemble the original dataset?

__Hint.__ You might want to use [```numpy.random.multivariate_normal```](https://numpy.org/doc/stable/reference/random/generated/numpy.random.multivariate_normal.html)


## Problem 3: A Naïve GMM

Next, we relax the assumption that the whole dataset follows a Gaussian distribution. Here, we assume that __each digit__ follows a Gaussian distribution, i.e.,

$$P(\mathbf{x}) = \sum_{c=1}^{C}P(Y=c)P(X|Y=c)$$

where $P(X|Y=c) = \mathcal{N}(\mathbf{x}|\hat{\mu}_{c}, \hat{\Sigma}_{c})$, with,

$$\hat{\mu}_{c} = \dfrac{1}{n_{c}}\sum_{i:y_{i}=c}\mathbf{x}_{i}$$

and,

$$\hat{\Sigma}_{c} = \dfrac{1}{n_{c}-1}\sum_{i:y_{i}=c}(\mathbf{x}_{i}-\mu_{c})(\mathbf{x}_{i}-\mu_{c})^{T}$$

and,

$$P(Y=c) = \dfrac{n_{c}}{n}$$

for $n_{c}$, the number of elements in class $c$.

### Problem 3.1. Sampling from a GMM

To sample from a GMM, you should do the following,

- Select a component k, following $P(Y=k)$. Use, for instance, [```numpy.random.choice```](https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html)
- Sample a new example from $P(X|Y=k)$.

### Conceptual Questions

- Compare the means of each $P(X|Y=k)$ with the one you got in problem 2.
- Take a look on the generated samples. Are they better?

## Problem 4: Gaussian Mixture Model

More generally, set $K \in \mathbb{N}$ as the number of components in your GMM. Fit a GMM

$$P(\mathbf{x}) = \sum_{k=1}^{K}P(Z=k)P(X|Z=k)$$

using the code that we used in the last class.

### Practical Considerations

1. Explicitly computing the determinant and inverse will likely fail. Instead, compute $\Sigma = \mathbf{UDU}^{T}$, the eigendecomposition of the Covariance matrix, then use,

$$\log |\Sigma| = \sum_{i=1}^{d}\log \lambda_{i}$$

where $\lambda_{i}$ is the i-th eigenvalue of $\mathbf{D}$, and,

$$\Sigma^{-1} = \mathbf{U}\mathbf{D}^{-1}\mathbf{U}^{T}$$

Be sure to cast all of the arrays to real.

2. Regularize the covariance matrix. You can do that by clipping the minimum of $\lambda_{i}$, i.e.,

$$
\tilde{\lambda}_{i} = \begin{cases}
 \lambda_{i}&\text{ if } \lambda_{i} > \epsilon,\\
 \epsilon & \text{ otherwise}
\end{cases}
$$

that way, you ensure that the determinant and the inverse are well defined.

### Initialization

- Do a first run with a random initialization.

- Compare your results with an initialization that relies on the $k-$Means algorithm. This algorithm is defined as,

1. Choose $K$ samples randomly from $\mathbf{X}$ and set it to $\mu = [\mu_{1},\cdots,\mu_{k}]$.
2. Compute $D_{ik} = \lVert \mathbf{x}_{i} - \mu_{k} \rVert$
3. Assign $c_{i} = \text{argmin}_{k}D_{ik}$
4. Recompute,

$$\mu_{k} = \dfrac{1}{n_{k}}\sum_{i:c_{i}=k}\mathbf{x}_{i}$$

5. Repeat step 2

- Note, when the k-Means algorithm stop, you may compute,

$$\hat{\mu}_{k} = \dfrac{1}{n_{k}}\sum_{i:c_{i}=k}\mathbf{x}_{i}$$

and,

$$\hat{\Sigma}_{k} = \dfrac{1}{n_{k}-1}\sum_{i:c_{i}=k}(\mathbf{x}_{i}-\mu_{k})(\mathbf{x}_{i}-\mu_{k})^{T}$$

and,

$$\pi_{k} = \dfrac{n_{k}}{n}$$

which will be the initialization of the GMM. Compare the evolution of the Likelihood using random and k-Means initializations.