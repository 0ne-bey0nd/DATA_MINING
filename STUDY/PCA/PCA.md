# 数据挖掘学习记录——PCA

---

## 需要用到的定义、定理、引理等

### 定义1：向量的内积与投影

#### 向量内积（Vector Dot Product）

内积，也称为点积或数量积，是在向量空间中两个向量之间的一种运算。给定两个向量 $\mathbf{a} = (a_1, a_2, \ldots, a_n)^T$ 和 $\mathbf{b} = (b_1, b_2, \ldots, b_n)^T$，它们的内积 $\mathbf{a} \cdot \mathbf{b}$ 定义为它们对应分量的乘积之和：

$$
\mathbf{a} \cdot \mathbf{b} = (a_1, a_2, \ldots, a_n)^T \cdot (b_1, b_2, \ldots, b_n)^T = a_1b_1 + a_2b_2 + \ldots + a_nb_n = \sum_{i=1}^{n} a_ib_i
$$

其中，$a_i$ 和 $b_i$ 分别表示向量 $\mathbf{a}$ 和 $\mathbf{b}$ 的第 $i$ 个分量。

内积的几何意义是两个向量之间的投影乘积之和。

#### 向量投影（Vector Projection）

给定两个非零向量 $\mathbf{a}$ 和 $\mathbf{b}$，$\mathbf{a}$ 在 $\mathbf{b}$ 方向上的投影是一个向量，它的方向与 $\mathbf{b}$ 相同，长度为 $\mathbf{a}$ 在 $\mathbf{b}$ 方向上的投影长度。

如果 $\theta$ 是向量 $\mathbf{a}$ 和向量 $\mathbf{b}$ 之间的夹角（角度），则向量 $\mathbf{a}$ 在向量 $\mathbf{b}$ 上的投影（记作 $\text{proj}_{\mathbf{b}} \mathbf{a}$）的长度为 $||\mathbf{a}|| \cos \theta$，方向为 $\mathbf{b}$。

向量 $\mathbf{a}$ 在向量 $\mathbf{b}$ 上的投影可以用内积来表示：

$$
\text{proj}_{\mathbf{b}} \mathbf{a} = \frac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{b}||^2} \mathbf{b}
$$

### 定义2: 特征值与特征向量

给定一个方阵 $\mathbf{A}$，如果存在一个标量 $\lambda$ 和一个非零向量 $\mathbf{v}$，使得下式成立：

$$
\mathbf{A} \cdot \mathbf{v} = \lambda \cdot \mathbf{v}
$$

其中，$\mathbf{v}$ 被称为 $\mathbf{A}$ 的特征向量，$\lambda$ 被称为 $\mathbf{A}$ 的特征值。

特征向量表示在矩阵变换下不改变方向的向量，而特征值表示这个方向上的缩放比例。

### 定义3：正交矩阵

> 参考https://blog.csdn.net/MyArrow/article/details/53445369

旋转矩阵属于正交矩阵

#### 定理1：实对称矩阵的特征向量是正交的

---

## PCA算法原理关键

### <span id="jump">PCA的目标（数学表示）</span>
通过线性变换将一组N维向量降为K维（K大于0，小于N），即择K个单位（模为1）正交基，使得原始数据变换到这组基上（将原始数据映射到一个新的坐标系）后，使得变换后的数据下符合以下条件：

- 相同特征之间**方差最大**
- 不同特征之间**协方差最小**

### PCA算法的核心思想（PCA算法实际在做什么）

**将数据投影到数据的协方差矩阵的特征向量上。**

- 因为数据的协方差矩阵是对称矩阵，所以特征向量是正交的，选择前K个特征向量，就是选择了一个K维的子空间，将数据投影到这个子空间上，实现了数据的降维。
- 又因为特征向量排序是按照特征值的大小排序的，不同的特征向量代表不同的主成分，最大的特征值对应的特征向量上投影的方差最大（即保留信息最多的主成分），所以选择前K个特征向量，降维后的数据保留了最多的信息。

### PCA算法的黑盒描述（算法使用者视角，用户视角）

#### 输入：

假设我们有 $m$ 条数据，每条数据具有 $N$ 个特征。我们将这些数据表示为一个 $N \times m$ 的矩阵 $X$，其中每一列代表一条数据，每一行代表一个特征。矩阵 $X$ 可以表示为：

$$
X = \begin{bmatrix} x_{11} & x_{12} & \cdots & x_{1m} \\ x_{21} & x_{22} & \cdots & x_{2m} \\ \vdots & \vdots & \ddots & \vdots \\ x_{N1} & x_{N2} & \cdots & x_{Nm} \end{bmatrix}
$$

其中，$x_{ij}$ 表示第 $i$ 条数据的第 $j$ 个特征的取值。

#### 输出：

PCA 算法的输出包括两部分：降维后的数据和变换矩阵（特征向量矩阵）。

1. **降维后的数据**：假设我们希望将数据降维到 $K$ 维，其中 $K$ 是一个小于 $N$ 的整数。降维后的数据可以表示为一个 $K \times m$ 的矩阵 $Y$，其中每一列代表一条数据，每一行代表一个新的特征（主成分）。矩阵 $Y$ 可以表示为：

$$
Y = \begin{bmatrix} y_{11} & y_{12} & \cdots & y_{1m} \\ y_{21} & y_{22} & \cdots & y_{2m} \\ \vdots & \vdots & \ddots & \vdots \\ y_{K1} & y_{K2} & \cdots & y_{Km} \end{bmatrix}
$$

其中，$y_{ij}$ 表示第 $i$ 个主成分在第 $j$ 条数据上的取值。

2. **变换矩阵（特征向量矩阵）**：PCA 算法会输出一个 $N \times K$ 的矩阵 $V$，其中每一列代表一个主成分对应的特征向量。这个矩阵可以表示为：

$$
V = \begin{bmatrix} v_{11} & v_{12} & \cdots & v_{1K} \\ v_{21} & v_{22} & \cdots & v_{2K} \\ \vdots & \vdots & \ddots & \vdots \\ v_{N1} & v_{N2} & \cdots & v_{NK} \end{bmatrix}
$$

其中，$v_{ij}$ 表示第 $j$ 个主成分对应的特征向量在原始特征空间的第 $i$ 个维度上的取值。

$X$, $Y$, $K$ 三者之间的关系可以用下式表示：

$$
Y = V^T \cdot X
$$

输出Y符合[PCA的目标](#jump),即相同特征之间**方差最大**，不同特征之间**协方差最小**，实现了数据的降维。



### PCA算法的白盒描述（算法实现者视角）


## 不错的内容分享

- [PCA的数学原理](https://blog.codinglabs.org/articles/pca-tutorial.html)
- [chrome LaTex不显示问题推荐插件](https://chromewebstore.google.com/detail/tex-all-the-things/cbimabofgmfdkicghcadidpemeenbffn)
- 

