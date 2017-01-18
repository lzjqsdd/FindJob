# ICP相关知识

## 旋转矩阵是正交矩阵

旋转矩阵是具有特殊性质的一类矩阵

正交矩阵的性质：$X^TX=E$

假设三维空间中的坐标系，坐标系上的单位向量$X, Y, Z$，经过一个旋转矩阵R，得到一组新的单位向量$A, B, C$，但是新的单位向量仍然应当是相互正交的，那么我们可得：
$$
R(X, Y, Z) = (A, B, C)
$$

$$
A^TB=X^TR^TRY=0
$$

假如$R^TR=E$，则上式满足。

因此通过这个不严谨的证明，我们得到了旋转矩阵就是正交矩阵这个结论。

所有旋转矩阵的行列式都为1。

$(1, 0)$经过逆时针旋转$\theta$变为$(cos\theta,sin\theta)$，$(0, 1)$经过旋转变为$(-sin\theta, cos\theta)$。
$$
RX=A
$$
则就可以得到R矩阵：
$$
R=[\begin{array}\\cos\theta&-sin\theta\\sin\theta&\cos\theta\end{array}]
$$

## 向量的不同基底表示形式

$$
[\begin{array}\\1&0\\0&1\end{array}][\begin{array}\\3\\4\end{array}]=[\begin{array}\\2&0\\0&2\end{array}][\begin{array}\\\frac{3}{2}\\\frac{4}{2}\end{array}]
$$

左边是在标准单位坐标系下，坐标为$(3,4)$，右边是在新坐标系下，坐标为$(\frac{2}{3},\frac{4}{2})$，右边的矩阵是将右边的坐标系转换到左边的坐标系的矩阵。
$$
[\begin{array}\\\frac{1}{2} & 0\\0&\frac{1}{2}\end{array}][\begin{array}\\3\\4\end{array}]=[\begin{array}\\\frac{3}{2}\\\frac{4}{2}\end{array}]
$$
坐标的矩阵就是将前一个坐标系，转换到后一个坐标系的变换矩阵。