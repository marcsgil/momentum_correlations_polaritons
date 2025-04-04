The windowed Discrete Fourier Transform of our field is defined by

$$
\begin{equation}
    \psi_j(k_m) = \sum_{n=0}^{N_j - 1} \psi(x_n + x_{j,0}) \Pi_{j, m} e^{-i k_m x_n}
\end{equation}
$$

Here, $\psi$ is our quantum field in position space, $\Pi_j$ is a sequence of length $N_j$ representing our window. The field is evaluated at points spaced by a distane $\delta L$. We then have $x_{j,0} = n_{j,0} \delta L$ and $x_n = n \delta L$. Therefore, the window starts at position $x_{j,0}$ and has a total length of $L_j = N_j \delta L$. The points in momentum space are given by $k_m = 2\pi m / L$. Therefore, $\psi_j(k_m) = \psi_j(k_{m+N_j})$.


For the truncated Wigner method, we have that

$$
\begin{equation}
    \begin{aligned}
        \langle \psi^\dagger_2(k^\prime) \psi^\dagger_1(k) \psi_1(k) \psi_2(k^\prime) \rangle = \langle \left| \psi_1 \right|  \rangle_W
    \end{aligned}
\end{equation}
$$