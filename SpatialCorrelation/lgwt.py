def lgwt(N, a, b):
    """
    Compute Gauss–Legendre quadrature nodes and weights on [a, b].
    """
    x, w = leggauss(N)
    # Map from [-1, 1] to [a, b]
    x = 0.5 * (b - a) * (x + 1) + a
    w = 0.5 * (b - a) * w
    return x, w
