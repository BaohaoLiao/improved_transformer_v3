

def kurtosis(x, eps=1e-6):
    """x - (B, d)"""
    mu = x.mean(dim=1, keepdims=True)
    s = x.std(dim=1)
    mu4 = ((x - mu) ** 4.0).mean(dim=1)
    k = mu4 / (s**4.0 + eps)
    return k