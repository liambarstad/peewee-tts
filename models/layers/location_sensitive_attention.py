from torch import nn

class LocationSensitiveAttention(nn.Module):
    def __init__(self,
                 hidden_dims: int
                 ):
        # f(i) = F * a(i-1) // convolving it with matrix [kXr]
        # e(i,j) = wT*tanh(W*s(i-1) + V*h(j) + U*f(i,j) + b)
        # use windowing technique, w/ window size
        # a(i,j) = sig(Be(i,j)) / sum(sig(Be(i,j)))
        # g(i) = sum(a(i,j)*h(j))
        # y(i) = generate(g(i), s(i-1))

        # end of sequence token
        pass

    def forward(self, x):
        import ipdb; ipdb.sset_trace()
        pass