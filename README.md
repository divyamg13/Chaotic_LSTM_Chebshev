I've currently made the n, alpha and beta parameters as NON trainable for testing purposes within v3_Multivariate_WithActivation
If you want to make it trainable again change this
self.n = n
return self.n
to
self.n = nn.Parameter(torch.tensor(float(n), dtype=torch.float32), requires_grad=True)
return self.n.item()