import torch

x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2 #basic graph with 2 inputs: x and 2 and output y
print(y)
z = y*y*2
print(z)

a = torch.tensor(1.0)
b = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

#forward pass and compute the loss

b_hat = w * a
loss = (b_hat - b)**2

print(loss)


#backward pass
loss.backward()
print(w.grad)