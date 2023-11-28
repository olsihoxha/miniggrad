# MiniGGrad
![Mini-G Logo](mini-g.png)

Introducing MiniGGrad – a pint-sized Autograd engine! 🐜🚀 This tiny powerhouse implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and supports a small neural networks library on top. With around 100 lines for the DAG and 50 for the library, it operates on scalar values, slicing neurons into minuscule adds and multiplies. Surprisingly robust,
it can build entire deep neural nets, making it potentially useful for educational purposes.

## Example Usage

Here is a somewhat artificial example demonstrating various supported operations:
```python
from miniggrad.engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db
```

## How to train neural net
The notebook `demo.ipynb` provides a full demo of training an 2-layer neural network (MLP) binary classifier.


## Visualization
The notebook `demo.ipynb` provides an example on how to visualize 
with `graphviz` based on the `draw_dot` function we have.  
