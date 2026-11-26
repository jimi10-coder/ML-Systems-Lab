import math

class Value:
    def __radd__(self, other):
        return self + other

    def __rtruediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other * (self ** -1)

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0

        # Parent nodes in the computation graph
        self._prev = set(_children)

        # Operation that created this node
        self._op = _op

        # Function used to calculate gradients
        self._backward = lambda: None

    def __rsub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other + (-self)
    
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data + other.data,
            (self, other),
            "+"
        )

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data * other.data,
            (self, other),
            "*"
        )

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
    
    def __pow__(self, power):
        assert isinstance(power, (int, float)), "Power must be a number"

        out = Value(
            self.data ** power,
            (self,),
            f"**{power}"
        )

        def _backward():
            self.grad += (
                power * self.data ** (power - 1)
            ) * out.grad

        out._backward = _backward
        return out
   
    def relu(self):
        out = Value(
            max(0, self.data),
            (self,),
            "ReLU"
        )

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        return 1.0 / (1.0 + (-self).exp())

    
    def __neg__(self):
        return self * -1

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1)   
    def backward(self):
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)

                for child in node._prev:
                    build_topo(child)

                topo.append(node)

        build_topo(self)

        # The output gradient starts at 1
        self.grad = 1.0

        # Backpropagate through the graph
        for node in reversed(topo):
            node._backward()


if __name__ == "__main__":
    a = Value(0.0)
    b = a.sigmoid()

    b.backward()

    print("Sigmoid output:", b.data)
    print("Gradient:", a.grad)