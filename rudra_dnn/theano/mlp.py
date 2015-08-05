import numpy

import theano
import theano.tensor as T

n_in = 784  # 28 * 28
n_hidden = 500
n_out = 10

def as_f32(v):
    return numpy.asarray(v).astype('float32')

W1 = theano.shared(value=numpy.asarray(
        numpy.random.uniform(
            low=-numpy.sqrt(6. / (n_in + n_hidden)),
            high=numpy.sqrt(6. / (n_in + n_hidden)),
            size=(n_in, n_hidden)
            ),
        dtype='float32'
        ),
                   name='W1', borrow=True)

b1 = theano.shared(value=numpy.zeros((n_hidden,), dtype='float32'),
                   name='b1', borrow=True)

W2 = theano.shared(value=numpy.zeros((n_hidden, n_out), dtype='float32'),
                   name='W2', borrow=True)
b2 = theano.shared(value=numpy.zeros((n_out,), dtype='float32'),
                   name='b2', borrow=True)

params = (W1, b1, W2, b2)
grads = [theano.shared(numpy.zeros_like(param.get_value(borrow=True)))
         for param in params]

x = T.fmatrix('x')
y = T.fvector('y')

hidden = T.tanh(T.dot(x, W1) + b1)
p_y_given_x = T.nnet.softmax(T.dot(hidden, W2) + b2)
pred = T.argmax(p_y_given_x, axis=1)
nll = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])

L2_sqr = (W1 ** 2).sum() + (W2 ** 2).sum()

cost = nll + L2_sqr * as_f32(0.0001)

gparams = [T.grad(cost, param) for param in params]

updates = [(grad, grad - as_f32(0.01) * gparam)
           for grad, gparam in zip(grads, gparams)]

# This does not update values, only accumulate gradients
train = theano.function(inputs=[x, y], updates=updates)

def set_params(W1v, b1v, W2v, b2v):
    W1.set_value(W1v)
    b1.set_value(b1v)
    W2.set_value(W2v)
    b2.set_value(b2v)


# This also resets the current gradients to 0
def get_updates(W1u, b1u, W2u, b2u):
    W1u[:] = grads[0].get_value()
    grads[0].set_value(numpy.zeros_like(W1u))
    b1u[:] = grads[1].get_value()
    grads[1].set_value(numpy.zeros_like(b1u))
    W2u[:] = grads[2].get_value()
    grads[2].set_value(numpy.zeros_like(W2u))
    b2u[:] = grads[3].get_value()
    grads[3].set_value(numpy.zeros_like(b2u))
