import numpy

import theano
import theano.tensor as T

n_in = 784  # 28 * 28
n_hidden = 500
n_out = 10

def as_f32(v):
    return numpy.asarray(v).astype('float32')

class Model(object):
    def __init__(self, params):
        # This should respect the spec in params rather than use the
        # fixed arch below.
        self.W1 = theano.shared(value=numpy.asarray(
                numpy.random.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_hidden)),
                    high=numpy.sqrt(6. / (n_in + n_hidden)),
                    size=(n_in, n_hidden)
                    ),
                dtype='float32'
                ),
                           name='W1', borrow=True)

        self.b1 = theano.shared(value=numpy.zeros((n_hidden,), dtype='float32'),
                                name='b1', borrow=True)

        self.W2 = theano.shared(value=numpy.zeros((n_hidden, n_out), dtype='float32'),
                                name='W2', borrow=True)
        self.b2 = theano.shared(value=numpy.zeros((n_out,), dtype='float32'),
                                name='b2', borrow=True)
        self.lr = theano.shared(value=numpy.asarray(0.1, dtype='float32'),
                                name='lr', borrow=True)

        self.params = (W1, b1, W2, b2)
        self.grads = [theano.shared(numpy.zeros_like(param.get_value(borrow=True)))
                      for param in params]

        x = T.fmatrix('x')
        y = T.fmatrix('y')

        # Drop the last dim (which should be 1)
        y_r = y.dimshuffle(0)

        hidden = T.tanh(T.dot(x, W1) + b1)
        p_y_given_x = T.nnet.softmax(T.dot(hidden, W2) + b2)
        pred = T.argmax(p_y_given_x, axis=1)
        nll = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y_r])

        L2_sqr = (W1 ** 2).sum() + (W2 ** 2).sum()

        cost = nll + L2_sqr * as_f32(0.0001)

        gparams = [T.grad(cost, param) for param in params]

        updates = [(grad, grad - lr * gparam)
                   for grad, gparam in zip(grads, gparams)]

        # This does not update values, only accumulate gradients
        train = theano.function([x, y], cost, updates=updates)

        test = theano.function([x, y], cost)

    def size(self):
        return self.W1.size + self.W2.size + self.b1.size + self.b2.size

    @staticmethod
    def updbuf(buf, val, p, acc=False):
        l = val.size
        if acc:
            buf[p:p+l] += val.reshape(val.size)
        else:
            buf[p:p+l] = val.reshape(val.size)
        return p+l

    def getgrads(self, buf):
        p = 0
        for g in self.grads:
            p = self.updbuf(buf, g.get_value(borrow=True), p)

    def accgrads(self, buf):
        p = 0
        for g in self.grads:
            p = self.updbuf(buf, g.get_value(borrow=True), p, acc=True)

    def updatelr(self, newLR):
        self.lr.set_value(newLR)

    def getweights(self, buf):
        s = 0
        for p in self.params:
            s = self.updbuf(buf, p.get_value(borrow=True), s)

    def setweights(self, buf):
        s = 0
        for p in self.params:
            l = p.get_value(borrow=True).size
            p.set_value(b[s:s+l])
            s += l

    # This doesn't have adagrad yet, it's just to make sure the rest works.
    def updweights(self, buf, numMB):
        s = 0
        for p in self.params:
            pv = p.get_value(borrow=True)
            l = pv.size
            p.set_value(pv + b[s:s+l])
            s += l


def init(params):
    return Model(params)
