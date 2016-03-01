import numpy

import time
import theano
from theano.ifelse import ifelse
from theano import printing
import theano.tensor as T

n_in = 784  # 28 * 28
n_hidden = 500
n_out = 10
lr_init = 0.1

def as_f32(v):
    return numpy.asarray(v).astype('float32')

class Model(object):
    def __init__(self, params):
        # This should respect the spec in params rather than use the
        # fixed arch below.
#        print("In init.")
#        print(str(self))

        self.W1 = theano.shared(value=numpy.asarray(
                numpy.random.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_hidden)),
                    high=numpy.sqrt(6. / (n_in + n_hidden)),
                    size=(n_in, n_hidden)),
                dtype='float32'),
                name='W1', borrow=True)

        self.b1 = theano.shared(value=numpy.zeros((n_hidden,), dtype='float32'),
                                name='b1', borrow=True)

        self.W2 = theano.shared(value=numpy.zeros((n_hidden, n_out), dtype='float32'),
                                name='W2', borrow=True)
        self.b2 = theano.shared(value=numpy.zeros((n_out,), dtype='float32'),
                                name='b2', borrow=True)
        self.lr = theano.shared(value=numpy.asarray(lr_init, dtype='float32'),
                                name='lr', borrow=True)

        self.params = (self.W1, self.b1, self.W2, self.b2)
        self.grads = [theano.shared(numpy.zeros_like(param.get_value(borrow=True)))
                      for param in self.params]

        self.x = T.fmatrix('x')
        self.y = T.fmatrix('y')

        # Drop the last dim (which should be 1)
        #y_r = y.dimshuffle(0)

        hidden = T.tanh(T.dot(self.x, self.W1) + self.b1)
        p_y_given_x = T.nnet.softmax(T.dot(hidden, self.W2) + self.b2)
        pred = T.argmax(p_y_given_x, axis=1)
        #nll = -T.mean(T.log(p_y_given_x)[T.arange(self.y.shape[0]), T.arange(self.y.shape[0])])
        # self.y.argmax(axis=1) this is just the target index.
        # Currently y is defined as a matrix with the one hot encoding.
        # If Rudra can pass y as a index of the target, this could give a very small speed up
        nll = -T.mean(T.log(p_y_given_x)[T.arange(self.y.shape[0]), self.y.argmax(axis=1)])

        L2_sqr = (self.W1 ** 2).sum() + (self.W2 ** 2).sum()

        cost = nll + L2_sqr * as_f32(0.0001)

        gparams = T.grad(cost, self.params)

        gs = [(grad, gparam) for grad, gparam in zip(self.grads, gparams)]
        #f_gshared = theano.function([x, y], [], updates=gs]  # not required if we do update as part of train

        # This does not update values, only accumulate gradients
        #self.train = theano.function([x, y], cost, updates=updates)
        self.train = theano.function([self.x, self.y], cost, updates=gs)

        #updates = [(grad, grad - self.lr * gparam)
        #           for grad, gparam in zip(self.grads, gparams)]
        param_updates = [(param, param - self.lr * grad)
                       for param, grad in zip(self.params, self.grads)]
        self.pup = theano.function([], [], updates=param_updates)

        error = T.mean(T.neq(pred, self.y.argmax(axis=1)) * 1.0)
        self.test = theano.function([self.x, self.y], error)

    def size(self):
        #return self.W1.size + self.W2.size + self.b1.size + self.b2.size
        return self.W1.get_value().size + self.W2.get_value().size + self.b1.get_value().size + self.b2.get_value().size

    @staticmethod
    def updbuf(buf, val, p, acc=False):
        l = val.size
        if acc:
            buf[p:p+l] += val.flatten()
        else:
            buf[p:p+l] = val.flatten()
        return p+l

    def get_grads(self, buf):
        s = 0
        for g in self.grads:
            s = self.updbuf(buf, g.get_value(borrow=True), s)

    def acc_grads(self, buf):
        s = 0
        for g in self.grads:
            s = self.updbuf(buf, g.get_value(borrow=True), s, acc=True)

    def set_lr_mult(self, lrMult):
        self.lr.set_value(float(lr_init * lrMult), allow_input_downcast=True)

    def get_params(self, buf):
        #print(self)
        s = 0
        tot_size = 0
        for p in self.params:
            val = p.get_value(borrow=True)
            tot_size += val.size
            if buf.size == 0:
                buf = numpy.zeros(tot_size, dtype=numpy.float32)
            elif buf.size < tot_size:
                buf.resize(tot_size)

            s = self.updbuf(buf, val, s)

#        print 'Buf mean: ', buf.mean()
#        print 'buf[0:5]: ', buf[0:5]

    def set_params(self, buf):
        #print(self)
        s = 0
        new_params = []
        for p in self.params:
            p_val = p.get_value(borrow=True)
            t = s + p_val.size
            new_p = numpy.reshape(buf[s:t], p_val.shape)
            p.set_value(new_p)
            #new_params.append(new_p)
            s = t

        #upp = [(o_p, n_p) for o_p, n_p in zip(self.params, new_params)]
        #f_update_params = theano.function([], [], updates=upp)
        #f_update_params()
        """
        s = 0
        for p in self.params:
            l = p.get_value(borrow=True).size
            import pdb; pdb.set_trace()
            #p.set_value(T.reshape(buf[s:s+l], p.get_value().shape), borrow=False)
            p = numpy.reshape(buf[s:s+l], p.get_value().shape)
            s += l
        """

    # This doesn't have adagrad yet, it's just to make sure the rest works.
    # This should be update gradients NOT update parameters ** upd_grads **
    def upd_grads(self, buf, numMB):
        mult = 1.0 / numMB
        s = 0
        for g in self.grads:
            g_val = g.get_value(borrow=True)
            t = s + g_val.size
            new_g = numpy.reshape(buf[s:t], g_val.shape) * mult
            g.set_value(new_g)  # Can we use borrow=True? I don't think so, but I'm not 100% sure.
#            new_grads.append(new_g)
            s = t

        # Also update the params with these (all-reduced) grads
        self.pup()
        """
        for p in self.params:
            pv = p.get_value(borrow=True)
            l = pv.size
            p.set_value(pv + T.reshape(buf[s:s+l], pv.shape))
            s += l
        """

def myinit(params):
#    print("Golden: In init with params ")
#    print(str(params))
    return Model(params)
