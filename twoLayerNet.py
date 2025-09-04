import numpy as np

class OrderedDict:
    def __init__(self):
        self.keys=[]
        self._values_list=[] #这边self的属性不可以和后续的values方法同一个名字，否则会覆盖掉
        self._dict={}  #内部字典用于快速查找
    
    def __setitem__(self,key,value):
        if key not in self._dict:
            self.keys.append(key)
            self._values_list.append(value)
        self._dict[key]=value
    
    def __getitem__(self,key):
        return self._dict[key]
    
    def values(self):
        return self._values_list.copy()
    
    def __contains__(self,key):
        return key in self._dict

def numerical_gradient_edited(f,x):
    h=1e-4
    grad=np.zeros_like(x)
    it = np.nditer(x,flags=["multi_index"],op_flags=["readwrite"])
    while not it.finished:
        idx=it.multi_index
        original_value = x[idx]
        x[idx] = original_value + h
        fxh1=f(x)
        x[idx] = original_value - h
        fxh2=f(x)
        grad[idx] = (fxh1-fxh2)/(2*h)
        x[idx] = original_value
        it.iternext()
    return grad

def numerical_gradient_edited(f,x):
    h=1e-4
    grad=np.zeros_like(x)
    it = np.nditer(x,flags=["multi_index"],op_flags=["readwrite"])
    while not it.finished:
        idx=it.multi_index
        original_value = x[idx]
        x[idx] = original_value + h
        fxh1=f(x)
        x[idx] = original_value - h
        fxh2=f(x)
        grad[idx] = (fxh1-fxh2)/(2*h)
        x[idx] = original_value
        it.iternext()
    return grad

def cross_entropy_error_new(y,t):
    delta=1e-7
    batch_size=y.shape[0]
    #如果t标签是一维数组而不是one-hot的情况
    #也就是load_digits里数据的情况
    if t.ndim == 1:
        t_onehot = np.zeros_like(y)
        t_onehot[np.arange(batch_size),t]=1
        t=t_onehot
    return -np.sum(t*np.log(y+delta))/batch_size

class Affine:
    def __init__(self,W,b):
        self.W=W
        self.b=b
        self.x=None
        self.dW=None
        self.db=None
    def forward(self,x):
        self.x=x
        out=np.dot(x,self.W)+self.b
        return out
    def backward(self,dout):
        dx=np.dot(dout,self.W.T)
        self.dW=np.dot(self.x.T,dout)
        self.db=np.sum(dout,axis=0)
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss=None
        self.y=None
        self.t=None
    def forward(self,x,t):
        self.t= t
        self.y=softmax_new(x)
        self.loss=cross_entropy_error_new(self.y,self.t)
        return self.loss
    def backward(self,dout=1):
        #batch_size = self.t.shape[0]
        #dx=(self.y-self.t) / batch_size
        dx=backward_f(self.y,self.t)
        return dx
        #r如果没有这个return的话，SoftmaxWithLoss.backward 没有返回值，导致在 TwoLayerNet.gradient() 里，
        # 反向传播链条传下去的 dout 变成了 None，最后在 Affine.backward 里做矩阵乘法的时候就报了
        # dx=np.dot(dout,self.W.T)
        #TypeError: unsupported operand type(s) for *: 'NoneType' and 'float'

def backward_f(y,t):
    batch_size=y.shape[0]
    if t.ndim == 1:
        t_onehot = np.zeros_like(y)
        t_onehot[np.arange(batch_size),t]=1
        t=t_onehot
    return (y-t)/batch_size

def softmax_new(a):
    if a.ndim ==2: #说明这个是批量数据
        a = a - np.max(a,axis=1,keepdims=True)
        exp_a = np.exp(a)
        sum_exp_a=np.sum(exp_a,axis=1,keepdims=True)
        y=exp_a / sum_exp_a
    else:
        c=np.max(a) 
        exp_a=np.exp(a-c) #防止溢出
        sum_exp_a = np.sum(exp_a)
        y=exp_a/sum_exp_a
    return y

class Relu:
    def __init__(self):
        self.mask=None
    def forward(self,x):
        self.mask = (x<=0)
        out=x.copy()
        out[self.mask]=0
        return out
    def backward(self,dout):
        dout[self.mask]=0
        dx=dout
        return dx
    
class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        self.params = {}
        #self.params['W1']=weight_init_std*np.random.randn(input_size,hidden_size)
        self.params['W1']=weight_init_std*np.random.randn(input_size,hidden_size)*np.sqrt(2.0 / input_size)
        self.params['b1']=np.zeros(hidden_size)
        #self.params['W2']=weight_init_std*np.random.randn(hidden_size,output_size)
        self.params['W2']=weight_init_std*np.random.randn(hidden_size,output_size)*np.sqrt(1.0 / input_size)
        self.params['b2']=np.zeros(output_size)

        #生成层
        self.layers=OrderedDict()
        self.layers['Affine1']=Affine(self.params['W1'],self.params['b1'])
        self.layers['Relu1']=Relu()
        self.layers['Affine2']=Affine(self.params['W2'],self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self,x,t):
        y=self.predict(x)
        return self.lastLayer.forward(y,t)

    def accuracy(self,x,t):
        y=self.predict(x)
        y=np.argmax(y,axis=1)
        if t.ndim != 1 : t = np.argmax(t,axis=1)
        accuracy = np.sum(y==t)/float(x.shape[0])
        return accuracy
    
    
    def numerical_gradient(self,x,t):
        loss_W = lambda W : self.loss(x,t)
        grads={}
        global numerical_gradient_edited #设置为全局变量
        grads['W1']=numerical_gradient_edited(loss_W,self.params['W1'])
        grads['b1']=numerical_gradient_edited(loss_W,self.params['b1'])
        grads['W2']=numerical_gradient_edited(loss_W,self.params['W2'])
        grads['b2']=numerical_gradient_edited(loss_W,self.params['b2'])

        return grads
    
    def gradient(self,x,t):
        #forward
        self.loss(x,t)

        #backward
        dout=1
        dout=self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        grads={}
        grads['W1']=self.layers['Affine1'].dW
        grads['b1']=self.layers['Affine1'].db
        grads['W2']=self.layers['Affine2'].dW
        grads['b2']=self.layers['Affine2'].db

        return grads
