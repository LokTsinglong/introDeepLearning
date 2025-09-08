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
    
    def items(self):
        return [(key, self._dict[key]) for key in self.keys]


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
    
class Sigmoid:
    def __init__(self):
        self.out=None
    def forward(self,x):
        out=1/1+(np.exp(-x))
        self.out=out
        return out
    def backward(self,dout):
        dx=dout*(1.0-self.out)*self.out
        return dx
    
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

#此时，将这个class修改为可以选择权重初始化
class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size,weight_init="std",weight_init_std=0.01):
        if weight_init.lower()=='he':
            scale1=np.sqrt(2.0/input_size)
            scale2=np.sqrt(2.0/hidden_size)
        elif weight_init.lower()=='xavier':
            scale1=np.sqrt(1.0/input_size)
            scale2=np.sqrt(1.0/hidden_size)
        else:
            scale1=weight_init_std
            scale2=weight_init_std

        
        self.params = {}
        #self.params['W1']=weight_init_std*np.random.randn(input_size,hidden_size)*np.sqrt(2.0 / input_size)
        self.params['W1']=scale1*np.random.randn(input_size,hidden_size)
        self.params['b1']=np.zeros(hidden_size)
        #.params['W2']=weight_init_std*np.random.randn(hidden_size,output_size)*np.sqrt(1.0 / input_size)
        self.params['W2']=scale2*np.random.randn(hidden_size,output_size)
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
    
class Dropout:
    def __init__(self,dropout_ratio=0.5):
        self.dropout_ratio=dropout_ratio
        self.mask=None #掩码用于记录哪些神经元被drop掉了
    def forward(self,x,train_flag=True):
        if train_flag:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio #解包语法
            return x*self.mask
        else:
            #测试阶段不进行dropout，而是将输出值乘以(1-dropout_ratio)，测试阶段=推理阶段
            return x*(1.0 -self.dropout_ratio)
    def backward(self,dout):
        return dout*self.mask #只有被保留下来的神经元才有梯度 ？ #被drop掉的神经元梯度为0

class BatchNormalization:
    def __init__(self,gamma,beta,momentum=0.9,running_mean=None,running_var=None):
        self.gamma=gamma
        self.beta=beta
        self.momentum=momentum
        self.input_shape=None

        #训练时使用的中间结果
        self.batch_size=None
        self.xc = None
        self.std = None
        self.xn=None

        #推理时候用到的平均值和方差 
        self.running_mean = running_mean
        self.running_var = running_var

        #gradient
        self.dgamma=None
        self.dbeta=None
    
    def forward(self,x,train_flag=True):
        self.input_shape = x.shape
        if self.running_mean is None:
            N,D=x.shape #N is the number of samples, D is the number of features or dimensions
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        #training phase
        if train_flag:
            mu = x.mean(axis=0) #每个特征的均值
            xc = x-mu #mean
            var = np.mean(xc**2,axis=0) #variance
            std = np.sqrt(var+10e-7)
            xn = xc /std #standardization 
            #我就说这个公式这么这么熟悉，在都老师的数据分析课里面有上过

            self.batch_size = x.shape[0]
            self.xc= xc
            self.xn =xn
            self.std =std

            #更新huadong均值和方差
            self.running_mean = self.momentum*self.running_mean + (1-self.momentun)*mu
            self.running_var = self.momentum*self.running_var + (1-self.momentum)*var

        else:
            xc = x - self.running_mean
            xn = xc / (np.sqrt(self.running_var + 10e-7))
        out = self.gamma * xn + self.beta
        return out
    
    #根据计算图进行反向传播
    def backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std**2), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx


    


    
class MultiLayerNet:
    def __init__(self,input_size,hidden_size_list,output_size,activation='relu',weight_init_std='relu',
                 weight_decay_lambda=0,use_dropout=False,dropout_ratio=0.5,use_batchnorm=False):
        self.input_size=input_size
        self.output_size=output_size
        self.hidden_size_list=hidden_size_list
        self.hidden_layer_num=len(hidden_size_list)
        self.use_dropout=use_dropout
        self.weight_decay_lambda=weight_decay_lambda
        self.use_batchnorm = use_batchnorm
        self.params={}

        #初始化权重
        self.__init_weight(weight_init_std)

        #生成层
        activation_layer = {'sigmoid':Sigmoid,'relu':Relu}
        self.layers=OrderedDict()
        for idx in range(1,self.hidden_layer_num+1):
            self.layers['Affine'+str(idx)]=Affine(self.params['W'+str(idx)],self.params['b'+str(idx)])

            if self.use_batchnorm:
                self.params['gama'+str(idx)]=np.ones(hidden_size_list[idx-1]) #输出层的个数
                self.params['beta'+str(idx)]=np.zeros(hidden_size_list[idx-1])
                self.layers['BatchNorm'+str(idx)]=BatchNormalization(self.params['gama'+str(idx)],self.params['beta'+str(idx)])
            
            self.layers['Activation_function'+str(idx)]=activation_layer[activation]()

            if self.use_dropout:
                self.layers['Dropout'+str(idx)]=Dropout(dropout_ratio)
        #后面这段在循环外   
        idx=self.hidden_layer_num+1
        self.layers['Affine'+str(idx)]=Affine(self.params['W'+str(idx)],self.params['b'+str(idx)])
        self.lastLayer=SoftmaxWithLoss()

    def __init_weight(self,weight_init_std):
        all_size_list=[self.input_size]+self.hidden_size_list+[self.output_size]
        for idx in range(1,len(all_size_list)):
            scale=weight_init_std
            if str(weight_init_std).lower() in ('relu','he'):
                scale=np.sqrt(2.0/all_size_list[idx-1])
            elif str(weight_init_std).lower() in ('sigmoid','xavier'):
                scale=np.sqrt(1.0/all_size_list[idx-1])
            self.params['W'+str(idx)]=scale*np.random.randn(all_size_list[idx-1],all_size_list[idx])
            self.params['b'+str(idx)]=scale*np.zeros(all_size_list[idx])

    def predict(self,x,train_flag=False):
        for key,layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:
                x=layer.forward(x,train_flag=False)
            else:
                x=layer.forward(x)

        return x
    
    def loss(self,x,t,train_flag=False):
        y=self.predict(x,train_flag)
        weight_decay=0
        for idx in range(1,self.hidden_layer_num+2):
            W =self.params['W'+str(idx)]
            weight_decay += 0.5*self.weight_decay_lambda*np.sum(W**2) #原来是每一层的累加~
        return self.lastLayer.forward(y,t)+weight_decay
    
    def accuracy(self,X,T):
        Y=self.predict(X,train_flag=False)
        Y=np.argmax(Y,axis=1)
        if T.ndim !=1 : T=np.argmax(T,axis=1)
        accuracy = np.sum(Y==T)/float(X.shape[0])
        return accuracy
    
    def numerical_gradient(self,x,t):
        loss_W = lambda W : self.loss(x,t,train_flag=True)
        grads={}
        for idx in range(1,self.hidden_layer_num+2):
            global numerical_gradient_edited #设置为全局变量
            grads['W'+str(idx)]=numerical_gradient_edited(loss_W,self.params['W'+str(idx)])
            grads['b'+str(idx)]=numerical_gradient_edited(loss_W,self.params['b'+str(idx)])
            if self.use_batchnorm and idx !=self.hidden_size_list+1: #最后一层不需要
                grads['gama'+str(idx)]=numerical_gradient_edited(loss_W,self.params['gama'+str(idx)])
                grads['beta'+str(idx)]=numerical_gradient_edited(loss_W,self.params['beta'+str(idx)])
        return grads
    
    def gradient(self,x,t):
        #forward
        self.loss(x,t,train_flag=True)

        #backward
        dout=1
        self.lastLayer.backward(dout)

        layers=list(self.layers.values())
        layers.reverse()
        for layer in layers: #有点好奇这里的layer是什么结构
            dout = layer.backward(dout)
        #设定
        grads={}
        for idx in range(1,self.hidden_layer_num+2):
            grads['W'+str(idx)]=self.layers['Affine'+str(idx)].dW+self.weight_decay_lambda*self.layers['Affine'+str(idx)]
            grads['b'+str(idx)]=self.layers['Affine'+str(idx)].db

            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grads['gamma'+str(idx)]=self.layers['BatchNorm'+str(idx)].dgamma
                grads['beta'+str(idx)]=self.layers['BatchNorm'+str(idx)].dbeta
        return grads


            

