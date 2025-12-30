import torch.nn
import sympy as sp


#intro

print('Hi! My name is Coniec! Give me a polynomial function and I will try to guess what the output is for an input that you provide - without explicitly calculating it!')
function_input = input('Enter your function here - remember, use strict python syntax: * for multiplication, / for division, ** for exponentiation, + for addition, -for subtraction and coefficients before the variable!-----:')

#geet bounds
print('Enter the upper and lower bounds for x values you would like to give me:')
upper=float(input('Upper bound:'))
lower=float(input('Lower bound:'))


# make training input values
x_s = torch.arange(lower, upper, 0.11)


#get function from user
x_input = sp.symbols('x')
function = sp.lambdify(x_input, function_input, 'numpy')

# make training output values
y_s = function(x_s)

#choose number of neurons in layer
n_hidden_1 = 200

#make parameters
W1 = torch.randn(1, n_hidden_1)
b1 = torch.randn(1, n_hidden_1)

W2 = torch.randn(n_hidden_1,1)
b2 = torch.randn(1,1)


parameters = [W1, W2, b1, b2]



for p in parameters:
    p.requires_grad = True


#repeat training 100000 times!
max_steps = 100000

#we calculate loss function of a batch of inputs to avoid overfitting...
batch_size = 32

#we normalise training data and train our multilayer perceptron on this. This stops our loss function from exploding!
x_mean, x_std = x_s.mean(), x_s.std()
y_mean, y_std = y_s.mean(), y_s.std()

x_s_norm = (x_s - x_mean) / x_std
y_s_norm = (y_s - y_mean) / y_std

#define the loss function. This is a regression problem so we use the mean squared error loss function.
loss_fn = torch.nn.MSELoss()

#begin training
for i in range(max_steps):

    #generate batch indices, i.e. choose 32 random indices we can use to pick out values from our training set
    ix = torch.randint(0,len(x_s), (batch_size,1) )
    Xb, Yb = x_s_norm[ix], y_s_norm[ix]


    #forward pass

    #Linear layer
    hpreact = Xb @ W1 + b1



    #Non-linearity
    h = torch.tanh(hpreact)

    #2nd linear layer
    output = h @ W2 + b2

    #find loss
    mse_loss = loss_fn(Yb, output)


    #backward pass - we calculate gradient of loss function with respect to EVERY SINGLE PARAMETER.
    for p in parameters:
        p.grad=None
    mse_loss.backward()

    #we update every parameter by a small step in the opposite direction to the partial derivative of the loss function with respect to this parameter.
    #This changes each parameter as to minimise the loss function and make our neural network as accurate as possible. A classic step of the backpropagation algorithm.
    lr = 0.01 if i <50000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad


    #print our loss function value every now and then. Ideally it would keep decreasing, but here it fluctuate towards the end. We risk overfitting our neural net
    #to our data. More time would have let me set up batch normalisation for each layer to combat this issue.
    if i %10000 ==0:
        print(f'{i:7d}/{max_steps:7d} training rounds : loss = {mse_loss.item():.4f}')




#sample from model
flag=True
while flag==True:
    x_sample = float(input(f'enter an x value between {lower} and {upper} to enter into your function {function_input}: '))
    #our sample is normalised
    x_sample_norm = (x_sample-x_mean)/x_std
    x_sample_norm = torch.tensor([[x_sample_norm]])
    y_sample_norm = torch.tanh(x_sample_norm @ W1 +b1) @ W2 + b2
    #the network output is denormalised before being displayed.
    y_sample =  y_sample_norm*y_std+y_mean
    print('I think the y value is', y_sample.item())
    cont = input(f'The actual value is {function(x_sample)} Was I close?, Type "y" if you want to input another x between {lower} and {upper} or any other key to stop ')
    if cont=='y':
        pass
    else:
        flag=False
