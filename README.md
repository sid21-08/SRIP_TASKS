# SRIP_TASKS
TASK-2:
#Multivariate Normal Distribution sampling



In the beginning I randomly created standard normal samples by using Box-Mueller transformation. I also created covariance and mean matrix by using jax library
I then multiplied covariance matrix with its transpose in order to make the covariance matrix symmetric. This is dBone for the cholesky decomposition.
CHOLESKY DECOMPOSITION:
The Cholesky decomposition or Cholesky factorization is a decomposition of a Hermitian, positive-definite matrix into the product of a lower triangular matrix and its conjugate transpose. The Cholesky decomposition is roughly twice as efficient as the LU decomposition for solving systems of linear equations.
In univariate normal distribution, Y=T+PX where T is mean vector and P is standard deviation. For multivariate normal distribution, Y=T+LX. Here we have the covariance matrix instead of variance. Like in univariate distribution where we take the square root of variance into consideration, here we take the square root of symmetric positive definite matrix i.e. Cholesky factor into consideration.
![image](https://user-images.githubusercontent.com/59621102/162633495-25a0d635-95a3-4c28-b3f3-197eca0ba527.png)


This is the initial covariance matrix.
![image](https://user-images.githubusercontent.com/59621102/162633525-36951ac0-b60b-4d44-8d48-6997ebd07037.png)



This is the final covariance matrix

In the end I used jax.numpy package to find mean and covariance of the newly formed sample. Comparing mean and covariance of both the samples i.e the beginning and the newly formed sample I found out that mean and covariance values are very close to each other.
Hence sampling method is valid or correct.
LIMITATIONS:
I tried to implement 10 dimensional matrix but I faced an error of the covariance matrix of not being symmetric even after Multiplying it with its transpose. I tried varying the order of the covariance and mean matrix but to no avail.  Hence I did the sampling on 2D dataset.

TASK-3:
#Hidden layer neural network classifier




In the beginning I loaded the MNIST dataset with the help of tensorflow datasets. Once I have loaded the dataset, I can easily extract the training and testing dataset with the built references. Instead of splitting the MNIST dataset into 80:20 I have used built in reference. 
I also used validation i.e. 10% of training data.

The next step is to outline the model. I have taken the input layer_size as 784 and output layer_size as 10. I have manually taken number of neurons in the hidden layer as 50.
I then successfully created two hidden layers whose activation function is ReLu. The output layer's activation function is SoftMax.



![image](https://user-images.githubusercontent.com/59621102/162633595-8280da63-5392-443c-9581-f7294db3d4cf.png)




I then compiled the model by using Adam Optimizer

Next, I tried training the model with maximum number of epochs set to 5.
![image](https://user-images.githubusercontent.com/59621102/162633630-08e427d3-aac0-4b9d-819e-bf067885ad51.png)


At the end, I tried finding out the accuracy of my model. My model's accuracy is 96.57%.
![image](https://user-images.githubusercontent.com/59621102/162633646-0a30d717-3fcc-4169-a3a3-48bef913eea2.png)

LIMITATIONS:
I was unable to use the JAX gradient descent method.

TASK-4:
#Bayesian Linear Regression using BlackJax sampler




In the beginning I generated artificial linear data.




![image](https://user-images.githubusercontent.com/59621102/162633677-3110e9f0-de66-4a72-b0df-8de8f7eacd50.png)





I then tried using the blackjax sampler(HMC).
![image](https://user-images.githubusercontent.com/59621102/162633697-5f0f85b4-dd65-44e5-8126-71120e7c8144.png)

I also created PYMC3 module. I then created a basic module which contains three parameters:alpha, beta and sigma. 
![image](https://user-images.githubusercontent.com/59621102/162633718-245ca8db-a99a-480b-9383-43d27e25a5a0.png)

I then tried posterior sampling using pm model. PyMC3 automatically chooses appropriate model depending on the type of data. In my case of continuous data, NUTS is used.
![image](https://user-images.githubusercontent.com/59621102/162633729-2795f02d-7e46-4d7b-81e5-9c5c8eff6d22.png)

I also tried nuts sampling using blackjax. But I was unable to implement the black jax samplers.
![image](https://user-images.githubusercontent.com/59621102/162633753-d9b41e93-5488-4862-b3ff-92978101d37d.png)


I then summarised the PyC3 sampler(nuts) in pm model.
![image](https://user-images.githubusercontent.com/59621102/162633782-7b4505a5-8970-4129-854e-466a6c4c41a7.png)


In order to compare the simple linear regression and bayesian linear regression I plotted both graphs. The mean value for both parameters is same for both simple and bayesian linear regression models.


![image](https://user-images.githubusercontent.com/59621102/162633806-1319ee6a-cbfa-4ef7-b4b6-f0e1a79e73ea.png)


Simple LR



![image](https://user-images.githubusercontent.com/59621102/162633819-aafadb68-a1a8-4a48-8e4e-b4f235f10fe3.png)





Bayesian LR


All the graphs have 94% HDI(Highest Density Interval) which means credible interval has to remove 3% from each tail of distribution. Credible interval means the range containing a particular percentage of probable values




![image](https://user-images.githubusercontent.com/59621102/162633845-46379258-71f9-4da4-827b-4fdea900f84d.png)

![image](https://user-images.githubusercontent.com/59621102/162633865-f01ef305-0f64-46e5-86b8-3dfa6850963e.png)



LIMITATIONS:


I have successfully sampled the data using blackjax sampler(both nuts and hmc). But I was unable to use the sampler in the model due to which I used PyMC package.







