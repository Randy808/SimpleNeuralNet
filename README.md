# SimpleNeuralNet

This is a simple neural network architecture that uses 1 hidden layer with a configurable number of neurons.

```java
//declare your inputs for a single example
double[] inputs = {2.0,3.0,4.0,5.0,6.0};
//declare your target outputs for a single example
double[] targets = {1.0, 0.0, 1.0, 0.0, 1.0};

//set the desired number of hidden neurons
int hiddenNeuronCount = 1;
//set the desired learning rate
double learningRate = .01;

//declare the neural network
Network nn;
//initialize the network
nn = new Network(inputs, targets, hiddenNeuronCount, learningRate);

//I think these 2 are pretty self explanatory lol
nn.forwardPropagate();
nn.backPropagate();
```
