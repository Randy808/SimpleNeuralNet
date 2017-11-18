class Network{
	double[] inputs;
	HiddenNeuron[] hiddenNeurons;	
	HiddenNeuron[] outputNeurons;
	double[] outputs;
	double[] targets;
	double learningRate;
	double totalError;

	Network(double[] inputs, double[] targets, int hiddenNeuronCount, double learningRate){

		this.inputs = inputs;
		this.targets = targets;
		//initializeWeights();
		hiddenNeurons = new HiddenNeuron[hiddenNeuronCount];
		outputNeurons = new HiddenNeuron[targets.length];
		initializeHiddenNeurons();
		initializeOutputNeurons();
		this.learningRate = learningRate;
	}

	void initializeHiddenNeurons(){
		for(int i = 0 ; i < hiddenNeurons.length ; i++){
			hiddenNeurons[i] = new HiddenNeuron(inputs);
		}
	}

	void initializeOutputNeurons(){
		//resetting all the weights!
		System.out.println("\n\n\n");
		for(int i = 0 ; i < outputNeurons.length ; i++){
			outputNeurons[i] = new HiddenNeuron(hiddenNeurons.length);
		}
	}
	
	void setOutputInputs(double[] inputs){

		for(int i = 0 ; i < outputNeurons.length ; i++){
			outputNeurons[i].setInputs(inputs);
		}
	}

	double[] getHiddenNeuronOutputs(){
		double[] hiddenOutput = new double [hiddenNeurons.length];
		for(int i = 0 ; i < hiddenNeurons.length ; i++){
			hiddenOutput[i] = hiddenNeurons[i].activate();
		}
		return hiddenOutput;
	}

	double[] getOutputNeuronOutputs(){
		double[] outputs  = new double [outputNeurons.length];
		for(int i = 0 ; i < outputNeurons.length ; i++){
			outputs[i] = outputNeurons[i].activate();
		}
		return outputs;
	}

	void printOutputs(){
		for(int i = 0 ; i < outputs.length ; i++){
			System.out.println("Output" + i + ": " + outputs[i] );
		}
	}

	double calculateError(){
		double totalError = 0;
		
		for(int i = 0 ; i < outputs.length ; i++){
			totalError += Math.pow(targets[i] - outputs[i], 2);
		}
		totalError/=outputs.length;
		
		return totalError;
	}
	
	
	
	void forwardPropagate(){
		double[] hiddenNeuronOutputs = getHiddenNeuronOutputs();
		setOutputInputs(hiddenNeuronOutputs);
		this.outputs = getOutputNeuronOutputs();
		//printOutputs();
		this.totalError = calculateError();
		//System.out.println("Error: " + totalError);
		return;
	}
		
	void backPropagate(){
		double errorFromNextLayer = 0;
		
		for(int i = 0 ; i < outputNeurons.length ; i++){
			outputNeurons[i].error = outputs[i]*(1 - outputs[i])*(targets[i] - outputs[i]);
		}
		//calculate errors for hidden neurons
		for(int i = 0; i < hiddenNeurons.length ; i++ ){
			
			//for every neuron, reinitialize
			hiddenNeurons[i].error = 0;
			errorFromNextLayer = 0;
			
			for(int j = 0; j < outputNeurons.length ; j++ ){
				errorFromNextLayer +=  (outputNeurons[j].inputWeights[i] * outputNeurons[j].error);
			}
			hiddenNeurons[i].error +=  (1 - hiddenNeurons[i].output) * hiddenNeurons[i].output * errorFromNextLayer;
		}
		
		//update weights
		for(int i = 0; i < hiddenNeurons.length ; i++ ){
			hiddenNeurons[i].updateWeights(learningRate);
		}
		
		for(int i = 0; i < outputNeurons.length ; i++ ){
			outputNeurons[i].updateWeights(learningRate);
		}
		
		//finish
		return;
	}


}