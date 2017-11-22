import java.util.Random;
import java.io.Serializable;
import java.lang.Math;

public class HiddenNeuron implements Serializable{
	double[] inputs;
	double[] inputWeights;
	double error;
	double output;

	HiddenNeuron(double[] inputs){
		this.inputs = inputs;
		//outputs = new double[outputSize];
		inputWeights = new double[inputs.length];
		initializeWeights();

	}
	
	HiddenNeuron(int inputSize){
		//outputs = new double[outputSize];
		inputWeights = new double[inputSize];
		initializeWeights();

	}

	void setInputs(double[] inputs){
		this.inputs = inputs;
	}
	void initializeWeights(){
		Random randomWeightGenerator = new Random();
		for(int i = 0 ; i < inputWeights.length ; i++){
			inputWeights[i] = randomWeightGenerator.nextFloat();
		}
		return;
	}

	double weightedSum(){
		double sum = 0;
		for(int i = 0 ; i < inputs.length ; i++){
			sum+=inputs[i]*inputWeights[i];
		}
		return sum;
	}

	double activate(){
		this.output = 1/( 1 + Math.exp(-weightedSum()) );
		return output;
	}
	
	void updateWeights(double learningRate){
		for(int i = 0 ; i < inputs.length ; i++){
			inputWeights[i]+=learningRate*this.error*this.inputs[i];
		}
	}


}