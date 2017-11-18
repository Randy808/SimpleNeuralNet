
public class Main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		double[] inputs = {2.0,3.0,4.0,5.0,6.0};
		double[] targets = {1.0, 0.0, 1.0, 0.0, 1.0};
		int hiddenNeuronCount = 1;
		double learningRate = .01;
		
		Network nn;
		nn = new Network(inputs, targets, hiddenNeuronCount, learningRate);
		
		for(int i = 0 ; i < 1000000 ; i++){
			nn.forwardPropagate();
			nn.backPropagate();
		}
		nn.printOutputs();
		
		
		
	}

}
