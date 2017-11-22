import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.Scanner;

public class Main {

	public static void main(String[] args) {
		int epochNum = 11;
		int hiddenNeuronCount = 10;
		
		
		Scanner s = null;
		double[] inputs = {2.0,3.0,4.0,5.0,6.0};
		double[] targets = {1.0};
		double learningRate = .01;
	
		Network nn = null;
		
		
		
		/*
		try{
			FileInputStream fileIn = new FileInputStream("network");
	         ObjectInputStream in = new ObjectInputStream(fileIn);
	         nn = (Network) in.readObject();
	         in.close();
	         fileIn.close();
        } 
        catch (Exception e)
        { 
            e.printStackTrace(); 
        }
        */
		
		try {
			s = new Scanner(new File("data.csv"));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			System.out.println("FAILED");
			e.printStackTrace();
		}
		
		s.nextLine();
		
		System.out.print("Loading in File");
		int j = 0;
		String[] tExtract;
		String[] stringData;
		double data[][] = new double[11500][];
		double t[] = new double[11500];
		
		
		while(s.hasNextLine()){
			String raw =  s.nextLine();
			char[] rawChar = raw.toCharArray();
			rawChar[rawChar.length - 2] = 'P';
			raw = String.valueOf(rawChar);
			//System.out.println(raw);
			
			t[j] = Double.parseDouble(raw.split("P")[1]);
			if(t[j] > 1){
				t[j] = 0;
			}
			else{
				t[j] = 1;
			}
			System.out.println(t[j]);

			
			stringData = raw.split("P")[0].split(",");
			data[j] = Arrays.stream(stringData)
                    .mapToDouble(Double::parseDouble)
                    .toArray();
			
			//System.out.println(s.nextLine());
			if(j % 100000 == 0)
				System.out.print(".");
			j++;
		}
		
		System.out.println("\nFinished Reading in File.");
		
		
		
		
		nn = new Network(inputs, targets, hiddenNeuronCount, learningRate);
		
		
		for(int epochs = 0 ; epochs < epochNum; epochs++){
			for(int k = 0 ; k < data.length/2 ; k++){
				double[] tr = {t[k]};
				nn.reinitialize(data[k], tr, learningRate);
				nn.forwardPropagate();
				nn.backPropagate();
			}
		}
		//nn.printOutputs();
		
		//nn.saveNetworkToFile("network");
		
		
		double wrong = 0;
		double total = 11500;
		for(int k = 0 ; k < data.length/2 ; k++){
			double[] tr = {t[k]};
			nn.reinitialize(data[k], tr, learningRate);
			nn.forwardPropagate();
			if(Math.round(nn.outputs[0]) != t[k]){
				wrong++;
			}
			
		}
		System.out.println( "ACCURACY: " + ((total - wrong) /total) + "%" );
		
		
		
    
        
		//nn.forwardPropagate();
		//nn.printOutputs();
		
		
		
		
		
		
		
		
		//System.out.println(s.nextLine());
		
	}

}
