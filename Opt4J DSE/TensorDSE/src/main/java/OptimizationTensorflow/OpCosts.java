package OptimizationTensorflow;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Enumeration;
import java.util.Hashtable;
/**
 * The {@code OpCosts} is a class that would be later used when 
 * defining the specification for initializing the cost of mapping. 
 * It allows updating the costs in case of further benchmarks.  
 * @param costfilepath 
 * 		The path to the csv file that is a summary of cosr results obtained from benchmarks and tests
 * 
 * 
 * @author z0040rwx
 *
 */
public class OpCosts {
	
	public Hashtable<String, Hashtable<String, Hashtable<String,Hashtable<String,Double[]>> >> OpCost ;
	public Double[] comCost; 
	
	public OpCosts(String costsfilepath) {
		this.OpCost = new Hashtable<String, Hashtable<String, Hashtable<String,Hashtable<String,Double[]>> >>(); 
		BufferedReader csvReader = null;
		try {
			csvReader = new BufferedReader(new FileReader(costsfilepath));
			String row;
			//Integer numrows_ncs= 0;
			//Integer numrows_coral = 0;
			Double min_ncs = Double.POSITIVE_INFINITY;
			Double min_coral= Double.POSITIVE_INFINITY;

			while((row=csvReader.readLine())!= null) {
				Hashtable<String, Double[]> shave_map_empty = new Hashtable<String, Double[]>();
				Hashtable<String, Hashtable<String,Double[]>> type_map_empty = new Hashtable<String, Hashtable<String,Double[]>>();
				Hashtable<String,Hashtable<String, Hashtable<String,Double[]>> > op_map_empty = new Hashtable<String, Hashtable<String, Hashtable<String,Double[]>> >();
				String[] data = row.split(",");
				Hashtable<String,Hashtable<String, Hashtable<String,Double[]>> > op_map = OpCost.putIfAbsent(data[1], op_map_empty);
				if (op_map == null) op_map = op_map_empty;
				Hashtable<String, Hashtable<String,Double[]>> type_map = op_map.putIfAbsent(data[0], type_map_empty);
				if (type_map == null) type_map = type_map_empty;
				Hashtable<String, Double[]> shave_map = type_map.putIfAbsent(data[2], shave_map_empty);
				if (shave_map == null) shave_map = shave_map_empty;
				shave_map.put(data[3], new Double[]{new Double(data[4]), new Double(data[5])});
					
				if ((data[1].equals("NCS2")) && (Double.parseDouble(data[5])!= 0.0) && (min_ncs > Double.parseDouble(data[5]))) {
					min_ncs = Double.parseDouble(data[5]);
				}
				
				if ((data[1].equals("coral edgetpu")) && (Double.parseDouble(data[5])!= 0.0) && (min_coral > Double.parseDouble(data[5]))) {
					min_coral = Double.parseDouble(data[5]);
				}
				/*
				if (data[1].equals("NCS2")) {
					sum_ncs = sum_ncs + new Double(data[5]);
					//numrows_ncs++; 
				}
				if (data[1].equals("coral edgetpu")) {
					sum_coral = sum_coral + new Double(data[5]);
					//numrows_coral++;
				}
				*/
			}

			//Enumeration<Hashtable<String, Hashtable<String,Hashtable<String,Double[]>> >> hw_keys = this.OpCost.keys();
			Enumeration<String> hw_keys = this.OpCost.keys();
			Enumeration<String> op_keys, type_keys, s_keys;
			String hw_key, op_key, type_key, s_key;
			Double[] intermed;
			while(hw_keys.hasMoreElements()){
				hw_key = hw_keys.nextElement();
				if (hw_key == "Desktop (CPU + GPU)") {
					continue;
				}
				else {
					op_keys = this.OpCost.get(hw_key).keys();
					while(op_keys.hasMoreElements()) {
						op_key = op_keys.nextElement();
						type_keys = this.OpCost.get(hw_key).get(op_key).keys();
						while (type_keys.hasMoreElements()) {
							type_key = type_keys.nextElement();
							s_keys = this.OpCost.get(hw_key).get(op_key).get(type_key).keys();
							while (s_keys.hasMoreElements()) {
								s_key = s_keys.nextElement();
								if (hw_key=="NCS2")  {
									intermed = this.OpCost.get(hw_key).get(op_key).get(type_key).get(s_key);
									if (intermed[1] != 0.0) this.OpCost.get(hw_key).get(op_key).get(type_key).replace(s_key, new Double[]{intermed[0], intermed[1] - min_ncs});
									
								}
								
								if (hw_key=="coral edgetpu") {
									intermed = this.OpCost.get(hw_key).get(op_key).get(type_key).get(s_key);
									if (intermed[1] != 0.0) this.OpCost.get(hw_key).get(op_key).get(type_key).replace(s_key, new Double[]{intermed[0], intermed[1] - min_coral});
									
								}
								
							}
						}
					}
				}
	            //String key = keys.nextElement();
	            //System.out.println("Value of "+key+" is: "+hm.get(key));
	        }
			
			this.comCost = new Double[]{min_ncs, min_coral};	
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (NumberFormatException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
	public Hashtable<String, Hashtable<String, Hashtable<String,Hashtable<String,Double[]>> >> getOpCost() {
		return this.OpCost;
	}

}
