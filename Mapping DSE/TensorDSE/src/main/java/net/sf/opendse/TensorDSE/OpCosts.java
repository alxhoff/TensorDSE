package net.sf.opendse.TensorDSE;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Hashtable;

import org.javatuples.Pair;

/**
 * The {@code OpCosts} is a class that would be later used when
 * defining the specification for initializing the cost of mapping.
 * It allows updating the costs in case of further benchmarks.
 * 
 * @param costfilepath
 *                     The path to the csv file that is a summary of cosr
 *                     results obtained from benchmarks and tests
 * 
 * 
 * @author z0040rwx
 *
 */

// device_type <min_val, sum, <--- op_map --->
// operation_type <--- type_map --->
// data_type <--- shave_map --->
// Index, {a, b}

public class OpCosts {

    public Hashtable<String, Pair<Double, Hashtable<String, Hashtable<String, Hashtable<String, Double[]>>>>> OpCost;
    public Double[] comCost;

    public Hashtable CreateEmptyMap() {
        return new Hashtable<String, Pair<Double, Hashtable<String, Hashtable<String, Hashtable<String, Double[]>>>>>();
    }

    public Hashtable CreateEmptyOpMap() {
        return new Hashtable<String, Hashtable<String, Hashtable<String, Double[]>>>();
    }

    public Hashtable CreateEmptyTypeMap() {
        return new Hashtable<String, Hashtable<String, Double[]>>();
    }

    public Hashtable CreateEmptyShaveMap() {
        return new Hashtable<String, Double[]>();
    }

    public Double GetOpCoeffucuentA(String device_type, String operation_type, String data_type, String index) {
        return this.OpCost.get(device_type).getValue1().get(operation_type).get(data_type).get(index)[0];
    }

    public Double GetOpCoeffucuentB(String device_type, String operation_type, String data_type, String index) {
        return this.OpCost.get(device_type).getValue1().get(operation_type).get(data_type).get(index)[1];
    }

    public Double GetOpMin(String device_type) {
        return (Double) this.OpCost.get(device_type).getValue0();
    }

    public void SetOpMin(String device_type, Double value) {
        this.OpCost.put(device_type, this.OpCost.get(device_type).setAt0(value));
    }

    public Hashtable GetOpMap(String device_type) {
        return (Hashtable) this.OpCost.get(device_type).getValue1();
    }

    public Hashtable GetTypeMap(String device_type, String operation_type) {
        // return this.GetOpMap(device_type).get(operation_type);
        return (Hashtable) this.OpCost.get(device_type).getValue1().get(operation_type);
    }

    public Hashtable GetShaveMap(String device_type, String operation_type, String data_type) {
        return (Hashtable) this.OpCost.get(device_type).getValue1().get(operation_type).get(data_type);
    }

    public OpCosts(String costsfilepath) {
        this.OpCost = this.CreateEmptyMap();
        BufferedReader csvReader = null;
        try {
            csvReader = new BufferedReader(new FileReader(costsfilepath));
            String row;

            while ((row = csvReader.readLine()) != null) {

                String[] data = row.split(",");

                // Human readable
                String operation_type = data[0];
                String device_type = data[1];
                String data_type = data[2];
                String index = data[3];
                Double a = Double.valueOf(data[4]);
                Double b = Double.valueOf(data[5]);

                // think this is just populating items for each device type
                if (!this.OpCost.containsKey(device_type)) {
                    Hashtable new_op_map = this.CreateEmptyOpMap();
                    this.OpCost.put(device_type, Pair.with(Double.POSITIVE_INFINITY, new_op_map));
                }

                if (!this.GetOpMap(device_type).containsKey(operation_type)) {
                    Hashtable new_type_map = this.CreateEmptyTypeMap();
                    this.GetOpMap(device_type).put(operation_type, new_type_map);
                }

                if (!this.GetTypeMap(device_type, operation_type).containsKey(data_type)) {
                    Hashtable new_shave_map = this.CreateEmptyShaveMap();
                    this.GetTypeMap(device_type, operation_type).put(data_type, new_shave_map);
                }

                if (!this.GetShaveMap(device_type, operation_type, data_type).containsKey(index))
                    this.GetShaveMap(device_type, operation_type, data_type).put(index, new Double[] { a, b });

                // Finding minimum value
                if (b < this.GetOpMin(device_type))
                    this.SetOpMin(device_type, b);
            }


            //Now that we have the minimum values for each device this value should be subtracted from each b value in OpCosts
            this.OpCost.forEach((device_type, min_sum_op_map) -> {
                Double device_min = this.GetOpMin(device_type);
                min_sum_op_map.getValue1().forEach((operation_type, type_map) -> {
                    type_map.forEach((data_type, shave_map) -> {
                        shave_map.forEach((index, coefficients) -> {
                            Double new_val = this.GetOpCoeffucuentB(device_type, operation_type, data_type, index) - device_min;
                            this.OpCost.get(device_type).getValue1().get(operation_type).get(data_type).get(index)[1] = new_val;
                        });
                    });
                });
            });

            // this.comCost = new Double[] { this.OpCost, min_coral };

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

    public Hashtable getOpCost() {
        return this.OpCost;
    }

}
