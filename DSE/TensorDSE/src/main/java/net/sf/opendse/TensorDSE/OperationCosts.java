package net.sf.opendse.TensorDSE;

import java.util.Hashtable;

import com.google.gson.Gson;
import com.google.gson.stream.JsonReader;
import net.sf.opendse.TensorDSE.JSON.Benchmark.BenchmarkJSON;
import net.sf.opendse.TensorDSE.JSON.Benchmark.Model;
import net.sf.opendse.TensorDSE.JSON.Benchmark.USB;
import net.sf.opendse.TensorDSE.JSON.Benchmark.Device;
import net.sf.opendse.TensorDSE.JSON.Benchmark.Layer;
import java.io.*;

/**
 * The {@code OpCosts} is a class that would be later used when defining the specification for
 * initializing the cost of mapping. It allows updating the costs in case of further benchmarks.
 * 
 * @param costfilepath The path to the csv file that is a summary of cosr results obtained from
 *        benchmarks and tests
 * 
 * 
 * @author Ines Ben Hmida
 * @author Alex Hoffman
 *
 */

// top level: device_type <--- op_map --->
// op_map: operation_type <--- type_map --->
// type_map: data_type, mean

public class OperationCosts {

    public Hashtable<String, Hashtable<String, Hashtable<String, Double>>> operation_costs;
    public Hashtable<String, Hashtable<String, Hashtable<String, Double>>> communication_costs;

    public Hashtable<String, Hashtable<String, Hashtable<String, Double>>> CreateEmptyDeviceTypeTable() {
        return new Hashtable<String, Hashtable<String, Hashtable<String, Double>>>();
    }

    public Hashtable<String, Hashtable<String, Double>> CreateEmptyOpTypeTable() {
        return new Hashtable<String, Hashtable<String, Double>>();
    }

    public Hashtable<String, Double> CreateEmptyDataTypeTable() {
        return new Hashtable<String, Double>();
    }

    public Hashtable<String, Hashtable<String, Double>> GetOpTypeTable(String device_type) {

        if (!this.operation_costs.containsKey(device_type)) {
            Hashtable<String, Hashtable<String, Double>> new_op_map = this.CreateEmptyOpTypeTable();
            this.operation_costs.put(device_type, new_op_map);
        }


        return (Hashtable<String, Hashtable<String, Double>>) this.operation_costs.get(device_type);
    }

    public Hashtable<String, Hashtable<String, Double>> GetCommTypeTable(String device_type) {

        if (!this.communication_costs.containsKey(device_type)) {
            Hashtable<String, Hashtable<String, Double>> new_op_map = this.CreateEmptyOpTypeTable();
            this.communication_costs.put(device_type, new_op_map);
        }

        return (Hashtable<String, Hashtable<String, Double>>) this.communication_costs
                .get(device_type);
    }

    public Hashtable<String, Double> GetOpDataTypeTable(String device_type, String operation_type) {

        if (!this.GetOpTypeTable(device_type).containsKey(operation_type)) {
            Hashtable<String, Double> new_type_map = this.CreateEmptyDataTypeTable();
            this.GetOpTypeTable(device_type).put(operation_type, new_type_map);
        }

        return (Hashtable<String, Double>) this.GetOpTypeTable(device_type).get(operation_type);

    }

    public Hashtable<String, Double> GetCommDataTypeTable(String device_type,
            String operation_type) {

        if (!this.GetCommTypeTable(device_type).containsKey(operation_type)) {
            Hashtable<String, Double> new_type_map = this.CreateEmptyDataTypeTable();
            this.GetCommTypeTable(device_type).put(operation_type, new_type_map);
        }

        return (Hashtable<String, Double>) this.GetCommTypeTable(device_type).get(operation_type);

    }

    public Double GetOpCost(String device_type, String operation_type, String data_type) {

        if (!this.GetOpDataTypeTable(device_type, operation_type).containsKey(data_type)) {
            this.GetOpDataTypeTable(device_type, operation_type).put(data_type, 0.0);
        }

        return (Double) this.GetOpDataTypeTable(device_type, operation_type).get(data_type);
    }

    public Double GetCommCost(String device_type, String operation_type, String data_type) {

        if (!this.GetCommDataTypeTable(device_type, operation_type).containsKey(data_type)) {
            this.GetCommDataTypeTable(device_type, operation_type).put(data_type, 0.0);
        }

        return (Double) this.GetCommDataTypeTable(device_type, operation_type).get(data_type);
    }

    public BenchmarkJSON GetBenchmarkResultsFromJSON(String json_file_path) {

        Gson gson = new Gson();
        BenchmarkJSON model = null;

        try {
            JsonReader jr = new JsonReader(new FileReader(json_file_path));
            model = gson.fromJson(jr, BenchmarkJSON.class);
        } catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        return model;
    }

    public OperationCosts(String cost_file_path) {

        this.communication_costs = this.CreateEmptyDeviceTypeTable();
        this.operation_costs = this.CreateEmptyDeviceTypeTable();

        try {

            BenchmarkJSON benchmark = GetBenchmarkResultsFromJSON(cost_file_path);

            for (Model model : benchmark.getModels()) {

                for (Layer layer : model.getLayers()) {

                    String operation_type = layer.getName().toLowerCase();

                    for (Device deligate : layer.getDelegates()) {

                        String device_type = deligate.getDevice().toLowerCase();
                        String data_type = deligate.getInput().getType().toLowerCase();
                        Double mean_cost = deligate.getMean();
                        USB usb = deligate.getUsb();
                        Double comm_send = usb.getSend();
                        Double comm_recv = usb.getRecv();


                        this.GetOpDataTypeTable(device_type, operation_type).put(data_type,
                                mean_cost);

                        this.GetCommDataTypeTable(device_type, operation_type).put(data_type,
                                comm_send + comm_recv);
                    }
                }
            }

        } catch (NumberFormatException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    public Hashtable<String, Hashtable<String, Hashtable<String, Double>>> getOpCost() {
        return this.operation_costs;
    }

}
