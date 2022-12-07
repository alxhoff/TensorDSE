package net.sf.opendse.TensorDSE;

import java.util.Hashtable;
import org.javatuples.Pair;
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
    public Hashtable<String, Hashtable<String, Hashtable<String, Pair<Double, Double>>>> communication_costs;

    public Hashtable<String, Hashtable<String, Hashtable<String, Double>>> CreateEmptyDeviceTypeTable() {
        return new Hashtable<String, Hashtable<String, Hashtable<String, Double>>>();
    }

    public Hashtable<String, Hashtable<String, Hashtable<String, Pair<Double, Double>>>> CreateEmptyCommDeviceTypeTable() {
        return new Hashtable<String, Hashtable<String, Hashtable<String, Pair<Double, Double>>>>();
    }

    public Hashtable<String, Hashtable<String, Double>> CreateEmptyOpTypeTable() {
        return new Hashtable<String, Hashtable<String, Double>>();
    }

    public Hashtable<String, Hashtable<String, Pair<Double, Double>>> CreateEmptyCommOpTypeTable() {
        return new Hashtable<String, Hashtable<String, Pair<Double, Double>>>();
    }

    public Hashtable<String, Double> CreateEmptyDataTypeTable() {
        return new Hashtable<String, Double>();
    }

    public Hashtable<String, Pair<Double, Double>> CreateEmptyCommDataTypeTable() {
        return new Hashtable<String, Pair<Double, Double>>();
    }

    public Hashtable<String, Hashtable<String, Double>> GetOpTypeTable(String device_type) {

        if (!this.operation_costs.containsKey(device_type)) {
            Hashtable<String, Hashtable<String, Double>> new_op_map = this.CreateEmptyOpTypeTable();
            this.operation_costs.put(device_type, new_op_map);
        }


        return (Hashtable<String, Hashtable<String, Double>>) this.operation_costs.get(device_type);
    }

    public Hashtable<String, Hashtable<String, Pair<Double, Double>>> GetCommTypeTable(
            String device_type) {

        if (!this.communication_costs.containsKey(device_type)) {
            Hashtable<String, Hashtable<String, Pair<Double, Double>>> new_op_map =
                    this.CreateEmptyCommOpTypeTable();
            this.communication_costs.put(device_type, new_op_map);
        }

        return (Hashtable<String, Hashtable<String, Pair<Double, Double>>>) this.communication_costs
                .get(device_type);
    }

    public Hashtable<String, Double> GetOpDataTypeTable(String device_type, String operation_type) {

        Hashtable<String, Hashtable<String, Double>> op_type_table =
                this.GetOpTypeTable(device_type);

        if (!op_type_table.containsKey(operation_type)) {
            Hashtable<String, Double> new_type_map = this.CreateEmptyDataTypeTable();
            op_type_table.put(operation_type, new_type_map);
        }

        return (Hashtable<String, Double>) op_type_table.get(operation_type);

    }

    public Hashtable<String, Pair<Double, Double>> GetCommDataTypeTable(String device_type,
            String operation_type) {

        Hashtable<String, Hashtable<String, Pair<Double, Double>>> comm_type_table =
                this.GetCommTypeTable(device_type);

        if (!comm_type_table.containsKey(operation_type)) {
            Hashtable<String, Pair<Double, Double>> new_type_map =
                    this.CreateEmptyCommDataTypeTable();
            comm_type_table.put(operation_type, new_type_map);
        }

        return (Hashtable<String, Pair<Double, Double>>) comm_type_table.get(operation_type);

    }

    public Double GetOpCost(String device_type, String operation_type, String data_type) {

        Hashtable<String, Double> data_type_table =
                this.GetOpDataTypeTable(device_type, operation_type);

        if (!data_type_table.containsKey(data_type)) {
            data_type_table.put(data_type, Double.POSITIVE_INFINITY);
        }

        return (Double) data_type_table.get(data_type);
    }

    public Pair<Double, Double> GetCommCost(String device_type, String operation_type,
            String data_type) {

        Hashtable<String, Pair<Double, Double>> data_type_table =
                this.GetCommDataTypeTable(device_type, operation_type);

        if (!data_type_table.containsKey(data_type)) {
            data_type_table.put(data_type,
                    new Pair(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY));
        }

        return (Pair<Double, Double>) data_type_table.get(data_type);
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

        this.communication_costs = this.CreateEmptyCommDeviceTypeTable();
        this.operation_costs = this.CreateEmptyDeviceTypeTable();

        try {

            BenchmarkJSON benchmark = GetBenchmarkResultsFromJSON(cost_file_path);

            for (Model model : benchmark.getModels()) {

                for (Layer layer : model.getLayers()) {

                    String operation_type = layer.getName().toLowerCase();

                    for (Device deligate : layer.getDelegates()) {

                        String device_type = deligate.getDevice().toLowerCase();
                        Integer count = deligate.getCount();

                        String data_type;
                        Double mean_cost;
                        Double comm_send;
                        Double comm_recv;

                        if (count == 0) {
                            data_type = "null";
                            mean_cost = Double.POSITIVE_INFINITY;
                            comm_send = Double.POSITIVE_INFINITY;
                            comm_recv = Double.POSITIVE_INFINITY;
                        } else {
                            data_type = deligate.getInput().getType().toLowerCase();
                            mean_cost = deligate.getMean();
                            USB usb = deligate.getUsb();
                            comm_send = usb.getSend().getMean();
                            comm_recv = usb.getRecv().getMean();
                        }

                        this.GetOpDataTypeTable(device_type, operation_type).put(data_type,
                                mean_cost);

                        this.GetCommDataTypeTable(device_type, operation_type).put(data_type,
                                new Pair(comm_send, comm_recv));
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
