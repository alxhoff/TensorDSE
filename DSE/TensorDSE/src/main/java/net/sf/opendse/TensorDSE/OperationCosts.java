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

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The {@code OpCosts} is a class that would be later used when defining the
 * specification for
 * initializing the cost of mapping. It allows updating the costs in case of
 * further benchmarks.
 *
 * @param costfilepath The path to the csv file that is a summary of cosr
 *                     results obtained from
 *                     benchmarks and tests
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
    public String model_name = null;

    /**
     * @return Hashtable<String, Hashtable<String, Hashtable<String, Double>>>
     */
    public Hashtable<String, Hashtable<String, Hashtable<String, Double>>> CreateEmptyDeviceTypeTable() {
        return new Hashtable<String, Hashtable<String, Hashtable<String, Double>>>();
    }

    /**
     * @return Hashtable<String, Hashtable<String, Hashtable<String, Pair<Double,
     *         Double>>>>
     */
    public Hashtable<String, Hashtable<String, Hashtable<String, Pair<Double, Double>>>> CreateEmptyCommDeviceTypeTable() {
        return new Hashtable<String, Hashtable<String, Hashtable<String, Pair<Double, Double>>>>();
    }

    /**
     * @return Hashtable<String, Hashtable<String, Double>>
     */
    public Hashtable<String, Hashtable<String, Double>> CreateEmptyOpTypeTable() {
        return new Hashtable<String, Hashtable<String, Double>>();
    }

    /**
     * @return Hashtable<String, Hashtable<String, Pair<Double, Double>>>
     */
    public Hashtable<String, Hashtable<String, Pair<Double, Double>>> CreateEmptyCommOpTypeTable() {
        return new Hashtable<String, Hashtable<String, Pair<Double, Double>>>();
    }

    /**
     * @return Hashtable<String, Double>
     */
    public Hashtable<String, Double> CreateEmptyDataTypeTable() {
        return new Hashtable<String, Double>();
    }

    /**
     * @return Hashtable<String, Pair<Double, Double>>
     */
    public Hashtable<String, Pair<Double, Double>> CreateEmptyCommDataTypeTable() {
        return new Hashtable<String, Pair<Double, Double>>();
    }

    /**
     * @param device_type
     * @return Hashtable<String, Hashtable<String, Double>>
     */
    public Hashtable<String, Hashtable<String, Double>> GetOpTypeTable(String device_type) {

        if (!this.operation_costs.containsKey(device_type)) {
            Hashtable<String, Hashtable<String, Double>> new_op_map = this.CreateEmptyOpTypeTable();
            this.operation_costs.put(device_type, new_op_map);
        }

        return (Hashtable<String, Hashtable<String, Double>>) this.operation_costs.get(device_type);
    }

    /**
     * @param device_type
     * @return Hashtable<String, Hashtable<String, Pair<Double, Double>>>
     */
    public Hashtable<String, Hashtable<String, Pair<Double, Double>>> GetCommTypeTable(
            String device_type) {

        if (!this.communication_costs.containsKey(device_type)) {
            Hashtable<String, Hashtable<String, Pair<Double, Double>>> new_op_map = this.CreateEmptyCommOpTypeTable();
            this.communication_costs.put(device_type, new_op_map);
        }

        return (Hashtable<String, Hashtable<String, Pair<Double, Double>>>) this.communication_costs
                .get(device_type);
    }

    /**
     * @param device_type
     * @param operation_type
     * @return Hashtable<String, Double>
     */
    public Hashtable<String, Double> GetOpDataTypeTable(String device_type, String operation_type) {

        Hashtable<String, Hashtable<String, Double>> op_type_table = this.GetOpTypeTable(device_type);

        if (!op_type_table.containsKey(operation_type)) {
            Hashtable<String, Double> new_type_map = this.CreateEmptyDataTypeTable();
            op_type_table.put(operation_type, new_type_map);
        }

        return (Hashtable<String, Double>) op_type_table.get(operation_type);
    }

    /**
     * @param device_type
     * @param operation_type
     * @return Hashtable<String, Pair<Double, Double>>
     */
    public Hashtable<String, Pair<Double, Double>> GetCommDataTypeTable(String device_type,
            String operation_type) {

        Hashtable<String, Hashtable<String, Pair<Double, Double>>> comm_type_table = this.GetCommTypeTable(device_type);

        if (!comm_type_table.containsKey(operation_type)) {
            Hashtable<String, Pair<Double, Double>> new_type_map = this.CreateEmptyCommDataTypeTable();
            comm_type_table.put(operation_type, new_type_map);
        }

        return (Hashtable<String, Pair<Double, Double>>) comm_type_table.get(operation_type);
    }

    /**
     * @param device_type
     * @param operation_type
     * @param data_type
     * @return Double
     */
    public Double GetOpCost(String device_type, String operation_type, String data_type) {

        Hashtable<String, Double> data_type_table = this.GetOpDataTypeTable(device_type, operation_type);

        if (!data_type_table.containsKey(data_type)) {
            data_type_table.put(data_type, 10000.0);
        }

        return (Double) data_type_table.get(data_type);
    }

    /**
     * @param device_type
     * @param operation_type
     * @param data_type
     * @return Pair<Double, Double>
     */
    public Pair<Double, Double> GetCommCost(String device_type, String operation_type,
            String data_type) {

        Hashtable<String, Pair<Double, Double>> data_type_table = this.GetCommDataTypeTable(device_type,
                operation_type);

        if (!data_type_table.containsKey(data_type)) {
            data_type_table.put(data_type, new Pair<>(10000.0, 10000.0));
        }

        return (Pair<Double, Double>) data_type_table.get(data_type);
    }

    /**
     * @param json_file_path
     * @return BenchmarkJSON
     */
    public BenchmarkJSON GetProfilingResultsFromJSON(String json_file_path) {

        Gson gson = new Gson();
        BenchmarkJSON model = null;

        try {
            System.out.println("Working Directory: " + System.getProperty("user.dir"));
            JsonReader jr = new JsonReader(new FileReader(json_file_path));
            model = gson.fromJson(jr, BenchmarkJSON.class);
        } catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        return model;
    }

    public static String removeMatchingGroups(String input) {
        Pattern pattern = Pattern.compile("(?:\\d+_)+\\d+");
        Matcher matcher = pattern.matcher(input);

        // Replace matching groups with an empty string
        String output = matcher.replaceAll("");

        if (output.endsWith("_")) {
            output = output.substring(0, output.length() - 1);
        }

        return output;
    }

    public OperationCosts(String profiling_costs_directory_path, String model_name) throws Exception {

        this.communication_costs = this.CreateEmptyCommDeviceTypeTable();
        this.operation_costs = this.CreateEmptyDeviceTypeTable();
        this.model_name = model_name;

        try {

            BenchmarkJSON benchmark = GetProfilingResultsFromJSON(
                    String.format("%s/%s.json", profiling_costs_directory_path, model_name));

            for (Model model : benchmark.getModels()) {

                for (Layer layer : model.getLayers()) {

                    // Pattern pattern = Pattern.compile("([a-zA-Z]+_\\d+[a-zA-Z]+)");

                    // Matcher matcher = pattern.matcher();

                    String operation_type = removeMatchingGroups(layer.getName().toLowerCase());
                    // while (matcher.find()) {
                    //     operation_type = matcher.group(1);
                    
                    if (operation_type == null){
                        throw new Exception(
                                String.format("Couldnt extract layer name from %s", layer.getName().toLowerCase()));
                    }
                    
                    for (Device deligate : layer.getDelegates()) {

                        String device_type = deligate.getDevice().toLowerCase();
                        Integer count = deligate.getCount();

                        String data_type;
                        Double mean_exec_cost;
                        Double comm_send_cost;
                        Double comm_recv_cost;

                        if (count == 0) {
                            data_type = "null";
                            mean_exec_cost = 10000.0;
                            comm_send_cost = 10000.0;
                            comm_recv_cost = 10000.0;
                        } else {
                            data_type = deligate.getInput().getType().toLowerCase();
                            if(device_type.equals("tpu"))
                                mean_exec_cost = deligate.getUsb().getInference().getMean();
                            else
                                mean_exec_cost = deligate.getMean();
                            USB usb = deligate.getUsb();
                            comm_send_cost = usb.getSend().getMean();
                            comm_recv_cost = usb.getRecv().getMean();
                        }

                        this.GetOpDataTypeTable(device_type, operation_type).put(data_type,
                                mean_exec_cost);

                        Pair<Double, Double> comm_costs = new Pair<Double, Double>(comm_send_cost, comm_recv_cost);
                        this.GetCommDataTypeTable(device_type, operation_type).put(data_type,
                                comm_costs);
                    }
                }
            }

        } catch (NumberFormatException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    /**
     * @return Hashtable<String, Hashtable<String, Hashtable<String, Double>>>
     */
    public Hashtable<String, Hashtable<String, Hashtable<String, Double>>> getOpCost() {
        return this.operation_costs;
    }

}
