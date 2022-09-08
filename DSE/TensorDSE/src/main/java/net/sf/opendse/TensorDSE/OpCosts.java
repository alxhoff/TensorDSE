package net.sf.opendse.TensorDSE;

import java.util.Hashtable;
import java.util.List;
import java.util.ArrayList;

import org.javatuples.Pair;
import com.google.common.io.Files;
import com.google.common.reflect.TypeToken;
import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonParser;
import com.google.gson.stream.JsonReader;
import net.sf.opendse.TensorDSE.JSON.BenchmarkJSON;
import net.sf.opendse.TensorDSE.JSON.Device;
import net.sf.opendse.TensorDSE.JSON.Layer;
import net.sf.opendse.TensorDSE.JSON.Model;
import java.io.*;
import java.lang.reflect.Type;

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

// top level:   device_type <--- op_map --->
// op_map:      operation_type <--- type_map --->
// type_map:    data_type, mean

public class OpCosts {

    public Hashtable<String, Hashtable<String, Hashtable<String, Double>>> operation_costs;
    public List<Double> communication_costs;

    public Hashtable CreateEmptyDeviceTypeTable() {
        return new Hashtable<String, Hashtable<String, Hashtable<String, Double>>>();
    }

    public Hashtable CreateEmptyOpTypeTable() {
        return new Hashtable<String, Hashtable<String, Double>>();
    }

    public Hashtable CreateEmptyDataTypeTable() {
        return new Hashtable<String, Double>();
    }

    public Hashtable GetOpTypeTable(String device_type) {
        return (Hashtable) this.operation_costs.get(device_type);
    }

    public Hashtable GetDataTypeTable(String device_type, String operation_type) {
        // return this.GetOpMap(device_type).get(operation_type);
        return (Hashtable) this.GetOpTypeTable(device_type).get(operation_type);
    }

    public Double GetMean(String device_type, String operation_type, String data_type) {
        return (Double) this.GetDataTypeTable(device_type, operation_type).get(data_type);
    }

    public BenchmarkJSON GetModelFromJSON(String json_file_path) {
        Gson gson = new Gson();
        BenchmarkJSON model = null;
        JsonReader jr;
        try {
            jr = new JsonReader(new FileReader(json_file_path));
            model = gson.fromJson(jr, BenchmarkJSON.class);
        } catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        return model;
    }

    public OpCosts(String costsfilepath) {
        this.communication_costs = new ArrayList();
        this.operation_costs = this.CreateEmptyDeviceTypeTable();
        BufferedReader csvReader = null;
        try {

            BenchmarkJSON benchmark = GetModelFromJSON(costsfilepath);

            for (Model model : benchmark.getModels()) {

                for (Layer layer : model.getLayers()) {

                    String operation_type = layer.getName();

                    for (Device deligate : layer.getDelegates()) {

                        String device_type = deligate.getDevice();
                        String data_type = deligate.getInput().getType();
                        Double mean = deligate.getMean();

                        if (!this.operation_costs.containsKey(device_type)) {
                            Hashtable new_op_map = this.CreateEmptyOpTypeTable();
                            this.operation_costs.put(device_type, new_op_map);
                        }

                        if (!this.GetOpTypeTable(device_type).containsKey(operation_type)) {
                            Hashtable new_type_map = this.CreateEmptyDataTypeTable();
                            this.GetOpTypeTable(device_type).put(operation_type, new_type_map);
                        }

                        if (!this.GetDataTypeTable(device_type, operation_type).containsKey(data_type)) {
                            this.GetDataTypeTable(device_type, operation_type).put(data_type, mean);
                        }
                    }
                }
            }

        } catch (NumberFormatException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    public Hashtable getOpCost() {
        return this.operation_costs;
    }

}
