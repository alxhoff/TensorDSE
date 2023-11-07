package net.sf.opendse.TensorDSE.JSON.Model;

import java.util.ArrayList;
import java.util.List;

public class Model {
    String name;
    Double deadline;
    Integer starting_tensor;
    Integer finishing_tensor;
    List<Layer> layers;

    public List<Layer> getLayers() {
        return layers;
    }

    public void setLayers(List<Layer> layers) {
        this.layers = layers;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Integer getStarting_tensor() {
        return starting_tensor;
    }

    public void setStarting_tensor(Integer starting_tensor) {
        this.starting_tensor = starting_tensor;
    }

    public Integer getFinishing_tensor() {
        return finishing_tensor;
    }

    public void setFinishing_tensor(Integer finishing_tensor) {
        this.finishing_tensor = finishing_tensor;
    }

    public List<Layer> getLayersWithInputTensor(Integer tensor) {
        ArrayList<Layer> ret = new ArrayList<Layer>();
        for (int i = 0; i < this.layers.size(); i++) {
            if (this.layers.get(i).getInputTensorArray().contains(tensor))
                ret.add(this.layers.get(i));
        }

        return ret;
    }

    public Layer getLayerWithOutputTensor(Integer tensor) {
        for (int i = 0; i < this.layers.size(); i++) {
            if (this.layers.get(i).getOutputTensorArray().contains(tensor))
                return this.layers.get(i);
        }

        return null;
    }

    public Double getDeadline() {
        return deadline * 1000;
    }

    public void setDeadline(Double deadline) {
        this.deadline = deadline;
    }
}
