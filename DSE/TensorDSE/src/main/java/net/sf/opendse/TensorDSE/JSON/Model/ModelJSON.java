package net.sf.opendse.TensorDSE.JSON.Model;

import java.util.List;

public class ModelJSON {
    String name;
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

    public Layer getLayerWithInputTensor(Integer tensor) {
        for (int i = 0; i < this.layers.size(); i++) {
            if (this.layers.get(i).getInputTensorArray().contains(tensor))
                return this.layers.get(i);
        }

        return null;
    }

    public Layer getLayerWithOutputTensor(Integer tensor) {
        for (int i = 0; i < this.layers.size(); i++) {
            if (this.layers.get(i).getOutputTensorArray().contains(tensor))
                return this.layers.get(i);
        }

        return null;
    }
}
