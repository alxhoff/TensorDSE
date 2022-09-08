package net.sf.opendse.TensorDSE.JSON;

import java.util.List;

public class Model {
    String name;
    Integer runs;
    List<Layer> layers;

    public Integer getRuns() {
        return runs;
    }

    public void setRuns(Integer runs) {
        this.runs = runs;
    }

    public List<Layer> getLayers() {
        return layers;
    }

    public void setLayers(List<Layer> layers) {
        this.layers = layers;
    }
}

