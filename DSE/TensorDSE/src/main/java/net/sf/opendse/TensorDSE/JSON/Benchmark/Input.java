package net.sf.opendse.TensorDSE.JSON.Benchmark;

import java.util.List;

public class Input {
    List<Integer> shape;
    String type;
    public List<Integer> getShape() {
        return shape;
    }
    public void setShape(List<Integer> shape) {
        this.shape = shape;
    }
    public String getType() {
        return type;
    }
    public void setType(String type) {
        this.type = type;
    }
}
