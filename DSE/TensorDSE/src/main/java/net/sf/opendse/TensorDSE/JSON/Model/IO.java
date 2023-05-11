package net.sf.opendse.TensorDSE.JSON.Model;

import java.util.List;

public class IO {
    List<Integer> shape;
    String type;
    Integer tensor;

    public List<Integer> getShape() {
        return shape;
    }

    public void setShape(List<Integer> shape) {
        this.shape = shape;
    }

    public String getType() {
        if (type == null)
            return "null";
            
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public Integer getTensor() {
        return tensor;
    }

    public void setTensor(Integer tensor) {
        this.tensor = tensor;
    }

    public Integer getShapeProduct() {
        Integer ret = 0;
        
        for (int i = 0; i < this.shape.size(); i++) {
            if (ret == 0)
                ret = this.shape.get(i);
            else
                ret *= this.shape.get(i);
        }

        return ret;
    }
}
