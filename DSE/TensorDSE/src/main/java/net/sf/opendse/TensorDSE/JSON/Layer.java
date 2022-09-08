package net.sf.opendse.TensorDSE.JSON;

import java.util.List;

public class Layer {
    String name;
    Paths path;
    List<Device> delegates;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Paths getPath() {
        return path;
    }

    public void setPath(Paths path) {
        this.path = path;
    }

    public List<Device> getDelegates() {
        return delegates;
    }

    public void setDelegates(List<Device> delegates) {
        this.delegates = delegates;
    }
}
