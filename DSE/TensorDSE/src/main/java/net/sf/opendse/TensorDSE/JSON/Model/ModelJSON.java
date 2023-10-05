package net.sf.opendse.TensorDSE.JSON.Model;

import java.util.ArrayList;
import java.util.List;

public class ModelJSON {
    List<Model> models;

    public List<Model> getModels() {
        return models;
    }

    public List<Double> getDeadlines() {

        List<Double> ret = new ArrayList<Double>();

        for(Model m: this.models)
            ret.add(m.getDeadline());

        return ret;
    }

    public void setModels(List<Model> models) {
        this.models = models;
    }
}
