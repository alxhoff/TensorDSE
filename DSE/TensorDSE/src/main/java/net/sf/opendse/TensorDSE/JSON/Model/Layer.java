package net.sf.opendse.TensorDSE.JSON.Model;

import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;

public class Layer {
    Integer index;
    String type;
    List<Arg> args;
    List<IO> inputs;
    List<IO> outputs;

    public Integer getIndex() {
        return index;
    }

    public void setIndex(Integer index) {
        this.index = index;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public List<Arg> getArgs() {
        return args;
    }

    public void setArgs(List<Arg> args) {
        this.args = args;
    }

    public List<IO> getInputs() {
        return inputs;
    }

    public void setInputs(List<IO> inputs) {
        this.inputs = inputs;
    }

    public List<IO> getOutputs() {
        return outputs;
    }

    public void setOutputs(List<IO> outputs) {
        this.outputs = outputs;
    }

    public List<Integer> getInputTensorArray() {
        List<Integer> ret = new ArrayList<>();

        ListIterator<IO> io_it = this.inputs.listIterator();
        for (; io_it.hasNext();) {
            ret.add(io_it.next().tensor);
        }

        return ret;
    }

    public String getInputTensorString() {
        return this.getInputTensorArray().toString();
    }

    public List<Integer> getOutputTensorArray() {
        List<Integer> ret = new ArrayList<>();

        ListIterator<IO> io_it = this.outputs.listIterator();
        for (; io_it.hasNext();) {
            ret.add(io_it.next().tensor);
        }

        return ret;
    }

    public String getOutputTensorString() {
        return this.getOutputTensorArray().toString();
    }

}
