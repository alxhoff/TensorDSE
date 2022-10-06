package net.sf.opendse.TensorDSE.JSON.Benchmark;

public class USB {
    Timing communication;
    Timing inference;
    Timing total;

    public Timing getCommunication() {
        return communication;
    }

    public void setCommunication(Timing communication) {
        this.communication = communication;
    }

    public Timing getInference() {
        return inference;
    }

    public void setInference(Timing inference) {
        this.inference = inference;
    }

    public Timing getTotal() {
        return total;
    }

    public void setTotal(Timing total) {
        this.total = total;
    }
}
