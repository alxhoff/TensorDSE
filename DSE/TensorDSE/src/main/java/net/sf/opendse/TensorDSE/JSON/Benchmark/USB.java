package net.sf.opendse.TensorDSE.JSON.Benchmark;

import org.apache.xalan.xsltc.trax.TrAXFilter;

public class USB {
    Timing send;
    Timing recv;
    Timing communication;
    Timing inference;
    Timing total;

    public Timing getCommunication() {
        if (communication == null)
            this.communication = new Timing();
        return communication;
    }

    public void setCommunication(Timing communication) {
        this.communication = communication;
    }

    public Timing getInference() {
        if (inference == null)
            this.inference = new Timing();
        return inference;
    }

    public void setInference(Timing inference) {
        this.inference = inference;
    }

    public Timing getTotal() {
        if (total == null)
            this.total = new Timing();
        return total;
    }

    public void setTotal(Timing total) {
        this.total = total;
    }

    public Timing getSend() {
        if (send == null)
            this.send = new Timing();
        return send;
    }

    public void setSend(Timing send) {
        this.send = send;
    }

    public Timing getRecv() {
        if (recv == null)
            this.recv = new Timing();
        return recv;
    }

    public void setRecv(Timing recv) {
        this.recv = recv;
    }
}
