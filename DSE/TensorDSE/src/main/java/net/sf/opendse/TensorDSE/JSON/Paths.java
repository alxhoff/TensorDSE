package net.sf.opendse.TensorDSE.JSON;


public class Paths {
    String cpu;
    String tpu;

    public String getQuantized() {
        return cpu;
    }

    public void setQuantized(String quantized) {
        this.cpu = quantized;
    }

    public String getCompiled() {
        return tpu;
    }

    public void setCompiled(String compiled) {
        this.tpu = compiled;
    }
}