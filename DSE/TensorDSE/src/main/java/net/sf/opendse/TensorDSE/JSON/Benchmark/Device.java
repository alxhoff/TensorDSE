package net.sf.opendse.TensorDSE.JSON.Benchmark;

public class Device {
    String device;
    Input input;
    Double mean;
    Double median;
    Double standard_deviation;
    Double avg_absolute_deviation;
    String distribution;
    USB usb;

    public USB getUsb() {
        return usb;
    }

    public void setUsb(USB usb) {
        this.usb = usb;
    }

    public Double getMean() {
        return mean;
    }

    public void setMean(Double mean) {
        this.mean = mean;
    }

    public Double getMedian() {
        return median;
    }

    public void setMedian(Double median) {
        this.median = median;
    }

    public Double getStandard_deviation() {
        return standard_deviation;
    }

    public void setStandard_deviation(Double standard_deviation) {
        this.standard_deviation = standard_deviation;
    }

    public Double getAvg_absolute_deviation() {
        return avg_absolute_deviation;
    }

    public void setAvg_absolute_deviation(Double avg_absolute_deviation) {
        this.avg_absolute_deviation = avg_absolute_deviation;
    }

    public String getDistribution() {
        return distribution;
    }

    public void setDistribution(String distribution) {
        this.distribution = distribution;
    }

    public String getDevice() {
        return device;
    }

    public void setDevice(String device) {
        this.device = device;
    }

    public Input getInput() {
        return input;
    }

    public void setInput(Input input) {
        this.input = input;
    }
}

