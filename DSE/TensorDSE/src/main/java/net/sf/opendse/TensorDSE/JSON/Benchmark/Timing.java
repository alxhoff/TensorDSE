package net.sf.opendse.TensorDSE.JSON.Benchmark;

public class Timing {
    Double mean;
    Double median;
    Double standard_deviation;
    Double avg_absolute_deviation;

    public Timing() {
        this.mean = 0.0;
        this.median = 0.0;
        this.standard_deviation = 0.0;
        this.avg_absolute_deviation = 0.0;
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


}
