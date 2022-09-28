package net.sf.opendse.TensorDSE.JSON.Architecture;

public class ArchitectureJSON {
    Integer CPU_cores;
    Integer GPU_count;
    Integer TPU_count;

    public Integer getCPU_cores() {
        return CPU_cores;
    }

    public void setCPU_cores(Integer cPU_cores) {
        CPU_cores = cPU_cores;
    }

    public Integer getGPU_count() {
        return GPU_count;
    }

    public void setGPU_count(Integer gPU_count) {
        GPU_count = gPU_count;
    }

    public Integer getTPU_count() {
        return TPU_count;
    }

    public void setTPU_count(Integer tPU_count) {
        TPU_count = tPU_count;
    }
}
