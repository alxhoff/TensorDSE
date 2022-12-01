package net.sf.opendse.TensorDSE;

import gurobi.GRBVar;
import net.sf.opendse.model.Resource;
import net.sf.opendse.model.Task;

public class ILPTask {

    private Task task;
    private GRBVar grb_start_time;
    private GRBVar grb_finish_time;
    private GRBVar grb_execution_cost;
    private GRBVar grb_comm_cost;
    private Resource target_resource;
    private Double execution_cost;
    private Double comm_cost;

    public Double getExecution_cost() {
        return execution_cost;
    }

    public void setExecution_cost(Double execution_cost) {
        this.execution_cost = execution_cost;
    }

    public Task getTask() {
        return task;
    }

    public void setTask(Task task) {
        this.task = task;
    }

    public Resource getTarget_resource() {
        return target_resource;
    }

    public void setTarget_resource(Resource target_resource) {
        this.target_resource = target_resource;
    }

    public Double getComm_cost() {
        return comm_cost;
    }

    public void setComm_cost(Double comm_cost) {
        this.comm_cost = comm_cost;
    }

    public GRBVar getGrb_start_time() {
        return grb_start_time;
    }

    public void setGrb_start_time(GRBVar grb_start_time) {
        this.grb_start_time = grb_start_time;
    }

    public GRBVar getGrb_finish_time() {
        return grb_finish_time;
    }

    public void setGrb_finish_time(GRBVar grb_finish_time) {
        this.grb_finish_time = grb_finish_time;
    }

    public GRBVar getGrb_execution_cost() {
        return grb_execution_cost;
    }

    public void setGrb_execution_cost(GRBVar grb_execution_cost) {
        this.grb_execution_cost = grb_execution_cost;
    }

    public GRBVar getGrb_comm_cost() {
        return grb_comm_cost;
    }

    public void setGrb_comm_cost(GRBVar grb_comm_cost) {
        this.grb_comm_cost = grb_comm_cost;
    }

}
