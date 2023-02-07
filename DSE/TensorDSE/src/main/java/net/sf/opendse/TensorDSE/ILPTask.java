package net.sf.opendse.TensorDSE;

import java.util.ArrayList;
import java.util.HashMap;
import org.javatuples.Pair;
import gurobi.GRBVar;
import net.sf.opendse.model.Resource;
import net.sf.opendse.model.Task;

public class ILPTask {

    private String ID;
    private Task task;
    private GRBVar grb_start_time;
    private GRBVar grb_finish_time;
    private GRBVar grb_execution_cost;
    private GRBVar grb_comm_cost;
    private Resource target_resource;
    private ArrayList<Resource> target_resources;
    private Double execution_cost;
    private HashMap<Resource, Double> execution_costs;
    private Double send_cost;
    private HashMap<Resource, Double> send_costs;
    private Double recv_cost;
    private HashMap<Resource, Double> recv_costs;

    
    /** 
     * @return ArrayList<Resource>
     */
    public ArrayList<Resource> getTarget_resources() {
        return target_resources;
    }

    
    /** 
     * @param target_resources
     */
    public void setTarget_resources(ArrayList<Resource> target_resources) {
        this.target_resources = target_resources;
    }

    
    /** 
     * @return Double
     */
    public Double getExecution_cost() {
        return execution_cost;
    }

    
    /** 
     * @param execution_cost
     */
    public void setExecution_cost(Double execution_cost) {
        this.execution_cost = execution_cost;
    }

    
    /** 
     * @return Task
     */
    public Task getTask() {
        return task;
    }

    
    /** 
     * @param task
     */
    public void setTask(Task task) {
        this.task = task;
    }

    
    /** 
     * @return Resource
     */
    public Resource getTarget_resource() {
        return target_resource;
    }

    
    /** 
     * @param target_resource
     */
    public void setTarget_resource(Resource target_resource) {
        this.target_resource = target_resource;
    }


    
    /** 
     * @return GRBVar
     */
    public GRBVar getGrb_start_time() {
        return grb_start_time;
    }

    
    /** 
     * @param grb_start_time
     */
    public void setGrb_start_time(GRBVar grb_start_time) {
        this.grb_start_time = grb_start_time;
    }

    
    /** 
     * @return GRBVar
     */
    public GRBVar getGrb_finish_time() {
        return grb_finish_time;
    }

    
    /** 
     * @param grb_finish_time
     */
    public void setGrb_finish_time(GRBVar grb_finish_time) {
        this.grb_finish_time = grb_finish_time;
    }

    
    /** 
     * @return GRBVar
     */
    public GRBVar getGrb_execution_cost() {
        return grb_execution_cost;
    }

    
    /** 
     * @param grb_execution_cost
     */
    public void setGrb_execution_cost(GRBVar grb_execution_cost) {
        this.grb_execution_cost = grb_execution_cost;
    }

    
    /** 
     * @return GRBVar
     */
    public GRBVar getGrb_comm_cost() {
        return grb_comm_cost;
    }

    
    /** 
     * @param grb_comm_cost
     */
    public void setGrb_comm_cost(GRBVar grb_comm_cost) {
        this.grb_comm_cost = grb_comm_cost;
    }

    
    /** 
     * @return Double
     */
    public Double getSend_cost() {
        return send_cost;
    }

    
    /** 
     * @param send_cost
     */
    public void setSend_cost(Double send_cost) {
        this.send_cost = send_cost;
    }

    
    /** 
     * @return Double
     */
    public Double getRecv_cost() {
        return recv_cost;
    }

    
    /** 
     * @param recv_cost
     */
    public void setRecv_cost(Double recv_cost) {
        this.recv_cost = recv_cost;
    }

    
    /** 
     * @return HashMap<Resource, Double>
     */
    public HashMap<Resource, Double> getExecution_costs() {
        return execution_costs;
    }

    
    /** 
     * @param execution_costs
     */
    public void setExecution_costs(HashMap<Resource, Double> execution_costs) {
        this.execution_costs = execution_costs;
    }

    
    /** 
     * @return HashMap<Resource, Double>
     */
    public HashMap<Resource, Double> getSend_costs() {
        return send_costs;
    }

    
    /** 
     * @param send_costs
     */
    public void setSend_costs(HashMap<Resource, Double> send_costs) {
        this.send_costs = send_costs;
    }

    
    /** 
     * @return HashMap<Resource, Double>
     */
    public HashMap<Resource, Double> getRecv_costs() {
        return recv_costs;
    }

    
    /** 
     * @param recv_costs
     */
    public void setRecv_costs(HashMap<Resource, Double> recv_costs) {
        this.recv_costs = recv_costs;
    }

    
    /** 
     * @return String
     */
    public String getID() {
        return ID;
    }

    
    /** 
     * @param iD
     */
    public void setID(String iD) {
        ID = iD;
    }

}
