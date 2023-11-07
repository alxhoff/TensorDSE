package net.sf.opendse.TensorDSE;

import java.util.ArrayList;
import java.util.HashMap;
import gurobi.GRB;
import gurobi.GRBException;
import gurobi.GRBModel;
import gurobi.GRBVar;
import net.sf.opendse.model.Resource;
import net.sf.opendse.model.Task;

public class ILPTask {

    private String ID;
    private Task task;
    private Integer task_index;
    private ArrayList<Integer> prev_task_indexes = new ArrayList<Integer>();
    private Integer model_index;
    private GRBModel grb_model;
    private GRBVar start_time;
    private GRBVar finish_time;
    private GRBVar total_execution_cost;
    private GRBVar total_comm_cost;
    private GRBVar total_sending_comm_cost;
    private GRBVar total_receiving_comm_cost;
    private GRBVar benchmarked_execution_cost;
    private HashMap<Resource, GRBVar> benchmark_execution_costs;
    private GRBVar benchmarked_sending_cost;
    private HashMap<Resource, GRBVar> benchmark_sending_costs;
    private GRBVar benchmarked_receiving_cost;
    private HashMap<Resource, GRBVar> benchmark_receiving_costs;
    private GRBVar same_resource_sending_cost;
    private HashMap<Resource, GRBVar> same_resource_sending_costs;
    private GRBVar same_resource_receiving_cost;
    private HashMap<Resource, GRBVar> same_resource_receiving_costs;
    private HashMap<Resource, GRBVar> x_vars;
    private HashMap<Resource, GRBVar> z_vars;
    private HashMap<Integer, GRBVar> y_vars = new HashMap<Integer, GRBVar>();
    private Resource target_resource;
    private ArrayList<Resource> target_resources;
    private String target_resource_string;
    private Double execution_cost;
    private HashMap<Resource, Double> execution_costs;
    private Double send_cost;
    private HashMap<Resource, Double> send_costs;
    private Double recv_cost;
    private HashMap<Resource, Double> recv_costs;

    public ILPTask(GRBModel model, Integer prev_task_index, Integer task_index, Integer model_index) {
        this(model, prev_task_index, task_index, model_index, false);
    }

    public ILPTask(GRBModel model, Integer prev_task_index, Integer task_index, Integer model_index, Boolean verbose) {

        this.grb_model = model;
        this.task_index = task_index;
        this.model_index = model_index;
        this.prev_task_indexes.add(prev_task_index);
        try {

            String start_label = String.format("Ts_%d", task_index);
            this.start_time = this.grb_model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, start_label);
            if (verbose)
                System.out.print(String.format("Creating task with base variables: %s, ", start_label));

            String finish_label = String.format("Tf_%d", task_index);
            this.finish_time = this.grb_model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, finish_label);
            if (verbose)
                System.out.print(String.format("%s, ", finish_label));

            String tot_comm_label = String.format("Tc_%d", task_index);
            this.total_comm_cost = this.grb_model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, tot_comm_label);
            if (verbose)
                System.out.print(String.format("%s, ", tot_comm_label));

            String tot_exec_label = String.format("Te_%d", task_index);
            this.total_execution_cost = this.grb_model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, tot_exec_label);
            if (verbose)
                System.out.print(String.format("%s, ", tot_exec_label));

            String tot_send_label = String.format("Tsend_%d", task_index);
            this.total_sending_comm_cost = this.grb_model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS,
                    tot_send_label);
            if (verbose)
                System.out.print(String.format("%s, ", tot_send_label));

            String tot_recv_label = String.format("Trecv_%d", task_index);
            this.total_receiving_comm_cost = this.grb_model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS,
                    tot_recv_label);
            if (verbose)
                System.out.print(String.format("%s\n", tot_recv_label));

        } catch (GRBException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

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

    public Integer getTask_index() {
        return task_index;
    }

    public void setTask_index(Integer task_index) {
        this.task_index = task_index;
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
    public double getD_start_time() {
        try {
            return start_time.get(GRB.DoubleAttr.X);
        } catch (GRBException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return -1.0;
    }

    /**
     * @param grb_start_time
     */
    public void setStart_time(GRBVar grb_start_time) {
        this.start_time = grb_start_time;
    }

    /**
     * @return GRBVar
     */
    public double getD_finish_time() {
        try {
            return (double) finish_time.get(GRB.DoubleAttr.X);
        } catch (GRBException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return -1.0;
    }

    /**
     * @param grb_finish_time
     */
    public void setFinish_time(GRBVar grb_finish_time) {
        this.finish_time = grb_finish_time;
    }

    /**
     * @return GRBVar
     */
    public double getD_total_execution_cost() {
        try {
            return (double) total_execution_cost.get(GRB.DoubleAttr.X);
        } catch (GRBException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return -1.0;
    }

    public HashMap<Integer, GRBVar> getY_vars() {
        return y_vars;
    }

    public void setY_vars(HashMap<Integer, GRBVar> y_vars) {
        this.y_vars = y_vars;
    }

    /**
     * @param grb_execution_cost
     */
    public void setTotal_execution_cost(GRBVar grb_execution_cost) {
        this.total_execution_cost = grb_execution_cost;
    }

    /**
     * @return GRBVar
     */
    public double getD_total_comm_cost() {
        try {
            return (double) total_comm_cost.get(GRB.DoubleAttr.X);
        } catch (GRBException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return -1.0;
    }

    /**
     * @param grb_comm_cost
     */
    public void setTotal_comm_cost(GRBVar grb_comm_cost) {
        this.total_comm_cost = grb_comm_cost;
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

    public GRBModel getGrb_model() {
        return grb_model;
    }

    public void setGrb_model(GRBModel grb_model) {
        this.grb_model = grb_model;
    }

    public HashMap<Resource, GRBVar> getX_vars() {
        return x_vars;
    }

    public void setX_vars(HashMap<Resource, GRBVar> grb_x_vars) {
        this.x_vars = grb_x_vars;
    }

    public double getD_total_sending_comm_cost() {
        try {
            return total_sending_comm_cost.get(GRB.DoubleAttr.X);
        } catch (GRBException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return -1.0;
    }

    public void setTotal_sending_comm_cost(GRBVar grb_selected_sending_comm_cost) {
        this.total_sending_comm_cost = grb_selected_sending_comm_cost;
    }

    public double getD_total_receiving_comm_cost() {
        try {
            return (double) total_receiving_comm_cost.get(GRB.DoubleAttr.X);
        } catch (GRBException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return -1.0;
    }

    public void setTotal_receiving_comm_cost(GRBVar grb_selected_receiving_comm_cost) {
        this.total_receiving_comm_cost = grb_selected_receiving_comm_cost;
    }

    public double getD_benchmarked_sending_cost() {
        try {
            return (double) benchmarked_sending_cost.get(GRB.DoubleAttr.X);
        } catch (GRBException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return -1.0;
    }

    public void setBenchmarked_sending_cost(GRBVar grb_benchmarked_sending_cost) {
        this.benchmarked_sending_cost = grb_benchmarked_sending_cost;
    }

    public double getD_benchmarked_receiving_cost() {
        try {
            return (double) benchmarked_receiving_cost.get(GRB.DoubleAttr.X);
        } catch (GRBException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return -1.0;
    }

    public void setBenchmarked_receiving_cost(GRBVar grb_benchmarked_receiving_cost) {
        this.benchmarked_receiving_cost = grb_benchmarked_receiving_cost;
    }

    public double getD_same_resource_sending_cost() {
        try {
            return (double) same_resource_sending_cost.get(GRB.DoubleAttr.X);
        } catch (GRBException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return -1.0;
    }

    public void setSame_resource_sending_cost(GRBVar grb_same_resource_sending_cost) {
        this.same_resource_sending_cost = grb_same_resource_sending_cost;
    }

    public double getD_same_resource_receiving_cost() {
        try {
            return (double) same_resource_receiving_cost.get(GRB.DoubleAttr.X);
        } catch (GRBException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return -1.0;
    }

    public void setSame_resource_receiving_cost(GRBVar grb_same_resource_receiving_cost) {
        this.same_resource_receiving_cost = grb_same_resource_receiving_cost;
    }

    public HashMap<Resource, GRBVar> getZ_vars() {
        return z_vars;
    }

    public void setZ_vars(HashMap<Resource, GRBVar> grb_z_vars) {
        this.z_vars = grb_z_vars;
    }

    public double getD_benchmarked_execution_cost() {
        try {
            return (double) benchmarked_execution_cost.get(GRB.DoubleAttr.X);
        } catch (GRBException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return -1.0;
    }

    public void setBenchmarked_execution_cost(GRBVar grb_benchmarked_execution_cost) {
        this.benchmarked_execution_cost = grb_benchmarked_execution_cost;
    }

    public GRBVar getStart_time() {
        return start_time;
    }

    public GRBVar getFinish_time() {
        return finish_time;
    }

    public GRBVar getTotal_execution_cost() {
        return total_execution_cost;
    }

    public GRBVar getTotal_comm_cost() {
        return total_comm_cost;
    }

    public GRBVar getTotal_sending_comm_cost() {
        return total_sending_comm_cost;
    }

    public GRBVar getTotal_receiving_comm_cost() {
        return total_receiving_comm_cost;
    }

    public GRBVar getBenchmarked_execution_cost() {
        return benchmarked_execution_cost;
    }

    public GRBVar getBenchmarked_sending_cost() {
        return benchmarked_sending_cost;
    }

    public GRBVar getBenchmarked_receiving_cost() {
        return benchmarked_receiving_cost;
    }

    public GRBVar getSame_resource_sending_cost() {
        return same_resource_sending_cost;
    }

    public GRBVar getSame_resource_receiving_cost() {
        return same_resource_receiving_cost;
    }

    public HashMap<Resource, GRBVar> getBenchmark_execution_costs() {
        return benchmark_execution_costs;
    }

    public void setBenchmark_execution_costs(
            HashMap<Resource, GRBVar> grb_benchmark_execution_costs) {
        this.benchmark_execution_costs = grb_benchmark_execution_costs;
    }

    public HashMap<Resource, GRBVar> getBenchmark_sending_costs() {
        return benchmark_sending_costs;
    }

    public void setBenchmark_sending_costs(HashMap<Resource, GRBVar> grb_benchmark_sending_costs) {
        this.benchmark_sending_costs = grb_benchmark_sending_costs;
    }

    public HashMap<Resource, GRBVar> getBenchmark_receiving_costs() {
        return benchmark_receiving_costs;
    }

    public void setBenchmark_receiving_costs(
            HashMap<Resource, GRBVar> grb_benchmark_receiving_costs) {
        this.benchmark_receiving_costs = grb_benchmark_receiving_costs;
    }

    public HashMap<Resource, GRBVar> getSame_resource_sending_costs() {
        return same_resource_sending_costs;
    }

    public void setSame_resource_sending_costs(
            HashMap<Resource, GRBVar> grb_same_resource_sending_costs) {
        this.same_resource_sending_costs = grb_same_resource_sending_costs;
    }

    public HashMap<Resource, GRBVar> getSame_resource_receiving_costs() {
        return same_resource_receiving_costs;
    }

    public void setSame_resource_receiving_costs(
            HashMap<Resource, GRBVar> grb_same_resource_receiving_costs) {
        this.same_resource_receiving_costs = grb_same_resource_receiving_costs;
    }

    public String getTarget_resource_string() {
        return target_resource_string;
    }

    public void setTarget_resource_string(String target_resource_string) {
        this.target_resource_string = target_resource_string;
    }

    public Integer getModel_index() {
        return model_index;
    }

    public void setModel_index(Integer model_index) {
        this.model_index = model_index;
    }

    public ArrayList<Integer> getPrev_task_indexes() {
        return prev_task_indexes;
    }

    public void setPrev_task_indexes(ArrayList<Integer> prev_task_indexes) {
        this.prev_task_indexes = prev_task_indexes;
    }

}
