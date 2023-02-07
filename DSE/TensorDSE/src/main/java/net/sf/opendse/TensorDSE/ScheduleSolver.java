package net.sf.opendse.TensorDSE;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.javatuples.Pair;
import net.sf.opendse.model.Application;
import net.sf.opendse.model.Architecture;
import net.sf.opendse.model.Dependency;
import net.sf.opendse.model.Link;
import net.sf.opendse.model.Mappings;
import net.sf.opendse.model.Resource;
import net.sf.opendse.model.Routings;
import net.sf.opendse.model.Specification;
import net.sf.opendse.model.Task;

import gurobi.*;

public class ScheduleSolver {

    private Architecture<Resource, Link> architecture;
    private Application<Task, Dependency> application;
    private Mappings<Task, Resource> mapping;
    private Routings<Task, Resource, Link> routings;
    private Double K;

    // private HashMap<Integer, HashMap<Integer, Task>> application_graphs;
    private List<Task> starting_tasks;
    private OperationCosts operation_costs;

    public ScheduleSolver(Specification specification,
            HashMap<Integer, HashMap<Integer, Task>> application_graphs, List<Task> starting_tasks,
            OperationCosts operation_costs, Double K) {

        this.architecture = specification.getArchitecture();
        this.application = specification.getApplication();
        this.mapping = specification.getMappings();
        this.routings = specification.getRoutings();
        // this.application_graphs = application_graphs;
        this.starting_tasks = starting_tasks;
        this.operation_costs = operation_costs;
        this.K = K;

        addCommCosts(operation_costs);
    }

    public ScheduleSolver(Specification specification, HashMap<Integer, HashMap<Integer, Task>> tasks,
            List<Task> starting_tasks, OperationCosts operation_costs) {

        this.architecture = specification.getArchitecture();
        this.application = specification.getApplication();
        this.mapping = specification.getMappings();
        this.routings = specification.getRoutings();
        // this.application_graphs = tasks;
        this.starting_tasks = starting_tasks;
        this.operation_costs = operation_costs;
        this.K = 100.0;

        addCommCosts(operation_costs);
    }

    /**
     * @param operation_costs
     * @return Boolean
     */
    public Boolean addCommCosts(OperationCosts operation_costs) {

        // Tasks that were routed over will have the resource in the verticies of the routing
        // of the comm node that sits between the task and its predecessor

        for (Task task : this.application.getVertices()) {

            // Only interested in non-communication tasks
            if (task.getId().contains("comm"))
                continue;

            Collection<Task> predecessors = this.application.getPredecessors(task);

            if (predecessors.size() > 0) {

                // Comm task coming before current task
                Task predecessor_comm = predecessors.iterator().next();

                // Resources used in communication
                Collection<Resource> comm_resources = routings.get(predecessor_comm).getVertices();

                Boolean bus = false;

                for (Resource resource : comm_resources) {
                    if (resource.getId().contains("usb") || resource.getId().contains("pci")) {
                        bus = true;
                        break;
                    }
                }

                if (bus == true) {
                    Pair<Double, Double> comm_cost = operation_costs.GetCommCost("tpu",
                            task.getAttribute("type"), task.getAttribute("dtype"));
                    task.setAttribute("comm_cost", comm_cost);
                }
            }
        }

        return true;
    }

    /**
     * @param task
     * @return Resource
     */
    private Resource getTargetResource(Task task) {

        Resource target_resource = new ArrayList<Resource>(mapping.getTargets(task))
                .get(mapping.getTargets(task).size() - 1);

        return target_resource;
    }


    /**
     * @param task
     * @return ArrayList<Resource>
     */
    private ArrayList<Resource> getPossibleTargetResources(Task task) {
        ArrayList<Resource> target_resources = new ArrayList<Resource>(mapping.getTargets(task));

        return target_resources;
    }


    /**
     * @param comm
     * @param target_resource
     * @return Pair<Double, Double>
     */
    private Pair<Double, Double> getRoutedCommunicationCost(Task comm, Resource target_resource) {

        Pair<Double, Double> comm_cost = new Pair<>(0.0, 0.0);

        if (comm != null) {

            Collection<Resource> routed_targets = this.routings.get(comm).getVertices();

            for (Resource res : routed_targets) {
                // Resource -> bus communication, ie. send comms
                if (res.getId().contains("cpu") || res.getId().contains("tpu")
                        || res.getId().contains("gpu")) {
                    if (this.routings.get(comm).getPredecessorCount(res) > 0) {
                        Resource from =
                                this.routings.get(comm).getPredecessors(res).iterator().next();
                        // Because we are dealing with recv comms we need the prev op type for
                        // getting comms costs
                        Task prev_task = application.getPredecessors(comm).iterator().next();
                        if (from.getId().contains("pci") || from.getId().contains("usb")) {
                            Pair<Double, Double> comms =
                                    operation_costs.GetCommCost(res.getId().replaceAll("\\d", ""),
                                            prev_task.getAttribute("type"),
                                            prev_task.getAttribute("dtype"));
                            // Value0 is send, 1 is recv
                            comm_cost = comm_cost.setAt0(comms.getValue0() + comm_cost.getValue0());
                        }
                    }


                    // Bus -> resource communication, ie. recv comms
                } else if (res.getId().contains("usb") || res.getId().equals("pci")) {
                    if (this.routings.get(comm).getSuccessorCount(res) > 0) {
                        Resource to = architecture.getSuccessors(res).iterator().next();
                        Task next_task = application.getSuccessors(comm).iterator().next();
                        if (to.getId().contains("cpu") || to.getId().contains("tpu")
                                || to.getId().contains("gpu")) {
                            Pair<Double, Double> comms =
                                    operation_costs.GetCommCost(to.getId().replaceAll("\\d", ""),
                                            next_task.getAttribute("type"),
                                            next_task.getAttribute("dtype"));
                            // Value0 is send, 1 is recv
                            comm_cost = comm_cost.setAt1(comms.getValue1() + comm_cost.getValue1());
                        }
                    }
                }
            }
        }

        return comm_cost;
    }


    /**
     * @param task
     * @param target_resource
     * @return Double
     */
    private Double getExecutionCost(Task task, Resource target_resource) {
        Double exec_cost = operation_costs.GetOpCost(target_resource.getId().replaceAll("\\d", ""),
                task.getAttribute("type"), task.getAttribute("dtype"));

        return exec_cost;
    }


    /**
     * @return Boolean
     */
    public Boolean solveDSESchedule() {

        // Sequential arrays of models and their tasks
        ArrayList<ArrayList<ILPTask>> model_tasks = new ArrayList<ArrayList<ILPTask>>();

        // Hashmap for quickly accessing all tasks mapped to the same resource
        HashMap<String, ArrayList<ILPTask>> resource_mapped_tasks =
                new HashMap<String, ArrayList<ILPTask>>();

        try {

            ILPFormuation ilps = new ILPFormuation();
            GRBEnv env = new GRBEnv("bilinear.log");
            GRBModel model = new GRBModel(env);

            ArrayList<GRBVar> finish_times = new ArrayList<GRBVar>();

            // Process each branch of the application graph
            for (Task starting_task : this.starting_tasks) {

                // Array to sequentially store all the tasks in the model, used for sequential
                // constraints
                ArrayList<ILPTask> application_branch = new ArrayList<ILPTask>();
                model_tasks.add(application_branch);

                // Comm task coming after starting_task
                Task follwing_comm_task =
                        this.application.getSuccessors(starting_task).iterator().next();

                // Add to sequential model list
                Resource target_resource = getTargetResource(starting_task);
                Pair<Double, Double> comm_cost =
                        getRoutedCommunicationCost(follwing_comm_task, target_resource);
                Double exec_cost = getExecutionCost(starting_task, target_resource);
                ILPTask ilp_task = ilps.initILPTask(starting_task.getId(), target_resource,
                        comm_cost, exec_cost, model);
                application_branch.add(ilp_task);

                // Add to hashmap with resource as key
                ArrayList<ILPTask> list = resource_mapped_tasks.getOrDefault(
                        ilp_task.getTarget_resource().getId(), new ArrayList<ILPTask>());
                list.add(ilp_task);
                resource_mapped_tasks.put(ilp_task.getTarget_resource().getId(), list);

                int count = this.application.getSuccessorCount(starting_task);

                // Compile a list of all tasks in the application graph
                while (count > 0) {

                    // Get next task
                    Task task =
                            this.application.getSuccessors(follwing_comm_task).iterator().next();

                    // Get next comm task
                    if (this.application.getSuccessorCount(task) > 0)
                        follwing_comm_task = this.application.getSuccessors(task).iterator().next();
                    else
                        follwing_comm_task = null;

                    target_resource = getTargetResource(starting_task);
                    comm_cost = getRoutedCommunicationCost(follwing_comm_task, target_resource);
                    exec_cost = getExecutionCost(starting_task, target_resource);
                    ilp_task = ilps.initILPTask(starting_task.getId(), target_resource, comm_cost,
                            exec_cost, model);
                    application_branch.add(ilp_task);
                    list = resource_mapped_tasks.getOrDefault(ilp_task.getTarget_resource().getId(),
                            new ArrayList<ILPTask>());
                    list.add(ilp_task);
                    resource_mapped_tasks.put(ilp_task.getTarget_resource().getId(), list);

                    count = this.application.getSuccessorCount(task);

                    if (count == 0)
                        finish_times.add(ilp_task.getGrb_finish_time());
                }
            }

            GRBVar model_finish_times[] = new GRBVar[finish_times.size()];
            model_finish_times = finish_times.toArray(model_finish_times);

            // Step through sequential model list and create scheduling and finish time dependencies
            for (ArrayList<ILPTask> tasks : model_tasks) {

                // First task
                ILPTask first_task = tasks.get(0);

                ilps.addFinishTimeConstraint(first_task.getGrb_finish_time(),
                        first_task.getGrb_start_time(), first_task.getGrb_execution_cost(),
                        first_task.getGrb_comm_cost(), model);

                for (int i = 1; i < tasks.size(); i++) {

                    ilps.addTaskSchedulingDependencyConstraint(
                            tasks.get(i - 1).getGrb_finish_time(), tasks.get(i).getGrb_start_time(),
                            model);
                    ilps.addFinishTimeConstraint(tasks.get(i).getGrb_finish_time(),
                            tasks.get(i).getGrb_start_time(), tasks.get(i).getGrb_execution_cost(),
                            tasks.get(i).getGrb_comm_cost(), model);
                }
            }

            // Create scheduling variables, one per pair on same resource
            for (ArrayList<ILPTask> resource : resource_mapped_tasks.values())
                for (int i = 0; i < (resource.size() - 1); i++)
                    for (int j = i + 1; j < resource.size(); j++) {

                        // Add resource mapping constraint for each pair of tasks
                        ILPTask task_one = resource.get(i);
                        ILPTask task_two = resource.get(j);
                        GRBVar Y = model.addVar(0.0, 1.0, 0.0, GRB.BINARY,
                                String.format("Y:%d_%d", i, j));

                        ilps.addResourceMappingPairConstraint(task_one.getGrb_start_time(),
                                task_one.getGrb_finish_time(), task_two.getGrb_start_time(),
                                task_two.getGrb_finish_time(), Y, model);
                    }

            // Set objective
            GRBLinExpr obj = new GRBLinExpr();
            double[] coeffs = new double[model_finish_times.length];
            Arrays.fill(coeffs, 1.0 / model_finish_times.length);
            obj.addTerms(coeffs, model_finish_times);

            model.setObjective(obj, GRB.MINIMIZE);

            try {
                model.optimize();
            } catch (GRBException e) {
                System.out.println("Model optimization failed");
            }
        } catch (GRBException e) {
            System.out.println("Error code: " + e.getErrorCode() + ". " + e.getMessage());
        }

        int model_count = 0;
        for (ArrayList<ILPTask> model : model_tasks) {
            System.out.println(String.format("Model: #%d", model_count++));
            for (ILPTask task : model) {
                Double start_time = 0.0;
                Double finish_time = 0.0;
                try {
                    start_time = task.getGrb_start_time().get(GRB.DoubleAttr.X);
                    finish_time = task.getGrb_finish_time().get(GRB.DoubleAttr.X);
                } catch (GRBException e) {
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                }
                System.out.println(String.format("Task: %s, %,.4f -> %,.4f", task.getTask().getId(),
                        start_time, finish_time));
            }
        }
        return true;
    }

    public void solveILPMappingAndSchedule() {

        // Sequential arrays of models and their tasks
        ArrayList<ArrayList<ILPTask>> models = new ArrayList<ArrayList<ILPTask>>();
        ArrayList<GRBVar> final_task_finish_times = new ArrayList<GRBVar>();

        try {

            ILPFormuation ilps = new ILPFormuation();
            GRBEnv grb_env = new GRBEnv("bilinear.log");
            GRBModel grb_model = new GRBModel(grb_env);

            // Process each branch of the application graph
            for (Task starting_task : this.starting_tasks) {

                // Array to sequentially store all the tasks in the model, used for sequential
                // constraints
                ArrayList<ILPTask> model_tasks = new ArrayList<ILPTask>();
                models.add(model_tasks);

                // Comm task coming after starting_task
                Task dependent_comm =
                        this.application.getSuccessors(starting_task).iterator().next();

                // For all possible resources
                ArrayList<Resource> possible_resources = getPossibleTargetResources(starting_task);
                HashMap<Resource, Pair<Double, Double>> comm_costs =
                        new HashMap<Resource, Pair<Double, Double>>();
                HashMap<Resource, Double> exec_costs = new HashMap<Resource, Double>();
                for (Resource resource : possible_resources) {
                    comm_costs.put(resource,
                            this.operation_costs.GetCommCost(resource.getId().replaceAll("\\d", ""),
                                    starting_task.getAttribute("type"),
                                    starting_task.getAttribute("dtype")));
                    exec_costs.put(resource,
                            this.operation_costs.GetOpCost(resource.getId().replaceAll("\\d", ""),
                                    starting_task.getAttribute("type"),
                                    starting_task.getAttribute("dtype")));
                }
                ILPTask ilp_task = ilps.initILPTask(starting_task.getId(), possible_resources,
                        comm_costs, exec_costs, grb_model);
                model_tasks.add(ilp_task);

                // following tasks
                int count = this.application.getSuccessorCount(starting_task);

                // Compile a list of all tasks in the application graph
                while (count > 0) {

                    // Get next task
                    Task task = this.application.getSuccessors(dependent_comm).iterator().next();

                    // Get next comm task
                    if (this.application.getSuccessorCount(task) > 0)
                        dependent_comm = this.application.getSuccessors(task).iterator().next();
                    else
                        dependent_comm = null;

                    // For all possible resources
                    possible_resources = getPossibleTargetResources(task);
                    comm_costs = new HashMap<Resource, Pair<Double, Double>>();
                    exec_costs = new HashMap<Resource, Double>();
                    for (Resource resource : possible_resources) {
                        comm_costs.put(resource,
                                this.operation_costs.GetCommCost(
                                        resource.getId().replaceAll("\\d", ""),
                                        starting_task.getAttribute("type"),
                                        starting_task.getAttribute("dtype")));
                        exec_costs.put(resource,
                                this.operation_costs.GetOpCost(
                                        resource.getId().replaceAll("\\d", ""),
                                        starting_task.getAttribute("type"),
                                        starting_task.getAttribute("dtype")));
                    }
                    ilp_task = ilps.initILPTask(task.getId(), possible_resources, comm_costs,
                            exec_costs, grb_model);
                    model_tasks.add(ilp_task);

                    count = this.application.getSuccessorCount(task);
                }
            }

            // For debug printing
            ArrayList<ILPTask> all_tasks = new ArrayList<ILPTask>();
            ArrayList<String> task_names = new ArrayList<String>();
            ArrayList<GRBVar> start_times = new ArrayList<GRBVar>();
            ArrayList<GRBVar> execution_times = new ArrayList<GRBVar>();
            ArrayList<GRBVar> communication_times = new ArrayList<GRBVar>();
            ArrayList<GRBVar> selected_sending_comm_times = new ArrayList<GRBVar>();
            ArrayList<GRBVar> selected_receiving_comm_times = new ArrayList<GRBVar>();
            ArrayList<GRBVar> finish_times = new ArrayList<GRBVar>();
            ArrayList<HashMap<Resource, GRBVar>> x_mapping_vars =
                    new ArrayList<HashMap<Resource, GRBVar>>();
            ArrayList<HashMap<Resource, GRBVar>> benchmarked_execution_times =
                    new ArrayList<HashMap<Resource, GRBVar>>();
            ArrayList<HashMap<Resource, GRBVar>> z_vars =
                    new ArrayList<HashMap<Resource, GRBVar>>();
            ArrayList<HashMap<Resource, GRBVar>> benchmarked_sending_times =
                    new ArrayList<HashMap<Resource, GRBVar>>();
            ArrayList<HashMap<Resource, GRBVar>> same_resource_sending_times =
                    new ArrayList<HashMap<Resource, GRBVar>>();
            ArrayList<HashMap<Resource, GRBVar>> benchmarked_receiving_times =
                    new ArrayList<HashMap<Resource, GRBVar>>();
            ArrayList<HashMap<Resource, GRBVar>> same_resource_receiving_times =
                    new ArrayList<HashMap<Resource, GRBVar>>();

            // Step through sequential model list and create scheduling and finish time dependencies
            for (ArrayList<ILPTask> model : models) {

                GRBVar prev_task_finish_time = null;
                HashMap<Resource, GRBVar> prev_task_x_vars = null;
                HashMap<Resource, GRBVar> prev_task_z_vars = null;

                for (ILPTask task : model) {

                    task_names.add(task.getID());
                    all_tasks.add(task);

                    // 2.1 Start time

                    GRBVar ts = grb_model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "");
                    start_times.add(ts);

                    // 2.2 Total execution time

                    GRBVar te = grb_model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "");
                    execution_times.add(te);

                    // 2.3 Total communication time

                    GRBVar tc = grb_model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "");
                    communication_times.add(te);

                    // 2.4 Finish time

                    GRBVar tf = grb_model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "");
                    finish_times.add(te);

                    // tf = ts + te + tc
                    ilps.addFinishTimeConstraint(tf, ts, te, tc, grb_model);

                    // 2.5 Mapping variables

                    // One for each resource the task can be mapped to
                    HashMap<Resource, GRBVar> task_x_vars = new HashMap<Resource, GRBVar>();
                    for (Resource resource : task.getTarget_resources()) {
                        GRBVar x = grb_model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "");
                        task_x_vars.put(resource, x);
                    }
                    x_mapping_vars.add(task_x_vars);

                    // 5.1 Resource mapping

                    ilps.addResourceMappingConstraint(task_x_vars.values().toArray(new GRBVar[0]),
                            grb_model);

                    // 2.6 Resource mapped execution times

                    HashMap<Resource, GRBVar> task_benchmark_exec_time =
                            new HashMap<Resource, GRBVar>();
                    for (Map.Entry<Resource, Double> entry : task.getExecution_costs().entrySet()) {
                        Resource resource = entry.getKey();
                        Double exec_cost = task.getExecution_costs().get(resource);
                        GRBVar exec_time =
                                grb_model.addVar(exec_cost, exec_cost, 0.0, GRB.CONTINUOUS, "");
                        task_benchmark_exec_time.put(resource, exec_time);
                    }
                    benchmarked_execution_times.add(task_benchmark_exec_time);

                    ilps.addSumOfVectorsConstraint(te,
                            task_benchmark_exec_time.values().toArray(new GRBVar[0]),
                            task_x_vars.values().toArray(new GRBVar[0]), grb_model);

                    // 2.7 Task scheduling dependencies with prev task

                    if (prev_task_finish_time != null)
                        ilps.addTaskSchedulingDependencyConstraint(prev_task_finish_time, ts,
                                grb_model);

                    prev_task_finish_time = tf;

                    // 2.9 Communication cost selection

                    GRBVar selected_sending_comm_cost =
                            grb_model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "");
                    selected_sending_comm_times.add(selected_sending_comm_cost);

                    GRBVar selected_receiving_comm_cost =
                            grb_model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "");
                    selected_receiving_comm_times.add(selected_receiving_comm_cost);

                    // 2.8 Same resource communication costs
                    // 2.8.1 Z helper variable

                    HashMap<Resource, GRBVar> task_z_vars = new HashMap<Resource, GRBVar>();
                    for (Resource resource : task.getTarget_resources()) {
                        GRBVar z = grb_model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "");
                        task_z_vars.put(resource, z);
                    }
                    z_vars.add(task_z_vars);

                    if (prev_task_x_vars != null)
                        for (Resource resource : task.getTarget_resources()) {
                            GRBVar z_var = task_z_vars.get(resource);
                            GRBVar prev_task_x = prev_task_x_vars.get(resource);
                            GRBVar cur_task_x = task_x_vars.get(resource);
                            ilps.addPairAndConstrint(z_var, prev_task_x, cur_task_x, grb_model);
                        }

                    // 2.8.2 Same resource sending

                    // Benchmarked sending times
                    HashMap<Resource, GRBVar> task_benchmarked_sending_times =
                            new HashMap<Resource, GRBVar>();
                    for (Resource resource : task.getTarget_resources()) {
                        Double send_time = task.getSend_costs().get(resource);
                        GRBVar Cs = grb_model.addVar(send_time, send_time, 0.0, GRB.CONTINUOUS, "");
                        task_benchmarked_sending_times.put(resource, Cs);
                    }
                    benchmarked_sending_times.add(task_benchmarked_sending_times);

                    // sending times
                    HashMap<Resource, GRBVar> task_same_resource_sending_times =
                            new HashMap<Resource, GRBVar>();
                    for (Resource resource : task.getTarget_resources()) {
                        GRBVar send = grb_model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "");
                        task_same_resource_sending_times.put(resource, send);
                    }
                    same_resource_sending_times.add(task_same_resource_sending_times);

                    // Sending times are constrained from the first till the second last task.
                    // Thus, if the current task is not the last then we need to constrain between
                    // the prev and current tasks.
                    if (task != model.get(model.size() - 1)) {
                        for (Resource resource : task.getTarget_resources()) {
                            GRBVar same_resource_sending_time =
                                    task_same_resource_sending_times.get(resource);
                            GRBVar benchmarked_sending_time =
                                    task_benchmarked_sending_times.get(resource);
                            GRBVar z_var = task_z_vars.get(resource);
                            ilps.addSameResourceCommunicationCostConstraint(
                                    same_resource_sending_time, benchmarked_sending_time, z_var,
                                    grb_model);

                            // 2.9.1 Sending communication costs
                            ilps.addCommunicationCostSelectionConstraint(selected_sending_comm_cost,
                                    same_resource_sending_time, task_x_vars.get(resource),
                                    grb_model);
                        }
                    } else {
                        // Last task, so we can add its finish time to the list of final tasks
                        final_task_finish_times.add(tf);
                    }

                    // 2.8.3 Same resource receiving

                    // Benchmarked receiving times
                    HashMap<Resource, GRBVar> task_benchmarked_receiving_times =
                            new HashMap<Resource, GRBVar>();
                    for (Resource resource : task.getTarget_resources()) {
                        Double recv_time = task.getSend_costs().get(resource);
                        GRBVar Cs = grb_model.addVar(recv_time, recv_time, 0.0, GRB.CONTINUOUS, "");
                        task_benchmarked_receiving_times.put(resource, Cs);
                    }
                    benchmarked_receiving_times.add(task_benchmarked_receiving_times);

                    // receiving times
                    HashMap<Resource, GRBVar> task_same_resource_receiving_times =
                            new HashMap<Resource, GRBVar>();
                    for (Resource resource : task.getTarget_resources()) {
                        GRBVar recv = grb_model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "");
                        task_same_resource_receiving_times.put(resource, recv);
                    }
                    same_resource_receiving_times.add(task_same_resource_receiving_times);

                    // Receiving times are constrained from second task until the last task
                    // Thus, if the prev task is not null then we need to constrain between the prev
                    // task and the current task
                    if (prev_task_z_vars != null)
                        for (Resource resource : task.getTarget_resources()) {
                            GRBVar same_resource_receiving_time =
                                    task_same_resource_receiving_times.get(resource);
                            GRBVar benchmark_receiving_time =
                                    task_benchmarked_receiving_times.get(resource);
                            GRBVar z_var = prev_task_z_vars.get(resource);
                            ilps.addSameResourceCommunicationCostConstraint(
                                    same_resource_receiving_time, benchmark_receiving_time, z_var,
                                    grb_model);

                            // 2.9.2 Receiving communication costs
                            ilps.addCommunicationCostSelectionConstraint(
                                    selected_receiving_comm_cost, same_resource_receiving_time,
                                    task_x_vars.get(resource), grb_model);
                        }

                    // 2.10 Total communication costs

                    ilps.addTotalCommunicationCostConstraint(tc, selected_sending_comm_cost,
                            selected_receiving_comm_cost, grb_model);

                    prev_task_x_vars = task_x_vars;
                    prev_task_z_vars = task_z_vars;
                }
            }

            // 5.2 Resource sharing

            for (int i = 0; i < all_tasks.size(); i++) {
                for (int j = i + 1; j < all_tasks.size(); j++) {
                    GRBVar Y = grb_model.addVar(0.0, 1.0, 0.0, GRB.BINARY,
                            String.format("x_%d_%d", i, j));
                    ArrayList<Resource> resource_intersection =
                            new ArrayList<Resource>(all_tasks.get(i).getTarget_resources());
                    resource_intersection.retainAll(all_tasks.get(j).getTarget_resources());
                    for (Resource resource : resource_intersection) {
                        System.out.println(String.format("y_%d_%d_%s", i, j, resource.getId()));
                        ilps.addResourceMappingAllPairConstraint(start_times.get(i),
                                finish_times.get(i), start_times.get(j), finish_times.get(j), Y,
                                x_mapping_vars.get(i).get(resource),
                                x_mapping_vars.get(j).get(resource), this.K, grb_model);
                    }
                }
            }

            // Set objective
            GRBLinExpr obj = new GRBLinExpr();
            double[] coeffs = new double[final_task_finish_times.size()];
            Arrays.fill(coeffs, 1.0 / final_task_finish_times.size());
            obj.addTerms(coeffs, final_task_finish_times.toArray(new GRBVar[0]));

            grb_model.setObjective(obj, GRB.MINIMIZE);

            try {
                grb_model.optimize();
            } catch (GRBException e) {
                System.out.println("Model optimization failed");
            }

            for (int i = 0; i < start_times.size(); i++) {
                System.out.println(String.format("Task %d:%s, start: %f, finish: %f, exec: %f",
                        i + 1, task_names.get(i), start_times.get(i).get(GRB.DoubleAttr.X),
                        finish_times.get(i).get(GRB.DoubleAttr.X),
                        execution_times.get(i).get(GRB.DoubleAttr.X)));
                System.out.println(String.format("Comm: %f, send: %f, recv: %f",
                        communication_times.get(i).get(GRB.DoubleAttr.X),
                        selected_sending_comm_times.get(i).get(GRB.DoubleAttr.X),
                        selected_receiving_comm_times.get(i).get(GRB.DoubleAttr.X)));
                for (Map.Entry<Resource, GRBVar> entry : x_mapping_vars.get(i).entrySet())
                    if (entry.getValue().get(GRB.DoubleAttr.X) > 0.0)
                        System.out.println(
                                String.format("Mapped to resource %s", entry.getKey().getId()));

                System.out.println("------------------------------------------------");
                System.out.println("Benchmarks");
                System.out.print("Execution times,  ");
                for (Map.Entry<Resource, GRBVar> entry : benchmarked_execution_times.get(i)
                        .entrySet())
                    System.out.print(String.format("%s: %f, ", entry.getKey().getId(),
                            entry.getValue().get(GRB.DoubleAttr.X)));
                System.out.println();
                System.out.print("Sending times,  ");
                if (benchmarked_sending_times.get(i).size() == 0)
                    System.out.print("None");
                for (Map.Entry<Resource, GRBVar> entry : benchmarked_sending_times.get(i)
                        .entrySet())
                    System.out.print(String.format("%s: %f, ", entry.getKey().getId(),
                            entry.getValue().get(GRB.DoubleAttr.X)));
                System.out.println();

                System.out.print("Receiving times,  ");
                if (benchmarked_receiving_times.get(i).size() == 0)
                    System.out.print("None");
                for (Map.Entry<Resource, GRBVar> entry : benchmarked_receiving_times.get(i)
                        .entrySet())
                    System.out.print(String.format("%s: %f, ", entry.getKey().getId(),
                            entry.getValue().get(GRB.DoubleAttr.X)));
                System.out.println();

                System.out.println("------------------------------------------------");
                System.out.println("Same resource sending times");
                if (same_resource_sending_times.get(i).size() == 0)
                    System.out.print("None");
                for (Map.Entry<Resource, GRBVar> entry : same_resource_sending_times.get(i)
                        .entrySet())
                    System.out.print(String.format("%s: %f, ", entry.getKey().getId(),
                            entry.getValue().get(GRB.DoubleAttr.X)));
                System.out.println();

                System.out.println("Same resource receiving times");
                if (same_resource_receiving_times.get(i).size() == 0)
                    System.out.print("None");
                for (Map.Entry<Resource, GRBVar> entry : same_resource_receiving_times.get(i)
                        .entrySet())
                    System.out.print(String.format("%s: %f, ", entry.getKey().getId(),
                            entry.getValue().get(GRB.DoubleAttr.X)));
                System.out.println();
                System.out.println();
                System.out.println();
            }

            for (int i = 0; i < x_mapping_vars.size(); i++) {
                System.out.print(String.format("Task: %d:%s - ", i, task_names.get(i)));
                for (GRBVar x : x_mapping_vars.get(i).values().toArray(new GRBVar[0]))
                    System.out.print(String.format("%f, ", x.get(GRB.DoubleAttr.X)));
                System.out.println();
            }

            HashMap<Resource, ArrayList<Pair<String, Double>>> per_resource_schedule =
                    new HashMap<Resource, ArrayList<Pair<String, Double>>>();

            for (int i = 0; i < start_times.size(); i++) {
                Resource mapped_resource = null;
                for (Map.Entry<Resource, GRBVar> entry : x_mapping_vars.get(i).entrySet())
                    if (entry.getValue().get(GRB.DoubleAttr.X) > 0.0)
                        mapped_resource = entry.getKey();
                if (mapped_resource != null) {
                    per_resource_schedule.putIfAbsent(mapped_resource,
                            new ArrayList<Pair<String, Double>>());
                    per_resource_schedule.get(mapped_resource)
                            .add(new Pair<String, Double>(
                                    String.format("%s:%s", i, task_names.get(i)),
                                    start_times.get(i).get(GRB.DoubleAttr.X)));
                }
            }

            System.out.println();
            System.out.println("Resource wise schedule");
            for (Map.Entry<Resource, ArrayList<Pair<String, Double>>> entry : per_resource_schedule
                    .entrySet()) {
                System.out.println(String.format("Resource: %s", entry.getKey().getId()));
                // Sort tasks
                Collections.sort(entry.getValue(), Comparator.comparing(p -> p.getValue1()));
                for (Pair<String, Double> task : entry.getValue())
                    System.out
                            .println(String.format("%s @ %f", task.getValue0(), task.getValue1()));
            }
            System.out.println();

        } catch (GRBException e) {
            System.out.println("Error code: " + e.getErrorCode() + ". " + e.getMessage());
        }
    }

}
