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
import org.javatuples.Triplet;
import net.sf.opendse.TensorDSE.JSON.Model.Model;
import net.sf.opendse.model.Application;
import net.sf.opendse.model.Dependency;
import net.sf.opendse.model.Link;
import net.sf.opendse.model.Mapping;
import net.sf.opendse.model.Mappings;
import net.sf.opendse.model.Resource;
import net.sf.opendse.model.Routings;
import net.sf.opendse.model.Specification;
import net.sf.opendse.model.Task;

import gurobi.*;

public class ScheduleSolver {

    private Application<Task, Dependency> application;
    private Mappings<Task, Resource> mapping;
    private Routings<Task, Resource, Link> routings;
    private Double K;
    private Boolean verbose;

    private List<Task> starting_tasks;
    private OperationCosts operation_costs;

    public List<Model> json_models = null;

    public ScheduleSolver(SpecificationDefinition specification_definition, Double K,
            Boolean verbose) {

        this.application = specification_definition.getSpecification().getApplication();
        this.mapping = specification_definition.getSpecification().getMappings();
        this.routings = specification_definition.getSpecification().getRoutings();
        this.starting_tasks = specification_definition.getStarting_tasks();
        this.operation_costs = specification_definition.getOperation_costs();
        this.K = K;
        this.verbose = verbose;
        this.json_models = specification_definition.getJson_models();

        addCommCosts(operation_costs);
    }

    public ScheduleSolver(SpecificationDefinition specification_definition, Boolean verbose) {

        this.application = specification_definition.getSpecification().getApplication();
        this.mapping = specification_definition.getSpecification().getMappings();
        this.routings = specification_definition.getSpecification().getRoutings();
        this.starting_tasks = specification_definition.getStarting_tasks();
        this.operation_costs = specification_definition.getOperation_costs();
        this.K = 100.0;
        this.verbose = verbose;
        this.json_models = specification_definition.getJson_models();

        addCommCosts(operation_costs);
    }

    public ScheduleSolver(Specification specification, List<Task> starting_tasks,
            OperationCosts operation_costs, Boolean verbose) {

        this.application = specification.getApplication();
        this.mapping = specification.getMappings();
        this.routings = specification.getRoutings();
        this.starting_tasks = starting_tasks;
        this.operation_costs = operation_costs;
        this.K = 100.0;
        this.verbose = verbose;
        this.json_models = null;

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


    // /**
    // * @param comm
    // * @param target_resource
    // * @return Pair<Double, Double>
    // */
    // private Pair<Double, Double> getRoutedCommunicationCost(Task comm, Resource target_resource)
    // {

    // Pair<Double, Double> comm_cost = new Pair<>(0.0, 0.0);

    // if (comm != null) {

    // Collection<Resource> routed_targets = this.routings.get(comm).getVertices();

    // for (Resource res : routed_targets) {
    // // Resource -> bus communication, ie. send comms
    // if (res.getId().contains("cpu") || res.getId().contains("tpu")
    // || res.getId().contains("gpu")) {
    // if (this.routings.get(comm).getPredecessorCount(res) > 0) {
    // Resource from =
    // this.routings.get(comm).getPredecessors(res).iterator().next();
    // // Because we are dealing with recv comms we need the prev op type for
    // // getting comms costs
    // Task prev_task = application.getPredecessors(comm).iterator().next();
    // if (from.getId().contains("pci") || from.getId().contains("usb")) {
    // Pair<Double, Double> comms =
    // operation_costs.GetCommCost(res.getId().replaceAll("\\d", ""),
    // prev_task.getAttribute("type"),
    // prev_task.getAttribute("dtype"));
    // // Value0 is send, 1 is recv
    // comm_cost = comm_cost.setAt0(comms.getValue0() + comm_cost.getValue0());
    // }
    // }


    // // Bus -> resource communication, ie. recv comms
    // } else if (res.getId().contains("usb") || res.getId().equals("pci")) {
    // if (this.routings.get(comm).getSuccessorCount(res) > 0) {
    // Resource to = architecture.getSuccessors(res).iterator().next();
    // Task next_task = application.getSuccessors(comm).iterator().next();
    // if (to.getId().contains("cpu") || to.getId().contains("tpu")
    // || to.getId().contains("gpu")) {
    // Pair<Double, Double> comms =
    // operation_costs.GetCommCost(to.getId().replaceAll("\\d", ""),
    // next_task.getAttribute("type"),
    // next_task.getAttribute("dtype"));
    // // Value0 is send, 1 is recv
    // comm_cost = comm_cost.setAt1(comms.getValue1() + comm_cost.getValue1());
    // }
    // }
    // }
    // }
    // }

    // return comm_cost;
    // }


    // /**
    // * @param task
    // * @param target_resource
    // * @return Double
    // */
    // private Double getExecutionCost(Task task, Resource target_resource) {
    // Double exec_cost = operation_costs.GetOpCost(target_resource.getId().replaceAll("\\d", ""),
    // task.getAttribute("type"), task.getAttribute("dtype"));

    // return exec_cost;
    // }


    /**
     * @return Double
     */
    public Double solveDSESchedule(Mappings<Task, Resource> possible_mappings) {

        // Sequential arrays of models and their tasks
        ArrayList<ArrayList<ILPTask>> models = new ArrayList<ArrayList<ILPTask>>();
        ArrayList<GRBVar> final_task_finish_times = new ArrayList<GRBVar>();

        // Hashmap for quickly accessing all tasks mapped to the same resource
        HashMap<Resource, ArrayList<ILPTask>> resource_mapped_tasks =
                new HashMap<Resource, ArrayList<ILPTask>>();

        try {

            ILPFormuation ilps = new ILPFormuation();
            GRBEnv grb_env = new GRBEnv(true);
            grb_env.set(GRB.IntParam.OutputFlag, 0);
            grb_env.set(GRB.IntParam.LogToConsole, 0);
            grb_env.set(GRB.IntParam.TuneOutput, 0);
            grb_env.set(GRB.IntParam.CSIdleTimeout, 10);
            grb_env.start();
            GRBModel grb_model = new GRBModel(grb_env);

            // Process each branch of the application graph
            for (Task starting_task : this.starting_tasks) {

                // Array to sequentially store all the tasks in the model, used for sequential
                // constraints
                ArrayList<ILPTask> model_tasks = new ArrayList<ILPTask>();
                models.add(model_tasks);

                // Comm task coming after starting_task
                Task following_comm =
                        this.application.getSuccessors(starting_task).iterator().next();

                Resource target_resource = getTargetResource(starting_task);
                Pair<Double, Double> comm_costs = this.operation_costs.GetCommCost(
                        target_resource.getId().replaceAll("\\d", ""),
                        starting_task.getAttribute("type"), starting_task.getAttribute("dtype"));
                Double exec_costs = this.operation_costs.GetOpCost(
                        target_resource.getId().replaceAll("\\d", ""),
                        starting_task.getAttribute("type"), starting_task.getAttribute("dtype"));

                ILPTask ilp_task = ilps.initILPTask(starting_task, target_resource, comm_costs,
                        exec_costs, grb_model);
                model_tasks.add(ilp_task);

                // following tasks
                int count = this.application.getSuccessorCount(starting_task);

                // Compile a list of all tasks in the application graph
                while (count > 0) {

                    // Get next task
                    Task task = this.application.getSuccessors(following_comm).iterator().next();

                    // Get next comm task
                    if (this.application.getSuccessorCount(task) > 0)
                        following_comm = this.application.getSuccessors(task).iterator().next();
                    else
                        following_comm = null;

                    // For all possible resources
                    target_resource = getTargetResource(task);
                    comm_costs = this.operation_costs.GetCommCost(
                            target_resource.getId().replaceAll("\\d", ""),
                            task.getAttribute("type"), task.getAttribute("dtype"));
                    exec_costs = this.operation_costs.GetOpCost(
                            target_resource.getId().replaceAll("\\d", ""),
                            task.getAttribute("type"), task.getAttribute("dtype"));

                    ilp_task = ilps.initILPTask(task, target_resource, comm_costs, exec_costs,
                            grb_model);
                    model_tasks.add(ilp_task);

                    count = this.application.getSuccessorCount(task);
                }
            }

            ArrayList<ILPTask> all_tasks = new ArrayList<ILPTask>();

            // Step through sequential model list and create scheduling and finish time dependencies
            for (ArrayList<ILPTask> model : models) {

                ILPTask prev_task = null;

                for (ILPTask task : model) {

                    all_tasks.add(task);

                    // Get possible resource targets for task
                    ArrayList<Resource> target_resources = new ArrayList<Resource>();
                    for (Mapping<Task, Resource> mapping : possible_mappings.get(task.getTask()))
                        target_resources.add(mapping.getTarget());
                    task.setTarget_resources(target_resources);

                    // All done during ILPTask init
                    // 2.1 Start time
                    // 2.2 Total execution time
                    // 2.3 Total communication time
                    // 2.4 Finish time

                    // tf = ts + te + tc
                    ilps.addFinishTimeConstraint(task.getFinish_time(), task.getStart_time(),
                            task.getTotal_execution_cost(), task.getTotal_comm_cost(), grb_model);

                    // 2.5 Task scheduling dependencies
                    // ts_j >= tf_i
                    if (prev_task != null)
                        ilps.addTaskSchedulingDependencyConstraint(prev_task.getFinish_time(),
                                task.getStart_time(), grb_model);

                    // 3.1 Mapping variables
                    // When constraining resource sharing (4.2) we will create constraints for all
                    // unique pairs of tasks sharing the same resource. As such, we can use the
                    // HashMap resource_mapped_tasks to store tasks in terms of their resource.
                    // But we still need X variables for the communication constraints.
                    ArrayList<ILPTask> task_array = resource_mapped_tasks
                            .getOrDefault(task.getTarget_resource(), new ArrayList<ILPTask>());
                    task_array.add(task);
                    resource_mapped_tasks.put(task.getTarget_resource(), task_array);

                    // One for each resource the task can be mapped to
                    HashMap<Resource, GRBVar> task_x_vars = new HashMap<Resource, GRBVar>();

                    for (Resource resource : task.getTarget_resources()) {

                        GRBVar x = null;

                        if (resource.getId() == task.getTarget_resource().getId())
                            x = grb_model.addVar(1.0, 1.0, 0.0, GRB.CONTINUOUS, "");
                        else
                            x = grb_model.addVar(0.0, 0.0, 0.0, GRB.CONTINUOUS, "");
                        task_x_vars.put(resource, x);
                    }
                    task.setX_vars(task_x_vars);

                    // 3.2 Resource mapped execution times
                    // Since we know the mapped resource we can directly set the total exec time
                    GRBVar exec_time = grb_model.addVar(task.getExecution_cost(),
                            task.getExecution_cost(), 0.0, GRB.CONTINUOUS, "");
                    task.setBenchmarked_execution_cost(exec_time);

                    ilps.addDirectExecutionTimeConstraint(task.getTotal_execution_cost(), exec_time,
                            grb_model);


                    // 2.6 Same resource communication costs
                    // Benchmarked sending times
                    GRBVar task_benchmarked_sending_cost = grb_model.addVar(task.getSend_cost(),
                            task.getSend_cost(), 0.0, GRB.CONTINUOUS, "");
                    task.setBenchmarked_sending_cost(task_benchmarked_sending_cost);
                    GRBVar same_resource_sending_cost =
                            grb_model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "");
                    task.setSame_resource_sending_cost(same_resource_sending_cost);

                    // Benchmarked receiving times
                    GRBVar task_benchmarked_receiving_cost = grb_model.addVar(task.getRecv_cost(),
                            task.getRecv_cost(), 0.0, GRB.CONTINUOUS, "");
                    task.setBenchmarked_receiving_cost(task_benchmarked_receiving_cost);
                    GRBVar same_resource_receiving_cost =
                            grb_model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "");
                    task.setSame_resource_receiving_cost(same_resource_receiving_cost);

                    // 2.6.1 Z helper variable
                    // Start creating z vars as of the second task.
                    // The i index Z var is for connection between the i-1 and i indexed tasks
                    HashMap<Resource, GRBVar> task_z_vars = null;
                    if (prev_task != null) {
                        task_z_vars = new HashMap<Resource, GRBVar>();
                        for (Resource resource : task.getTarget_resources()) {

                            GRBVar z = grb_model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "");
                            GRBVar prev_task_x = prev_task.getX_vars().get(resource);
                            GRBVar cur_task_x = task_x_vars.get(resource);

                            ilps.addPairAndConstrint(z, prev_task_x, cur_task_x, grb_model);
                            task_z_vars.put(resource, z);
                        }
                    }
                    task.setZ_vars(task_z_vars);

                    // Sending times are constrained from the first till the second last task.
                    // Thus, constraints are created as of the second task between the previous task
                    // and the current task. The sending cost being constrained comes from the
                    // previous
                    // task
                    if (prev_task != null) {

                        // 2.6.2 Same resource sending
                        ilps.addSameResourceCommunicationCostConstraint(
                                prev_task.getSame_resource_sending_cost(),
                                prev_task.getBenchmarked_sending_cost(),
                                task.getZ_vars().get(task.getTarget_resource()), grb_model);

                        // 3.3.1 Total sending communication costs
                        ilps.addCommunicationCostConstraint(prev_task.getTotal_sending_comm_cost(),
                                prev_task.getSame_resource_sending_cost(), grb_model);
                    }

                    // Receiving times are constrained from second task until the last task
                    // Thus, if the prev task is not null then we need to constrain between the prev
                    // task and the current task
                    if (prev_task != null) {

                        // 2.6.3 Total receiving cost
                        ilps.addSameResourceCommunicationCostConstraint(
                                task.getSame_resource_receiving_cost(),
                                task.getBenchmarked_receiving_cost(),
                                task.getZ_vars().get(task.getTarget_resource()), grb_model);

                        // 3.5 Receiving communication costs
                        ilps.addCommunicationCostConstraint(task.getTotal_receiving_comm_cost(),
                                task.getSame_resource_receiving_cost(), grb_model);
                    }

                    // 2.7 Total communication costs
                    ilps.addTotalCommunicationCostConstraint(task.getTotal_comm_cost(),
                            task.getSame_resource_sending_cost(),
                            task.getSame_resource_receiving_cost(), grb_model);

                    if (task == model.get(model.size() - 1))
                        // Last task, so we can add its finish time to the list of final tasks
                        final_task_finish_times.add(task.getFinish_time());
                    else
                        prev_task = task;
                }
            }

            // 3.6 Resource sharing
            for (Map.Entry<Resource, ArrayList<ILPTask>> entry : resource_mapped_tasks.entrySet()) {
                for (int i = 0; i < entry.getValue().size(); i++) {
                    for (int j = i + 1; j < entry.getValue().size(); j++) {
                        GRBVar Y = grb_model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "");
                        ilps.addResourceMappingPairConstraint(
                                entry.getValue().get(i).getStart_time(),
                                entry.getValue().get(i).getFinish_time(),
                                entry.getValue().get(j).getStart_time(),
                                entry.getValue().get(j).getFinish_time(), Y, grb_model);
                    }
                }
            }

            // Set objective
            GRBLinExpr obj = new GRBLinExpr();
            double[] coeffs = new double[final_task_finish_times.size()];
            Arrays.fill(coeffs, 1.0 / final_task_finish_times.size());
            obj.addTerms(coeffs, final_task_finish_times.toArray(new GRBVar[0]));

            grb_model.setObjective(obj, GRB.MINIMIZE);

            Double obj_val = -1.0;

            try {
                grb_model.optimize();
                obj_val = grb_model.get(GRB.DoubleAttr.ObjVal);
            } catch (GRBException e) {
                System.out.println("Model optimization failed");
            }

            if (this.verbose == true) {
                System.out.println(String.format("ObjVal: %f", obj_val));

                for (int i = 0; i < all_tasks.size(); i++) {
                    System.out.print(String.format("Task: %d:%s - ", i, all_tasks.get(i).getID()));
                    for (GRBVar x : all_tasks.get(i).getX_vars().values().toArray(new GRBVar[0]))
                        System.out.print(String.format("%f, ", x.get(GRB.DoubleAttr.X)));
                    System.out.println();
                }

                System.out.println();
                System.out.println("Z vars");
                for (ILPTask task : all_tasks) {
                    if (task.getZ_vars() != null)
                        for (Map.Entry<Resource, GRBVar> entry : task.getZ_vars().entrySet()) {
                            System.out.print(String.format("%s: %f, ", entry.getKey().getId(),
                                    entry.getValue().get(GRB.DoubleAttr.X)));
                        }
                    System.out.println();
                }

                HashMap<Resource, ArrayList<Pair<String, Double>>> per_resource_schedule =
                        new HashMap<Resource, ArrayList<Pair<String, Double>>>();

                for (int i = 0; i < all_tasks.size(); i++) {
                    ILPTask task = all_tasks.get(i);
                    System.out.println(String.format("Task %d:%s, start: %f, finish: %f, exec: %f",
                            i + 1, task.getID(), task.getD_start_time(), task.getD_finish_time(),
                            task.getD_total_execution_cost()));
                    System.out.println(String.format("Comm: %f, send: %f, recv: %f",
                            task.getD_total_comm_cost(), task.getD_total_sending_comm_cost(),
                            task.getD_total_receiving_comm_cost()));
                    for (Map.Entry<Resource, GRBVar> entry : task.getX_vars().entrySet())
                        if (entry.getValue().get(GRB.DoubleAttr.X) > 0.0)
                            System.out.println(
                                    String.format("Mapped to resource %s", entry.getKey().getId()));

                    System.out.println("------------------------------------------------");
                    System.out.println("Benchmarks");

                    System.out.print("Execution times,  ");
                    System.out.print(String.format("%f, ", task.getD_benchmarked_execution_cost()));
                    System.out.println();

                    System.out.print("Sending times,  ");
                    System.out.print(String.format("%f, ", task.getD_benchmarked_sending_cost()));
                    System.out.println();

                    System.out.print("Receiving times,  ");
                    System.out.print(String.format("%f, ", task.getD_benchmarked_receiving_cost()));
                    System.out.println();

                    System.out.println("------------------------------------------------");
                    System.out.println("Same resource sending times");

                    System.out.print(String.format("%f, ", task.getD_same_resource_sending_cost()));
                    System.out.println();

                    System.out.println("Same resource receiving times");
                    System.out
                            .print(String.format("%f, ", task.getD_same_resource_receiving_cost()));
                    System.out.println();

                    Resource mapped_resource = null;
                    for (Map.Entry<Resource, GRBVar> entry : task.getX_vars().entrySet())
                        if (entry.getValue().get(GRB.DoubleAttr.X) > 0.0)
                            mapped_resource = entry.getKey();
                    if (mapped_resource != null) {
                        per_resource_schedule.putIfAbsent(mapped_resource,
                                new ArrayList<Pair<String, Double>>());
                        per_resource_schedule.get(mapped_resource).add(new Pair<String, Double>(
                                String.format("%s:%s", i, task.getID()), task.getD_start_time()));
                    }
                    System.out.println();
                    System.out.println();
                }

                for (ILPTask task : all_tasks) {
                    System.out.print(String.format("Task: %s - ", task.getID()));
                    for (GRBVar x : task.getX_vars().values().toArray(new GRBVar[0]))
                        System.out.print(String.format("%f, ", x.get(GRB.DoubleAttr.X)));
                    System.out.println();
                }

                System.out.println();
                System.out.println("Resource wise schedule");
                for (Map.Entry<Resource, ArrayList<Pair<String, Double>>> entry : per_resource_schedule
                        .entrySet()) {
                    System.out.println(String.format("Resource: %s", entry.getKey().getId()));
                    // Sort tasks
                    Collections.sort(entry.getValue(), Comparator.comparing(p -> p.getValue1()));
                    for (Pair<String, Double> task : entry.getValue())
                        System.out.println(
                                String.format("%s @ %f", task.getValue0(), task.getValue1()));
                }
                System.out.println();
                System.out.println();


                for (GRBVar finish_time : final_task_finish_times)
                    System.out.println(
                            String.format("Finish time: %f", finish_time.get(GRB.DoubleAttr.X)));
            }

            return obj_val;
        } catch (

        GRBException e) {
            System.out.println("Error code: " + e.getErrorCode() + ". " + e.getMessage());
        }

        return -1.0;
    }

    public ArrayList<ArrayList<ILPTask>> solveILPMappingAndSchedule() {

        // Sequential arrays of models and their tasks
        ArrayList<ArrayList<ILPTask>> models = new ArrayList<ArrayList<ILPTask>>();
        ArrayList<GRBVar> final_task_finish_times = new ArrayList<GRBVar>();

        try {

            ILPFormuation ilps = new ILPFormuation();
            GRBEnv grb_env = new GRBEnv(true);
            grb_env.set(GRB.IntParam.OutputFlag, 0);
            grb_env.set(GRB.IntParam.LogToConsole, 0);
            grb_env.set(GRB.IntParam.TuneOutput, 0);
            grb_env.set(GRB.IntParam.CSIdleTimeout, 10);
            grb_env.start();
            GRBModel grb_model = new GRBModel(grb_env);

            // Process each branch of the application graph
            for (Task starting_task : this.starting_tasks) {

                // Array to sequentially store all the tasks in the model, used for sequential
                // constraints
                ArrayList<ILPTask> model_tasks = new ArrayList<ILPTask>();
                models.add(model_tasks);

                // Comm task coming after starting_task
                Task following_comm =
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
                ILPTask ilp_task = ilps.initILPTask(starting_task, possible_resources, comm_costs,
                        exec_costs, grb_model);
                model_tasks.add(ilp_task);

                // following tasks
                int count = this.application.getSuccessorCount(starting_task);

                // Compile a list of all tasks in the application graph
                while (count > 0) {

                    // Get next task
                    Task task = this.application.getSuccessors(following_comm).iterator().next();

                    // Get next comm task
                    if (this.application.getSuccessorCount(task) > 0)
                        following_comm = this.application.getSuccessors(task).iterator().next();
                    else
                        following_comm = null;

                    // For all possible resources
                    possible_resources = getPossibleTargetResources(task);
                    comm_costs = new HashMap<Resource, Pair<Double, Double>>();
                    exec_costs = new HashMap<Resource, Double>();
                    for (Resource resource : possible_resources) {
                        comm_costs.put(resource,
                                this.operation_costs.GetCommCost(
                                        resource.getId().replaceAll("\\d", ""),
                                        task.getAttribute("type"), task.getAttribute("dtype")));
                        exec_costs.put(resource,
                                this.operation_costs.GetOpCost(
                                        resource.getId().replaceAll("\\d", ""),
                                        task.getAttribute("type"), task.getAttribute("dtype")));
                    }
                    ilp_task = ilps.initILPTask(task, possible_resources, comm_costs, exec_costs,
                            grb_model);
                    model_tasks.add(ilp_task);

                    count = this.application.getSuccessorCount(task);
                }
            }

            // For debug printing
            ArrayList<ILPTask> all_tasks = new ArrayList<ILPTask>();

            // Step through sequential model list and create scheduling and finish time dependencies
            for (ArrayList<ILPTask> model : models) {

                ILPTask prev_task = null;

                for (ILPTask task : model) {

                    all_tasks.add(task);

                    // All done during ILPTask init
                    // 2.1 Start time
                    // 2.2 Total execution time
                    // 2.3 Total communication time
                    // 2.4 Finish time

                    // tf = ts + te + tc
                    ilps.addFinishTimeConstraint(task.getFinish_time(), task.getStart_time(),
                            task.getTotal_execution_cost(), task.getTotal_comm_cost(), grb_model);

                    // 2.5 Task scheduling dependencies
                    if (prev_task != null)
                        ilps.addTaskSchedulingDependencyConstraint(prev_task.getFinish_time(),
                                task.getStart_time(), grb_model);

                    // 4.1 Possible mapping variables (X)
                    // One for each resource the task can be mapped to
                    HashMap<Resource, GRBVar> task_x_vars = new HashMap<Resource, GRBVar>();
                    for (Resource resource : task.getTarget_resources()) {
                        GRBVar x = grb_model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "");
                        task_x_vars.put(resource, x);
                    }
                    task.setX_vars(task_x_vars);

                    // 4.2 Resource mapping
                    // Sum of all X variables for a single task can be at most 1
                    ilps.addResourceMappingConstraint(task_x_vars.values().toArray(new GRBVar[0]),
                            grb_model);

                    // 4.3 Resource mapped execution times
                    HashMap<Resource, GRBVar> task_benchmark_exec_time =
                            new HashMap<Resource, GRBVar>();

                    for (Map.Entry<Resource, Double> entry : task.getExecution_costs().entrySet())
                        task_benchmark_exec_time.put(entry.getKey(),
                                grb_model.addVar(task.getExecution_costs().get(entry.getKey()),
                                        task.getExecution_costs().get(entry.getKey()), 0.0,
                                        GRB.CONTINUOUS, ""));

                    task.setBenchmark_execution_costs(task_benchmark_exec_time);

                    ilps.addSumOfVectorsConstraint(task.getTotal_execution_cost(),
                            task_benchmark_exec_time.values().toArray(new GRBVar[0]),
                            task_x_vars.values().toArray(new GRBVar[0]), grb_model);

                    // 2.6 Same resource communication costs
                    // Benchmarked sending times
                    HashMap<Resource, GRBVar> task_benchmarked_sending_times =
                            new HashMap<Resource, GRBVar>();
                    for (Resource resource : task.getTarget_resources()) {
                        GRBVar Cs = grb_model.addVar(task.getSend_costs().get(resource),
                                task.getSend_costs().get(resource), 0.0, GRB.CONTINUOUS, "");
                        task_benchmarked_sending_times.put(resource, Cs);
                    }
                    task.setBenchmark_sending_costs(task_benchmarked_sending_times);

                    // sending times
                    HashMap<Resource, GRBVar> task_same_resource_sending_costs =
                            new HashMap<Resource, GRBVar>();
                    for (Resource resource : task.getTarget_resources()) {
                        GRBVar send = grb_model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "");
                        task_same_resource_sending_costs.put(resource, send);
                    }
                    task.setSame_resource_sending_costs(task_same_resource_sending_costs);

                    // Benchmarked receiving times
                    HashMap<Resource, GRBVar> task_benchmarked_receiving_costs =
                            new HashMap<Resource, GRBVar>();
                    for (Resource resource : task.getTarget_resources()) {
                        Double recv_time = task.getSend_costs().get(resource);
                        GRBVar Cs = grb_model.addVar(recv_time, recv_time, 0.0, GRB.CONTINUOUS, "");
                        task_benchmarked_receiving_costs.put(resource, Cs);
                    }
                    task.setBenchmark_receiving_costs(task_benchmarked_receiving_costs);

                    // receiving times
                    HashMap<Resource, GRBVar> task_same_resource_receiving_costs =
                            new HashMap<Resource, GRBVar>();
                    for (Resource resource : task.getTarget_resources()) {
                        GRBVar recv = grb_model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "");
                        task_same_resource_receiving_costs.put(resource, recv);
                    }
                    task.setSame_resource_receiving_costs(task_same_resource_receiving_costs);

                    // 2.6.1 Z helper variable
                    // Start creating z vars as of the second task.
                    // The i index Z var is for connection between the i-1 and i indexed tasks
                    HashMap<Resource, GRBVar> task_z_vars = null;
                    if (prev_task != null) {
                        task_z_vars = new HashMap<Resource, GRBVar>();
                        for (Resource resource : task.getTarget_resources()) {

                            GRBVar z = grb_model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "");
                            GRBVar prev_task_x = prev_task.getX_vars().get(resource);
                            GRBVar cur_task_x = task_x_vars.get(resource);

                            ilps.addPairAndConstrint(z, prev_task_x, cur_task_x, grb_model);
                            task_z_vars.put(resource, z);
                        }
                    }
                    task.setZ_vars(task_z_vars);

                    // Sending times are constrained from the first till the second last task.
                    // Thus, if the current task is not the last then we need to constrain between
                    // the prev and current tasks.
                    if (prev_task != null) {
                        for (Resource resource : task.getTarget_resources()) {

                            // 2.6.2 Total sending cost
                            ilps.addSameResourceCommunicationCostConstraint(
                                    prev_task.getSame_resource_sending_costs().get(resource),
                                    prev_task.getBenchmark_sending_costs().get(resource),
                                    task.getZ_vars().get(resource), grb_model);

                            // 4.4.1 Sending communication costs
                            ilps.addCommunicationCostSelectionConstraint(
                                    prev_task.getTotal_sending_comm_cost(),
                                    prev_task.getSame_resource_sending_costs().get(resource),
                                    prev_task.getX_vars().get(resource), grb_model);
                        }
                    }

                    // Receiving times are constrained from second task until the last task
                    // Thus, if the prev task is not null then we need to constrain between the prev
                    // task and the current task
                    if (prev_task != null)
                        for (Resource resource : task.getTarget_resources()) {

                            // 2.6.3 Total receiving cost
                            ilps.addSameResourceCommunicationCostConstraint(
                                    task_same_resource_receiving_costs.get(resource),
                                    task_benchmarked_receiving_costs.get(resource),
                                    task.getZ_vars().get(resource), grb_model);

                            // 4.4.2 Receiving communication costs
                            ilps.addCommunicationCostSelectionConstraint(
                                    task.getTotal_receiving_comm_cost(),
                                    task_same_resource_receiving_costs.get(resource),
                                    task.getX_vars().get(resource), grb_model);
                        }

                    // 2.7 Total communication costs
                    ilps.addTotalCommunicationCostConstraint(task.getTotal_comm_cost(),
                            task.getTotal_sending_comm_cost(), task.getTotal_receiving_comm_cost(),
                            grb_model);

                    if (task == model.get(model.size() - 1))
                        // Last task, so we can add its finish time to the list of final tasks
                        final_task_finish_times.add(task.getFinish_time());
                    else
                        prev_task = task;
                }
            }

            // 5.3 Resource sharing
            for (int i = 0; i < all_tasks.size(); i++) {
                for (int j = i + 1; j < all_tasks.size(); j++) {
                    GRBVar Y = grb_model.addVar(0.0, 1.0, 0.0, GRB.BINARY,
                            String.format("Y_%d_%d", i, j));
                    ArrayList<Resource> resource_intersection =
                            new ArrayList<Resource>(all_tasks.get(i).getTarget_resources());
                    resource_intersection.retainAll(all_tasks.get(j).getTarget_resources());
                    for (Resource resource : resource_intersection) {
                        ilps.addResourceMappingAllPairConstraint(all_tasks.get(i).getStart_time(),
                                all_tasks.get(i).getFinish_time(), all_tasks.get(j).getStart_time(),
                                all_tasks.get(j).getFinish_time(), Y,
                                all_tasks.get(i).getX_vars().get(resource),
                                all_tasks.get(j).getX_vars().get(resource), this.K, grb_model);
                    }
                }
            }

            // Set objective
            GRBLinExpr obj = new GRBLinExpr();
            double[] coeffs = new double[final_task_finish_times.size()];
            Arrays.fill(coeffs, 1.0 / final_task_finish_times.size());
            obj.addTerms(coeffs, final_task_finish_times.toArray(new GRBVar[0]));

            grb_model.setObjective(obj, GRB.MINIMIZE);

            Double obj_val = -1.0;

            try {
                grb_model.optimize();
                obj_val = grb_model.get(GRB.DoubleAttr.ObjVal);
            } catch (GRBException e) {
                System.out.println("Model optimization failed");
            }

            System.out.println(String.format("ObjVal: %f", obj_val));

            // Save mappings into json
            for (ILPTask task : all_tasks) {
                for (Map.Entry<Resource, GRBVar> entry : task.getX_vars().entrySet())
                    if (entry.getValue().get(GRB.DoubleAttr.X) > 0.0) {
                        task.setTarget_resource_string(entry.getKey().getId());
                    }
            }

            if (this.verbose == true) {

                HashMap<Resource, ArrayList<Triplet<String, Double, Double>>> per_resource_schedule =
                        new HashMap<Resource, ArrayList<Triplet<String, Double, Double>>>();

                for (int i = 0; i < all_tasks.size(); i++) {
                    ILPTask task = all_tasks.get(i);
                    System.out.println(String.format("Task %d:%s, start: %f, finish: %f, exec: %f",
                            i + 1, task.getID(), task.getD_start_time(), task.getD_finish_time(),
                            task.getD_total_execution_cost()));
                    System.out.println(String.format("Comm: %f, send: %f, recv: %f",
                            task.getD_total_comm_cost(), task.getD_total_sending_comm_cost(),
                            task.getD_total_receiving_comm_cost()));
                    for (Map.Entry<Resource, GRBVar> entry : task.getX_vars().entrySet())
                        if (entry.getValue().get(GRB.DoubleAttr.X) > 0.0) {
                            System.out.println(
                                    String.format("Mapped to resource %s", entry.getKey().getId()));
                        }

                    System.out.println("------------------------------------------------");
                    System.out.println("Benchmarks");

                    System.out.print("Execution times,  ");
                    for (Map.Entry<Resource, GRBVar> entry : task.getBenchmark_execution_costs()
                            .entrySet())
                        System.out.print(String.format("%s: %f, ", entry.getKey().getId(),
                                entry.getValue().get(GRB.DoubleAttr.X)));
                    System.out.println();
                    System.out.print("Sending times,  ");
                    if (task.getBenchmark_sending_costs().size() == 0)
                        System.out.print("None");
                    for (Map.Entry<Resource, GRBVar> entry : task.getBenchmark_sending_costs()
                            .entrySet())
                        System.out.print(String.format("%s: %f, ", entry.getKey().getId(),
                                entry.getValue().get(GRB.DoubleAttr.X)));
                    System.out.println();

                    System.out.print("Receiving times,  ");
                    if (task.getBenchmark_receiving_costs().size() == 0)
                        System.out.print("None");
                    for (Map.Entry<Resource, GRBVar> entry : task.getBenchmark_receiving_costs()
                            .entrySet())
                        System.out.print(String.format("%s: %f, ", entry.getKey().getId(),
                                entry.getValue().get(GRB.DoubleAttr.X)));
                    System.out.println();

                    System.out.println("------------------------------------------------");
                    System.out.println("Same resource sending times");

                    if (task.getSame_resource_sending_costs().size() == 0)
                        System.out.print("None");
                    for (Map.Entry<Resource, GRBVar> entry : task.getSame_resource_sending_costs()
                            .entrySet())
                        System.out.print(String.format("%s: %f, ", entry.getKey().getId(),
                                entry.getValue().get(GRB.DoubleAttr.X)));
                    System.out.println();

                    System.out.println("Same resource receiving times");
                    if (task.getSame_resource_receiving_costs().size() == 0)
                        System.out.print("None");
                    for (Map.Entry<Resource, GRBVar> entry : task.getSame_resource_receiving_costs()
                            .entrySet())
                        System.out.print(String.format("%s: %f, ", entry.getKey().getId(),
                                entry.getValue().get(GRB.DoubleAttr.X)));
                    System.out.println();

                    Resource mapped_resource = null;
                    for (Map.Entry<Resource, GRBVar> entry : task.getX_vars().entrySet())
                        if (entry.getValue().get(GRB.DoubleAttr.X) > 0.0)
                            mapped_resource = entry.getKey();
                    if (mapped_resource != null) {
                        per_resource_schedule.putIfAbsent(mapped_resource,
                                new ArrayList<Triplet<String, Double, Double>>());
                        per_resource_schedule.get(mapped_resource)
                                .add(new Triplet<String, Double, Double>(
                                        String.format("%s:%s", i, task.getID()),
                                        task.getD_start_time(), task.getD_finish_time()));
                    }
                    System.out.println();
                }

                System.out.println("X vars");
                for (ILPTask task : all_tasks) {
                    System.out.print(String.format("Task: %s - ", task.getID()));
                    for (GRBVar x : task.getX_vars().values().toArray(new GRBVar[0]))
                        System.out.print(String.format("%f, ", x.get(GRB.DoubleAttr.X)));
                    System.out.println();
                }
                System.out.println();

                System.out.println("Z vars");
                for (ILPTask task : all_tasks) {
                    if (task.getZ_vars() != null) {
                        for (GRBVar z : task.getZ_vars().values().toArray(new GRBVar[0]))
                            System.out.print(String.format("%f, ", z.get(GRB.DoubleAttr.X)));
                        System.out.println();
                    }
                }
                System.out.println();

                System.out.println("Resource wise schedule");
                for (Map.Entry<Resource, ArrayList<Triplet<String, Double, Double>>> entry : per_resource_schedule
                        .entrySet()) {
                    System.out.println();
                    System.out.println(String.format("Resource: %s", entry.getKey().getId()));
                    // Sort tasks
                    Collections.sort(entry.getValue(), Comparator.comparing(p -> p.getValue1()));
                    for (Triplet<String, Double, Double> task : entry.getValue())
                        System.out.println(String.format("%s @ %f -> %f", task.getValue0(),
                                task.getValue1(), task.getValue2()));
                }
                System.out.println();

                System.out.println("Model wise shedule");
                for (ArrayList<ILPTask> model : models) {
                    for (ILPTask task : model) {
                        System.out.println(String.format("%s on %s: %f -> %f", task.getID(),
                                task.getTarget_resource_string(), task.getD_start_time(),
                                task.getD_finish_time()));
                    }
                    System.out.println();
                }

                for (GRBVar finish_time : final_task_finish_times)
                    System.out.println(
                            String.format("Finish time: %f", finish_time.get(GRB.DoubleAttr.X)));

                System.out.println();
            }

        } catch (GRBException e) {
            System.out.println("Error code: " + e.getErrorCode() + ". " + e.getMessage());
        }

        return models;
    }

    public Application<Task, Dependency> getApplication() {
        return application;
    }

    public void setApplication(Application<Task, Dependency> application) {
        this.application = application;
    }

    public Mappings<Task, Resource> getMapping() {
        return mapping;
    }

    public void setMapping(Mappings<Task, Resource> mapping) {
        this.mapping = mapping;
    }

    public Routings<Task, Resource, Link> getRoutings() {
        return routings;
    }

    public void setRoutings(Routings<Task, Resource, Link> routings) {
        this.routings = routings;
    }

    public Double getK() {
        return K;
    }

    public void setK(Double k) {
        K = k;
    }

    public Boolean getVerbose() {
        return verbose;
    }

    public void setVerbose(Boolean verbose) {
        this.verbose = verbose;
    }

    public List<Task> getStarting_tasks() {
        return starting_tasks;
    }

    public void setStarting_tasks(List<Task> starting_tasks) {
        this.starting_tasks = starting_tasks;
    }

    public OperationCosts getOperation_costs() {
        return operation_costs;
    }

    public void setOperation_costs(OperationCosts operation_costs) {
        this.operation_costs = operation_costs;
    }

}