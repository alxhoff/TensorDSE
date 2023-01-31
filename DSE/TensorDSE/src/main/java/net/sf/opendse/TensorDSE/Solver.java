package net.sf.opendse.TensorDSE;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Set;
import org.apache.xalan.templates.ElemSort;
import org.apache.xpath.operations.Bool;
import org.javatuples.Pair;
import com.google.ortools.sat.RoutesConstraintProto;
import edu.uci.ics.jung.visualization.control.AbstractPopupGraphMousePlugin;
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

public class Solver {

    private Architecture<Resource, Link> architecture;
    private Application<Task, Dependency> application;
    private Mappings<Task, Resource> mapping;
    private Routings<Task, Resource, Link> routings;

    private HashMap<Integer, HashMap<Integer, Task>> tasks;
    private List<Task> starting_tasks;
    private OperationCosts operation_costs;

    public Solver(Specification specification, HashMap<Integer, HashMap<Integer, Task>> tasks,
            List<Task> starting_tasks, OperationCosts operation_costs) {

        this.architecture = specification.getArchitecture();
        this.application = specification.getApplication();
        this.mapping = specification.getMappings();
        this.routings = specification.getRoutings();
        this.tasks = tasks;
        this.starting_tasks = starting_tasks;
        this.operation_costs = operation_costs;

        addCommCosts(operation_costs);
    }

    private Resource getTargetResource(Task task) {

        Resource target_resource = new ArrayList<Resource>(mapping.getTargets(task))
                .get(mapping.getTargets(task).size() - 1);

        return target_resource;
    }

    private Double getCommunicationCost(Task comm, Resource target_resource) {

        Double comm_cost = 0.0;

        if (comm != null) {

            Collection<Resource> routed_targets = this.routings.get(comm).getVertices();

            for (Resource res : routed_targets) {
                // Resource -> bus communication, ie. recv comms
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
                            comm_cost += comms.getValue0(); // Value0 is send, 1 is recv
                        }
                    }


                    // Bus -> resource communication, ie. send comms
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
                            comm_cost += comms.getValue1(); // Value0 is send, 1 is recv
                        }
                    }
                }
            }
        }

        return comm_cost;
    }

    private Double getExecutionCost(Task task, Resource target_resource) {
        Double exec_cost = operation_costs.GetOpCost(target_resource.getId().replaceAll("\\d", ""),
                task.getAttribute("type"), task.getAttribute("dtype"));

        return exec_cost;
    }

    public Boolean solveDSESchedule() {

        // Sequential arrays of models and their tasks
        ArrayList<ArrayList<ILPTask>> model_tasks = new ArrayList<ArrayList<ILPTask>>();

        // Hashmap for quickly accessing all tasks mapped to the same resource
        HashMap<String, ArrayList<ILPTask>> resource_mapped_tasks =
                new HashMap<String, ArrayList<ILPTask>>();

        try {

            ILPSolver ilps = new ILPSolver();

            GRBEnv env = new GRBEnv("bilinear.log");
            System.out.println(String.format("Using Gurobi version: %s",
                    env.getClass().getPackage().getSpecificationVersion()));
            try {
                System.out.println(String.format("Gurobi Loc: %s", env.getClass()
                        .getProtectionDomain().getCodeSource().getLocation().toURI().getPath()));
            } catch (URISyntaxException e1) {
                // TODO Auto-generated catch block
                e1.printStackTrace();
            }
            GRBModel model = new GRBModel(env);

            ArrayList<GRBVar> finish_times = new ArrayList<GRBVar>();

            // Process each branch of the application graph
            for (Task starting_task : this.starting_tasks) {

                // Array to sequentially store all the tasks in the model, used for sequential
                // constraints
                ArrayList<ILPTask> ilp_task_branch = new ArrayList<ILPTask>();
                model_tasks.add(ilp_task_branch);

                // Comm task coming after starting_task
                Task dependent_comm =
                        this.application.getSuccessors(starting_task).iterator().next();

                // Add to sequential model list
                Resource target_resource = getTargetResource(starting_task);
                Double comm_cost = getCommunicationCost(dependent_comm, target_resource);
                Double exec_cost = getExecutionCost(starting_task, target_resource);
                ILPTask ilp_task = ilps.initILPTask(starting_task.getId(), target_resource,
                        comm_cost, exec_cost, model);
                ilp_task_branch.add(ilp_task);

                // Add to hashmap with resource as key
                ArrayList<ILPTask> list = resource_mapped_tasks.getOrDefault(
                        ilp_task.getTarget_resource().getId(), new ArrayList<ILPTask>());
                list.add(ilp_task);
                resource_mapped_tasks.put(ilp_task.getTarget_resource().getId(), list);

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

                    target_resource = getTargetResource(starting_task);
                    comm_cost = getCommunicationCost(dependent_comm, target_resource);
                    exec_cost = getExecutionCost(starting_task, target_resource);
                    ilp_task = ilps.initILPTask(starting_task.getId(), target_resource, comm_cost,
                            exec_cost, model);
                    ilp_task_branch.add(ilp_task);
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

                    ilps.addTaskSchedulingDependencyConstraint(tasks.get(i - 1).getGrb_finish_time(),
                            tasks.get(i).getGrb_start_time(), model);
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

    public void solveILP() {

        // Sequential arrays of models and their tasks
        ArrayList<ArrayList<ILPTask>> model_tasks = new ArrayList<ArrayList<ILPTask>>();

        // Hashmap for quickly accessing all tasks mapped to the same resource
        HashMap<String, ArrayList<ILPTask>> resource_mapped_tasks =
                new HashMap<String, ArrayList<ILPTask>>();

        try {

            ILPSolver ilps = new ILPSolver();

            GRBEnv env = new GRBEnv("bilinear.log");
            System.out.println(String.format("Using Gurobi version: %s",
                    env.getClass().getPackage().getSpecificationVersion()));
            try {
                System.out.println(String.format("Gurobi Loc: %s", env.getClass()
                        .getProtectionDomain().getCodeSource().getLocation().toURI().getPath()));
            } catch (URISyntaxException e1) {
                // TODO Auto-generated catch block
                e1.printStackTrace();
            }
            GRBModel model = new GRBModel(env);

            ArrayList<GRBVar> finish_times = new ArrayList<GRBVar>();

            // Process each branch of the application graph
            for (Task starting_task : this.starting_tasks) {

                // Array to sequentially store all the tasks in the model, used for sequential
                // constraints
                ArrayList<ILPTask> ilp_task_branch = new ArrayList<ILPTask>();
                model_tasks.add(ilp_task_branch);

                // Comm task coming after starting_task
                Task dependent_comm =
                        this.application.getSuccessors(starting_task).iterator().next();

                // Add to sequential model list
                Resource target_resource = getTargetResource(starting_task);
                Double comm_cost = getCommunicationCost(dependent_comm, target_resource);
                Double exec_cost = getExecutionCost(starting_task, target_resource);
                ILPTask ilp_task = ilps.initILPTask(starting_task.getId(), target_resource,
                        comm_cost, exec_cost, model);
                ilp_task_branch.add(ilp_task);

                // Add to hashmap with resource as key
                ArrayList<ILPTask> list = resource_mapped_tasks.getOrDefault(
                        ilp_task.getTarget_resource().getId(), new ArrayList<ILPTask>());
                list.add(ilp_task);
                resource_mapped_tasks.put(ilp_task.getTarget_resource().getId(), list);

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

                    target_resource = getTargetResource(starting_task);
                    comm_cost = getCommunicationCost(dependent_comm, target_resource);
                    exec_cost = getExecutionCost(starting_task, target_resource);
                    ilp_task = ilps.initILPTask(starting_task.getId(), target_resource, comm_cost,
                            exec_cost, model);
                    ilp_task_branch.add(ilp_task);
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

                    ilps.addTaskSchedulingDependencyConstraint(tasks.get(i - 1).getGrb_finish_time(),
                            tasks.get(i).getGrb_start_time(), model);
                    ilps.addFinishTimeConstraint(tasks.get(i).getGrb_finish_time(),
                            tasks.get(i).getGrb_start_time(), tasks.get(i).getGrb_execution_cost(),
                            tasks.get(i).getGrb_comm_cost(), model);
                }
            }

            // Create scheduling variables, one per pair of tasks
            for (ArrayList<ILPTask> resource : resource_mapped_tasks.values())
                for (int i = 0; i < (resource.size() - 1); i++)
                    for (int j = i + 1; j < resource.size(); j++) {

                        // Add resource mapping constraint for each pair of tasks
                        ILPTask task_one = resource.get(i);
                        ILPTask task_two = resource.get(j);
                        GRBVar Y = model.addVar(0.0, 1.0, 0.0, GRB.BINARY,
                                String.format("Y:%d_%d", i, j));

                        // ilps.addResourceMappingAllPairConstraint(task_one.getGrb_start_time(),
                        // task_one.getGrb_execution_cost(), task_one.getGrb_comm_cost(),
                        // task_two.getGrb_start_time(), task_two.getGrb_execution_cost(),
                        // task_two.getGrb_comm_cost(), Y, model);
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

        System.out.println("wait here");
    }

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
}
