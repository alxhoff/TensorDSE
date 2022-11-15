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
import org.apache.xpath.operations.Bool;
import net.sf.opendse.model.Application;
import net.sf.opendse.model.Architecture;
import net.sf.opendse.model.Dependency;
import net.sf.opendse.model.Link;
import net.sf.opendse.model.Mapping;
import net.sf.opendse.model.Mappings;
import net.sf.opendse.model.Resource;
import net.sf.opendse.model.Routings;
import net.sf.opendse.model.Specification;
import net.sf.opendse.model.Task;

import gurobi.*;

public class SolutionHelper {

    private Architecture<Resource, Link> architecture;
    private Application<Task, Dependency> application;
    private Mappings<Task, Resource> mapping;
    private Routings<Task, Resource, Link> routings;

    private HashMap<Integer, HashMap<Integer, Task>> tasks;
    private List<Task> starting_tasks;
    private Integer longest_model;

    public SolutionHelper(Specification specification,
            HashMap<Integer, HashMap<Integer, Task>> tasks, List<Task> starting_tasks,
            Integer longest_model) {

        this.architecture = specification.getArchitecture();
        this.application = specification.getApplication();
        this.mapping = specification.getMappings();
        this.routings = specification.getRoutings();
        this.tasks = tasks;
        this.starting_tasks = starting_tasks;
        this.longest_model = longest_model;
    }

    // Scheduling dependency of the form Ts_j >= Ts_i + Ei + Ci
    // where task i is a dependency of task j with the execution time Ei and communication
    // cost Ci
    private Void addSchedulingDependencyConstraint(GRBVar precursor_task_start,
            GRBVar precursor_exec_time, GRBVar precursor_comm_time, GRBVar dependent_task,
            GRBModel model) {
        GRBLinExpr exp = new GRBLinExpr();
        exp.addTerm(1.0, precursor_task_start);
        exp.addTerm(1.0, precursor_exec_time);
        exp.addTerm(1.0, precursor_comm_time);

        try {
            model.addConstr(exp, GRB.LESS_EQUAL, dependent_task, "");
        } catch (GRBException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        return null;
    }

    // For each pair of tasks that share a resource mapping, ie. they are both assigned to the same
    // resource
    // for execution, then there is an inter-task dependency that one must run before the other.
    // For the pair of tasks i and j this is given via
    //
    // Ts_i >= (Ts_j + Ej + Cj) * Ys >= Ts_j * Ys + Ej * Ys + Cj * Ys
    // Ts_j >= (Ts_i + Ei + Ci) * (1 - Ys) >= Ts_i - Ts_i * Ys + Ei - Ei * Ys + Ci - Ci * Ys
    private Void addResourceMappingPairConstraint(GRBVar task_one_start, GRBVar task_one_exec_time,
            GRBVar task_one_comm_time, GRBVar task_two_start, GRBVar task_two_exec_time,
            GRBVar task_two_comm_time, GRBVar Y, GRBModel model) {

        GRBQuadExpr exp1 = new GRBQuadExpr();
        exp1.addTerm(1.0, task_two_start, Y);
        exp1.addTerm(1.0, task_two_exec_time, Y);
        exp1.addTerm(1.0, task_two_comm_time, Y);

        try {
            model.addQConstr(exp1, GRB.LESS_EQUAL, task_one_start, "");
        } catch (GRBException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        GRBQuadExpr exp2 = new GRBQuadExpr();
        exp2.addTerm(1.0, task_one_start);
        exp2.addTerm(-1.0, task_one_start, Y);
        exp2.addTerm(1.0, task_one_exec_time);
        exp2.addTerm(-1.0, task_one_exec_time, Y);
        exp2.addTerm(1.0, task_one_comm_time);
        exp2.addTerm(-1.0, task_one_comm_time, Y);

        try {
            model.addQConstr(exp2, GRB.LESS_EQUAL, task_two_start, "");
        } catch (GRBException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }


        return null;
    }

    // Finish times are simply the start time summed with the execution time and communication time
    // Tf = Ts + E1 + C1
    private Void addFinishTimeConstraint(GRBVar task_finish, GRBVar task_start,
            GRBVar task_exec_time, GRBVar task_comm_time, GRBModel model) {
        GRBLinExpr exp = new GRBLinExpr();
        exp.addTerm(1.0, task_start);
        exp.addTerm(1.0, task_exec_time);
        exp.addTerm(1.0, task_comm_time);

        try {
            model.addConstr(exp, GRB.EQUAL, task_finish, "");
        } catch (GRBException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        return null;
    }

    public Boolean calculateTaskFinishTimes() {

        // for (Task starting_task : this.starting_tasks) {

        // List<Task> dependents = new ArrayList<Task>(Arrays.asList(starting_task));

        // Collection<Task> comm = this.application.getSuccessors(starting_task);
        // int count = this.application.getSuccessorCount(starting_task);

        // while (count > 0) {
        // Task task = this.application.getSuccessors(comm.iterator().next()).iterator().next();
        // dependents.add(task);

        // comm = this.application.getSuccessors(task);
        // count = this.application.getSuccessorCount(task);
        // }

        // for (Task task: dependents){
        // Resource target = this.mapping.getTargets(starting_task).iterator().next();

        // System.out.println("wait here");

        // }

        // }

        try {

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

            // Example implementation for 4 tasks with the following dependencies and mapping
            //
            // Application
            //
            // n1 -> n2
            // n3 -> n4
            //
            // Mapping
            //
            // n1 -> r1
            // n2 -> r1
            // n3 -> r1
            // n4 -> r2
            //
            // For each pair of tasks on the same resource we require a variable Y
            //
            // Y1 : n1, n2
            // Y2 : n1, n3
            // Y3 : n2, n3
            //
            // We are after the start times of each task which are each represented by a variable
            //
            // t1, t2, t3, t4

            // Create variables

            // Scheduling variables
            GRBVar Y1 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y1");
            GRBVar Y2 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y2");
            GRBVar Y3 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y3");

            // Start times for each task
            GRBVar Ts1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Ts1");
            GRBVar Ts2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Ts2");
            GRBVar Ts3 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Ts3");
            GRBVar Ts4 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Ts4");

            // Finish times for each task
            GRBVar Tf1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tf1");
            GRBVar Tf2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tf2");
            GRBVar Tf3 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tf3");
            GRBVar Tf4 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tf4");

            GRBVar finish_times[] = new GRBVar[] {Tf1, Tf2, Tf3, Tf4};
            GRBVar end_finish_times[] = new GRBVar[] {Tf2, Tf4};

            // Execution times, min and max values are the same to create constants
            GRBVar E1 = model.addVar(2.0, 2.0, 0.0, GRB.CONTINUOUS, "E1");
            GRBVar E2 = model.addVar(3.0, 3.0, 0.0, GRB.CONTINUOUS, "E2");
            GRBVar E3 = model.addVar(1.5, 1.5, 0.0, GRB.CONTINUOUS, "E3");
            GRBVar E4 = model.addVar(2.0, 2.0, 0.0, GRB.CONTINUOUS, "E4");

            // Communication times

            GRBVar C1 = model.addVar(0.0, 0.0, 0.0, GRB.CONTINUOUS, "C1");
            GRBVar C2 = model.addVar(1.0, 1.0, 0.0, GRB.CONTINUOUS, "C2");
            GRBVar C3 = model.addVar(0.0, 0.0, 0.0, GRB.CONTINUOUS, "C3");
            GRBVar C4 = model.addVar(1.5, 1.5, 0.0, GRB.CONTINUOUS, "C4");

            // Set objective

            GRBLinExpr obj = new GRBLinExpr();

            // // Minimize the maximum finish time
            // // Maximum finish time
            // GRBVar Tfmax = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "max");
            // model.addGenConstrMax(Tfmax, finish_times, 0.0, "max");
            // obj.addTerm(1.0, Tfmax);

            // Minimize the average finish time of all end tasks

            double[] coeffs = new double[end_finish_times.length];
            Arrays.fill(coeffs, 1.0/end_finish_times.length);
            obj.addTerms(coeffs, end_finish_times);

            model.setObjective(obj, GRB.MINIMIZE);

            // Add constraints

            // Scheduling dependencies of each task, ie. Ts2 >= Ts1 + e1 + c1
            addSchedulingDependencyConstraint(Ts1, E1, C1, Ts2, model);
            addSchedulingDependencyConstraint(Ts3, E3, C3, Ts4, model);

            // Resource mappings for each pair of tasks mapped to the same resource,
            // ie. n1,n2 : n1,n3 : n2,n3
            addResourceMappingPairConstraint(Ts1, E1, C1, Ts2, E2, C2, Y1, model);
            addResourceMappingPairConstraint(Ts1, E1, C1, Ts3, E3, C3, Y2, model);
            addResourceMappingPairConstraint(Ts2, E2, C2, Ts3, E3, C3, Y3, model);

            // Finish times
            addFinishTimeConstraint(Tf1, Ts1, E1, C1, model);
            addFinishTimeConstraint(Tf2, Ts2, E2, C2, model);
            addFinishTimeConstraint(Tf3, Ts3, E3, C3, model);
            addFinishTimeConstraint(Tf4, Ts4, E4, C4, model);

            try {
                model.optimize();
            } catch (GRBException e) {
                System.out.println("Model optimization failed");
            }

            System.out.println(obj.getValue());


            System.out.println(String.format("Task 1: %s -> %s: E: %s C: %s",
                    Ts1.get(GRB.DoubleAttr.X), Tf1.get(GRB.DoubleAttr.X), E1.get(GRB.DoubleAttr.X), C1.get(GRB.DoubleAttr.X)));
            System.out.println(String.format("Task 2: %s -> %s: E: %s C: %s",
                    Ts2.get(GRB.DoubleAttr.X), Tf2.get(GRB.DoubleAttr.X), E2.get(GRB.DoubleAttr.X), C2.get(GRB.DoubleAttr.X)));
            System.out.println(String.format("Task 3: %s -> %s: E: %s C: %s",
                    Ts3.get(GRB.DoubleAttr.X), Tf3.get(GRB.DoubleAttr.X), E3.get(GRB.DoubleAttr.X), C3.get(GRB.DoubleAttr.X)));
            System.out.println(String.format("Task 4: %s -> %s: E: %s C: %s",
                    Ts4.get(GRB.DoubleAttr.X), Tf4.get(GRB.DoubleAttr.X), E4.get(GRB.DoubleAttr.X), C4.get(GRB.DoubleAttr.X)));

            // Dispose of model and environment
            model.dispose();
            env.dispose();

        } catch (GRBException e) {
            System.out.println("Error code: " + e.getErrorCode() + ". " + e.getMessage());
        }

        System.out.println("here");

        return true;
    }

    public Boolean addTPUCommCosts(OperationCosts operation_costs) {

        // Tasks that were routed over will have the resource in the verticies of the routing
        // of the comm node that sits between the task and its predecessor

        for (Task task : this.application.getVertices()) {

            // Only interested in non-communication tasks
            if (task.getId().contains("comm"))
                continue;

            Collection<Task> predecessors = this.application.getPredecessors(task);

            if (predecessors.size() > 0) {

                Task predecessor_comm = predecessors.iterator().next();
                Collection<Resource> comm_resources = routings.get(predecessor_comm).getVertices();
                Boolean TPU = false;

                for (Resource resource : comm_resources) {
                    if (resource.getId().contains("tpu")) {
                        TPU = true;
                        break;
                    }
                }

                if (TPU == true) {
                    Double cost = task.getAttribute("cost");
                    Double comm_cost = operation_costs.GetCommCost("tpu", task.getAttribute("type"),
                            task.getAttribute("dtype"));
                    task.setAttribute("cost", cost + comm_cost);
                }
            }
        }

        return true;
    }
}
