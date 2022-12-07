package net.sf.opendse.TensorDSE;

import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import gurobi.*;
import net.sf.opendse.model.Resource;

import net.sf.opendse.model.Task;

public class ILPSolver {

    public Double K;

    public ILPSolver() {
        this.K = 10.0;
    }

    public ILPSolver(Double K) {
        this.K = K;
    }

    // Scheduling dependency of the form Ts_j >= Ts_i + Ei + Ci
    // where task i is a dependency of task j with the execution time Ei and communication
    // cost Ci
    public Void addSchedulingDependencyConstraint(GRBVar precursor_task_start,
            GRBVar precursor_exec_time, GRBVar precursor_comm_time, GRBVar dependent_task_start,
            GRBModel model) {
        GRBLinExpr exp = new GRBLinExpr();
        exp.addTerm(1.0, precursor_task_start);
        exp.addTerm(1.0, precursor_exec_time);
        exp.addTerm(1.0, precursor_comm_time);

        try {
            model.addConstr(exp, GRB.LESS_EQUAL, dependent_task_start, "");
        } catch (GRBException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        return null;
    }

    // For each pair of tasks that share a resource mapping, ie. they are both assigned to the same
    // resource for execution, then there is an inter-task dependency that one must run before the
    // other.
    // For the pair of tasks i and j this is given via
    //
    // Ts_i >= (Ts_j + Ej + Cj) * Ys >= Ts_j * Ys + Ej * Ys + Cj * Ys
    // Ts_j >= (Ts_i + Ei + Ci) * (1 - Ys) >= Ts_i - Ts_i * Ys + Ei - Ei * Ys + Ci - Ci * Ys
    public Void addResourceMappingPairConstraint(GRBVar task_one_start, GRBVar task_one_exec_time,
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

    // When solving the mapping via the ILP, there should be a pair-wise resource mapping
    // constraints placed on all possible pairs of tasks.
    //
    // Ti_s >= (Tj_s * Ys) + (Ej * Ys) + (Cj * Ys) - (2K) + (Xi,r * K) + (Xj,r * K)
    //
    // Tj_s >= (Ti_s) - (Ti_s * Ys) + (Ei) - (Ei * Ys) + (Ci) - (Ci * Ys) - (2K) + (Xi,r * K) +
    // (Xj,r * K)
    public Void addResourceMappingAllPairConstraint(GRBVar task_one_start,
            GRBVar task_one_exec_time, GRBVar task_one_comm_time, GRBVar task_two_start,
            GRBVar task_two_exec_time, GRBVar task_two_comm_time, GRBVar Y, GRBVar X_one,
            GRBVar X_two, Double K, GRBModel model) {

        GRBQuadExpr exp1 = new GRBQuadExpr();
        exp1.addTerm(1.0, task_two_start, Y);
        exp1.addTerm(1.0, task_two_exec_time, Y);
        exp1.addTerm(1.0, task_two_comm_time, Y);
        exp1.addConstant(-2.0 * K);
        exp1.addTerm(K, X_one);
        exp1.addTerm(K, X_two);

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
        exp1.addConstant(-2.0 * K);
        exp1.addTerm(K, X_one);
        exp1.addTerm(K, X_two);

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
    public Void addFinishTimeConstraint(GRBVar task_finish, GRBVar task_start,
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

    public ILPTask initILPTask(String task_id, Resource target_resource, Double comm_cost,
            Double exec_cost, GRBModel model) {

        ILPTask ret = new ILPTask();

        try {
            ret.setGrb_start_time(model.addVar(0.0, GRB.INFINITY, 0, GRB.CONTINUOUS,
                    String.format("start_time:%s", task_id)));
            ret.setGrb_finish_time(model.addVar(0.0, GRB.INFINITY, 0, GRB.CONTINUOUS,
                    String.format("finish_time:%s", task_id)));
            ret.setGrb_execution_cost(model.addVar(exec_cost, exec_cost, 0.0, GRB.CONTINUOUS,
                    String.format("exec_cost:%s", task_id)));
            ret.setGrb_comm_cost(model.addVar(comm_cost, comm_cost, 0.0, GRB.CONTINUOUS,
                    String.format("exec_cost:%s", task_id)));
        } catch (GRBException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        ret.setComm_cost(comm_cost);
        ret.setExecution_cost(exec_cost);
        ret.setTarget_resource(target_resource);

        return ret;
    }

    private void gurobiDSEExample() {
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
            // // t1, t2, t3, t4

            // Create variables

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
            Arrays.fill(coeffs, 1.0 / end_finish_times.length);
            obj.addTerms(coeffs, end_finish_times);

            model.setObjective(obj, GRB.MINIMIZE);

            // Add constraints

            // Scheduling dependencies of each task, ie. Ts2 >= Ts1 + e1 + c1
            addSchedulingDependencyConstraint(Ts1, E1, C1, Ts2, model);
            addSchedulingDependencyConstraint(Ts3, E3, C3, Ts4, model);

            // Resource mappings for each pair of tasks mapped to the same resource,
            // ie. n1,n2 : n1,n3 : n2,n3
            // Scheduling variables - One for each pair of tasks m apped to same resource
            GRBVar Y1 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y1");
            GRBVar Y2 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y2");
            GRBVar Y3 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y3");

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
                    Ts1.get(GRB.DoubleAttr.X), Tf1.get(GRB.DoubleAttr.X), E1.get(GRB.DoubleAttr.X),
                    C1.get(GRB.DoubleAttr.X)));
            System.out.println(String.format("Task 2: %s -> %s: E: %s C: %s",
                    Ts2.get(GRB.DoubleAttr.X), Tf2.get(GRB.DoubleAttr.X), E2.get(GRB.DoubleAttr.X),
                    C2.get(GRB.DoubleAttr.X)));
            System.out.println(String.format("Task 3: %s -> %s: E: %s C: %s",
                    Ts3.get(GRB.DoubleAttr.X), Tf3.get(GRB.DoubleAttr.X), E3.get(GRB.DoubleAttr.X),
                    C3.get(GRB.DoubleAttr.X)));
            System.out.println(String.format("Task 4: %s -> %s: E: %s C: %s",
                    Ts4.get(GRB.DoubleAttr.X), Tf4.get(GRB.DoubleAttr.X), E4.get(GRB.DoubleAttr.X),
                    C4.get(GRB.DoubleAttr.X)));

            // Dispose of model and environment
            model.dispose();
            env.dispose();

        } catch (GRBException e) {
            System.out.println("Error code: " + e.getErrorCode() + ". " + e.getMessage());
        }
    }

    public void gurobiILPExampleGroupedComm() {

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

            // Example implementation for 4 tasks with the following dependencies and mapping
            //
            // Application
            //
            // n1 -> n2
            // n3 -> n4
            //
            //
            // For each pair of tasks we require a variable Y
            //
            // Y1 : n1, n2
            // Y2 : n1, n3
            // Y3 : n1, n4
            // Y4 : n2, n3
            // Y5 : n2, n4
            // Y6 : n3, n4
            //
            // We are after the start times of each task which are each represented by a variable
            // t1, t2, t3, t4

            // Create variables

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

            // Execution times, determined from constraint of mapping * resource specific exec times
            GRBVar Te1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Te1");
            GRBVar Te2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Te2");
            GRBVar Te3 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Te3");
            GRBVar Te4 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Te4");

            // Comm times, determined via the same means as exec times
            GRBVar Tc1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc1");
            GRBVar Tc2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc2");
            GRBVar Tc3 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc3");
            GRBVar Tc4 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc4");

            // GRBVar finish_times[] = new GRBVar[] {Tf1, Tf2, Tf3, Tf4};
            GRBVar end_finish_times[] = new GRBVar[] {Tf2, Tf4};

            // Execution times, min and max values are the same to create constants
            // Exec time for task 1 on resource 1
            GRBVar E1_1 = model.addVar(2.0, 2.0, 0.0, GRB.CONTINUOUS, "E1_1");
            // Exec time for task 1 on resource 2
            GRBVar E1_2 = model.addVar(1.0, 1.0, 0.0, GRB.CONTINUOUS, "E1_2");
            GRBVar E2_1 = model.addVar(1.5, 1.5, 0.0, GRB.CONTINUOUS, "E2_1");
            GRBVar E2_2 = model.addVar(1.5, 1.5, 0.0, GRB.CONTINUOUS, "E2_2");
            GRBVar E3_1 = model.addVar(1.0, 1.0, 0.0, GRB.CONTINUOUS, "E3_1");
            GRBVar E3_2 = model.addVar(3.0, 3.0, 0.0, GRB.CONTINUOUS, "E3_2");
            GRBVar E4_1 = model.addVar(2.0, 2.0, 0.0, GRB.CONTINUOUS, "E4_1");
            GRBVar E4_2 = model.addVar(2.5, 2.5, 0.0, GRB.CONTINUOUS, "E4_2");

            GRBVar[] exec_times = {E1_1, E1_2, E2_1, E2_2, E3_1, E3_2, E4_1, E4_2};

            // Communication times
            // Communication time for task 1 on resource 1
            GRBVar C1_1 = model.addVar(0.5, 0.5, 0.0, GRB.CONTINUOUS, "C1_1");
            // Communication time for task 1 on resource 2
            GRBVar C1_2 = model.addVar(2.0, 2.0, 0.0, GRB.CONTINUOUS, "C1_2");
            GRBVar C2_1 = model.addVar(0.5, 0.5, 0.0, GRB.CONTINUOUS, "C2_1");
            GRBVar C2_2 = model.addVar(1.0, 1.0, 0.0, GRB.CONTINUOUS, "C2_2");
            GRBVar C3_1 = model.addVar(1.0, 1.0, 0.0, GRB.CONTINUOUS, "C3_1");
            GRBVar C3_2 = model.addVar(0.5, 0.5, 0.0, GRB.CONTINUOUS, "C3_2");
            GRBVar C4_1 = model.addVar(1.0, 1.0, 0.0, GRB.CONTINUOUS, "C4_1");
            GRBVar C4_2 = model.addVar(0.5, 0.5, 0.0, GRB.CONTINUOUS, "C4_2");

            GRBVar[] comm_times = {C1_1, C1_2, C2_1, C2_2, C3_1, C3_2, C4_1, C4_2};

            // Add constraints

            // Scheduling dependencies of each task, ie. Ts2 >= Ts1 + e1 + c1
            ilps.addSchedulingDependencyConstraint(Ts1, Te1, Tc1, Ts2, model);
            ilps.addSchedulingDependencyConstraint(Ts3, Te3, Tc3, Ts4, model);

            // Resource mappings for each pair of tasks and for each resource, Ti,r.
            // ie. n1,1,n2,1 : n1,2,n2,1
            // Scheduling variables - One for each pair of tasks mapped to same resource
            GRBVar Y1 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y1");
            GRBVar Y2 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y2");
            GRBVar Y3 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y3");
            GRBVar Y4 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y4");
            GRBVar Y5 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y5");
            GRBVar Y6 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y6");
            GRBVar Y7 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y7");
            GRBVar Y8 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y8");
            GRBVar Y9 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y9");
            GRBVar Y10 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y10");
            GRBVar Y11 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y11");
            GRBVar Y12 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y12");

            // For each task and respectivley for each resource, a variable Xi,r is required
            // that can be either 0 or 1 to denote if that task is mapped to that resource.
            // This mapping is accompanied with the constraint that for a task i, and all r,
            // The sum of Xi,r must be 1.

            // N1
            // Resource 1
            GRBVar X1_1 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "X1,1");
            // Resource 2
            GRBVar X1_2 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "X1,2");
            // N2
            GRBVar X2_1 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "X2,1");
            GRBVar X2_2 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "X2,2");
            // N3
            GRBVar X3_1 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "X3,1");
            GRBVar X3_2 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "X3,2");
            // // N4
            GRBVar X4_1 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "X4,1");
            GRBVar X4_2 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "X4,2");

            GRBVar[] mapping_vars = {X1_1, X1_2, X2_1, X2_2, X3_1, X3_2, X4_1, X4_2};

            // Summ constraints
            // Meaning each task can only be allocated to one resource
            // ie. X1_1 (Task 1 on resource 1) + X1_2 (Task 1 on resource 2) = 1
            GRBLinExpr S1 = new GRBLinExpr();
            S1.addTerm(1.0, X1_1);
            S1.addTerm(1.0, X1_2);
            model.addConstr(S1, GRB.EQUAL, 1.0, "S1");
            GRBLinExpr S2 = new GRBLinExpr();
            S2.addTerm(1.0, X2_1);
            S2.addTerm(1.0, X2_2);
            model.addConstr(S2, GRB.EQUAL, 1.0, "S2");
            GRBLinExpr S3 = new GRBLinExpr();
            S3.addTerm(1.0, X3_1);
            S3.addTerm(1.0, X3_2);
            model.addConstr(S3, GRB.EQUAL, 1.0, "S3");
            GRBLinExpr S4 = new GRBLinExpr();
            S4.addTerm(1.0, X4_1);
            S4.addTerm(1.0, X4_2);
            model.addConstr(S4, GRB.EQUAL, 1.0, "S4");

            // Depending on mapping we need to activate the correct exec and comm times
            // ie. Te_1 = X1_1 * E1_1 + X1_2 * E1_2
            GRBQuadExpr E1 = new GRBQuadExpr();
            E1.addTerm(1.0, X1_1, E1_1);
            E1.addTerm(1.0, X1_2, E1_2);
            model.addQConstr(E1, GRB.EQUAL, Te1, "Te1");
            GRBQuadExpr E2 = new GRBQuadExpr();
            E2.addTerm(1.0, X2_1, E2_1);
            E2.addTerm(1.0, X2_2, E2_2);
            model.addQConstr(E2, GRB.EQUAL, Te2, "Te2");
            GRBQuadExpr E3 = new GRBQuadExpr();
            E3.addTerm(1.0, X3_1, E3_1);
            E3.addTerm(1.0, X3_2, E3_2);
            model.addQConstr(E3, GRB.EQUAL, Te3, "Te3");
            GRBQuadExpr E4 = new GRBQuadExpr();
            E4.addTerm(1.0, X4_1, E4_1);
            E4.addTerm(1.0, X4_2, E4_2);
            model.addQConstr(E4, GRB.EQUAL, Te4, "Te4");

            // Communication times
            // Tc_1 = X1_1 * C1_1 + X1_2 * C1_2
            GRBQuadExpr C1 = new GRBQuadExpr();
            C1.addTerm(1.0, X1_1, C1_1);
            C1.addTerm(1.0, X1_2, C1_2);
            model.addQConstr(C1, GRB.EQUAL, Tc1, "Tc1");
            GRBQuadExpr C2 = new GRBQuadExpr();
            C2.addTerm(1.0, X2_1, C2_1);
            C2.addTerm(1.0, X2_2, C2_2);
            model.addQConstr(C2, GRB.EQUAL, Tc2, "Tc2");
            GRBQuadExpr C3 = new GRBQuadExpr();
            C3.addTerm(1.0, X3_1, C3_1);
            C3.addTerm(1.0, X3_2, C3_2);
            model.addQConstr(C3, GRB.EQUAL, Tc3, "Tc3");
            GRBQuadExpr C4 = new GRBQuadExpr();
            C4.addTerm(1.0, X4_1, C4_1);
            C4.addTerm(1.0, X4_2, C4_2);
            model.addQConstr(C4, GRB.EQUAL, Tc4, "Tc4");

            // Resource constraints for each resource and each pair of tasks
            // N1 + N2
            // Resource 1
            ilps.addResourceMappingAllPairConstraint(Ts1, Te1, Tc1, Ts2, Te2, Tc2, Y1, X1_1, X2_1,
                    K, model);
            // Resource 2
            ilps.addResourceMappingAllPairConstraint(Ts1, Te1, Tc1, Ts2, Te2, Tc2, Y2, X1_2, X2_2,
                    K, model);
            // N1 + N3
            ilps.addResourceMappingAllPairConstraint(Ts1, Te1, Tc1, Ts3, Te3, Tc3, Y3, X1_1, X3_1,
                    K, model);
            ilps.addResourceMappingAllPairConstraint(Ts1, Te1, Tc1, Ts3, Te3, Tc3, Y4, X1_2, X3_2,
                    K, model);
            // N1 + N4
            ilps.addResourceMappingAllPairConstraint(Ts1, Te1, Tc1, Ts4, Te4, Tc4, Y5, X1_1, X4_1,
                    K, model);
            ilps.addResourceMappingAllPairConstraint(Ts1, Te1, Tc1, Ts4, Te4, Tc4, Y6, X1_2, X4_2,
                    K, model);
            // N2 + N3
            ilps.addResourceMappingAllPairConstraint(Ts2, Te2, Tc2, Ts3, Te3, Tc3, Y7, X2_1, X3_1,
                    K, model);
            ilps.addResourceMappingAllPairConstraint(Ts2, Te2, Tc2, Ts3, Te3, Tc3, Y8, X2_2, X3_2,
                    K, model);
            // N2 + N4
            ilps.addResourceMappingAllPairConstraint(Ts2, Te2, Tc2, Ts4, Te4, Tc4, Y9, X2_1, X4_1,
                    K, model);
            ilps.addResourceMappingAllPairConstraint(Ts2, Te2, Tc2, Ts4, Te4, Tc4, Y10, X2_2, X4_2,
                    K, model);
            // N3 + N4
            ilps.addResourceMappingAllPairConstraint(Ts3, Te3, Tc3, Ts4, Te4, Tc4, Y11, X3_1, X4_1,
                    K, model);
            ilps.addResourceMappingAllPairConstraint(Ts3, Te3, Tc3, Ts4, Te4, Tc4, Y12, X3_2, X4_2,
                    K, model);

            // Finish times
            // Tf = Ts + Ts + Tc
            ilps.addFinishTimeConstraint(Tf1, Ts1, Te1, Tc1, model);
            ilps.addFinishTimeConstraint(Tf2, Ts2, Te2, Tc2, model);
            ilps.addFinishTimeConstraint(Tf3, Ts3, Te3, Tc3, model);
            ilps.addFinishTimeConstraint(Tf4, Ts4, Te4, Tc4, model);

            // Set objective
            GRBLinExpr obj = new GRBLinExpr();

            // // Minimize the maximum finish time
            // // Maximum finish time
            // GRBVar Tfmax = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "max");
            // model.addGenConstrMax(Tfmax, finish_times, 0.0, "max");
            // obj.addTerm(1.0, Tfmax);

            // Minimize the average finish time of all end tasks
            double[] coeffs = new double[end_finish_times.length];
            Arrays.fill(coeffs, 1.0 / end_finish_times.length);
            obj.addTerms(coeffs, end_finish_times);
            model.setObjective(obj, GRB.MINIMIZE);

            try {
                model.optimize();
            } catch (GRBException e) {
                System.out.println("Model optimization failed");
            }

            System.out.println(obj.getValue());

            System.out.println(String.format("Task 1: %s -> %s: E: %s C: %s",
                    Ts1.get(GRB.DoubleAttr.X), Tf1.get(GRB.DoubleAttr.X), Te1.get(GRB.DoubleAttr.X),
                    Tc1.get(GRB.DoubleAttr.X)));
            System.out.println(String.format("Task 2: %s -> %s: E: %s C: %s",
                    Ts2.get(GRB.DoubleAttr.X), Tf2.get(GRB.DoubleAttr.X), Te2.get(GRB.DoubleAttr.X),
                    Tc2.get(GRB.DoubleAttr.X)));
            System.out.println(String.format("Task 3: %s -> %s: E: %s C: %s",
                    Ts3.get(GRB.DoubleAttr.X), Tf3.get(GRB.DoubleAttr.X), Te3.get(GRB.DoubleAttr.X),
                    Tc3.get(GRB.DoubleAttr.X)));
            System.out.println(String.format("Task 4: %s -> %s: E: %s C: %s",
                    Ts4.get(GRB.DoubleAttr.X), Tf4.get(GRB.DoubleAttr.X), Te4.get(GRB.DoubleAttr.X),
                    Tc4.get(GRB.DoubleAttr.X)));
            for (GRBVar var : mapping_vars)
                System.out.println(String.format("%s: %s", var.get(GRB.StringAttr.VarName),
                        var.get(GRB.DoubleAttr.X)));
            for (GRBVar var : exec_times)
                System.out.println(String.format("%s: %s", var.get(GRB.StringAttr.VarName),
                        var.get(GRB.DoubleAttr.X)));
            for (GRBVar var : comm_times)
                System.out.println(String.format("%s: %s", var.get(GRB.StringAttr.VarName),
                        var.get(GRB.DoubleAttr.X)));

            // Dispose of model and environment
            model.dispose();
            env.dispose();

        } catch (GRBException e) {
            System.out.println("Error code: " + e.getErrorCode() + ". " + e.getMessage());
        }
    }

    public void gurobiILPExampleSeparateComm() {

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

            // Example implementation for 4 tasks with the following dependencies and mapping
            //
            // Application
            //
            // n1 -> n2
            // n3 -> n4
            //
            //
            // For each pair of tasks we require a variable Y
            //
            // Y1 : n1, n2
            // Y2 : n1, n3
            // Y3 : n1, n4
            // Y4 : n2, n3
            // Y5 : n2, n4
            // Y6 : n3, n4
            //
            // We are after the start times of each task which are each represented by a variable
            // t1, t2, t3, t4

            // Create variables

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

            // Execution times, determined from constraint of mapping * resource specific exec times
            GRBVar Te1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Te1");
            GRBVar Te2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Te2");
            GRBVar Te3 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Te3");
            GRBVar Te4 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Te4");

            // Comm times, determined via the same means as exec times
            GRBVar Tc1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc1");
            GRBVar Tc2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc2");
            GRBVar Tc3 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc3");
            GRBVar Tc4 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc4");

            // GRBVar finish_times[] = new GRBVar[] {Tf1, Tf2, Tf3, Tf4};
            GRBVar end_finish_times[] = new GRBVar[] {Tf2, Tf4};

            // Execution times, min and max values are the same to create constants
            // Exec time for task 1 on resource 1
            GRBVar E1_1 = model.addVar(2.0, 2.0, 0.0, GRB.CONTINUOUS, "E1_1");
            // Exec time for task 1 on resource 2
            GRBVar E1_2 = model.addVar(1.0, 1.0, 0.0, GRB.CONTINUOUS, "E1_2");
            GRBVar E2_1 = model.addVar(1.5, 1.5, 0.0, GRB.CONTINUOUS, "E2_1");
            GRBVar E2_2 = model.addVar(1.5, 1.5, 0.0, GRB.CONTINUOUS, "E2_2");
            GRBVar E3_1 = model.addVar(1.0, 1.0, 0.0, GRB.CONTINUOUS, "E3_1");
            GRBVar E3_2 = model.addVar(3.0, 3.0, 0.0, GRB.CONTINUOUS, "E3_2");
            GRBVar E4_1 = model.addVar(2.0, 2.0, 0.0, GRB.CONTINUOUS, "E4_1");
            GRBVar E4_2 = model.addVar(2.5, 2.5, 0.0, GRB.CONTINUOUS, "E4_2");

            GRBVar[] exec_times = {E1_1, E1_2, E2_1, E2_2, E3_1, E3_2, E4_1, E4_2};

            // Communication times
            // Communication time for task 1 on resource 1
            GRBVar C1_1 = model.addVar(0.5, 0.5, 0.0, GRB.CONTINUOUS, "C1_1");
            // Communication time for task 1 on resource 2
            GRBVar C1_2 = model.addVar(2.0, 2.0, 0.0, GRB.CONTINUOUS, "C1_2");
            GRBVar C2_1 = model.addVar(0.5, 0.5, 0.0, GRB.CONTINUOUS, "C2_1");
            GRBVar C2_2 = model.addVar(1.0, 1.0, 0.0, GRB.CONTINUOUS, "C2_2");
            GRBVar C3_1 = model.addVar(1.0, 1.0, 0.0, GRB.CONTINUOUS, "C3_1");
            GRBVar C3_2 = model.addVar(0.5, 0.5, 0.0, GRB.CONTINUOUS, "C3_2");
            GRBVar C4_1 = model.addVar(1.0, 1.0, 0.0, GRB.CONTINUOUS, "C4_1");
            GRBVar C4_2 = model.addVar(0.5, 0.5, 0.0, GRB.CONTINUOUS, "C4_2");

            GRBVar[] comm_times = {C1_1, C1_2, C2_1, C2_2, C3_1, C3_2, C4_1, C4_2};

            // Add constraints

            // Scheduling dependencies of each task, ie. Ts2 >= Ts1 + e1 + c1
            ilps.addSchedulingDependencyConstraint(Ts1, Te1, Tc1, Ts2, model);
            ilps.addSchedulingDependencyConstraint(Ts3, Te3, Tc3, Ts4, model);

            // Resource mappings for each pair of tasks and for each resource, Ti,r.
            // ie. n1,1,n2,1 : n1,2,n2,1
            // Scheduling variables - One for each pair of tasks mapped to same resource
            GRBVar Y1 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y1");
            GRBVar Y2 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y2");
            GRBVar Y3 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y3");
            GRBVar Y4 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y4");
            GRBVar Y5 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y5");
            GRBVar Y6 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y6");
            GRBVar Y7 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y7");
            GRBVar Y8 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y8");
            GRBVar Y9 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y9");
            GRBVar Y10 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y10");
            GRBVar Y11 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y11");
            GRBVar Y12 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y12");

            // For each task and respectivley for each resource, a variable Xi,r is required
            // that can be either 0 or 1 to denote if that task is mapped to that resource.
            // This mapping is accompanied with the constraint that for a task i, and all r,
            // The sum of Xi,r must be 1.

            // N1
            // Resource 1
            GRBVar X1_1 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "X1,1");
            // Resource 2
            GRBVar X1_2 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "X1,2");
            // N2
            GRBVar X2_1 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "X2,1");
            GRBVar X2_2 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "X2,2");
            // N3
            GRBVar X3_1 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "X3,1");
            GRBVar X3_2 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "X3,2");
            // // N4
            GRBVar X4_1 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "X4,1");
            GRBVar X4_2 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "X4,2");

            GRBVar[] mapping_vars = {X1_1, X1_2, X2_1, X2_2, X3_1, X3_2, X4_1, X4_2};

            // Summ constraints
            // Meaning each task can only be allocated to one resource
            // ie. X1_1 (Task 1 on resource 1) + X1_2 (Task 1 on resource 2) = 1
            GRBLinExpr S1 = new GRBLinExpr();
            S1.addTerm(1.0, X1_1);
            S1.addTerm(1.0, X1_2);
            model.addConstr(S1, GRB.EQUAL, 1.0, "S1");
            GRBLinExpr S2 = new GRBLinExpr();
            S2.addTerm(1.0, X2_1);
            S2.addTerm(1.0, X2_2);
            model.addConstr(S2, GRB.EQUAL, 1.0, "S2");
            GRBLinExpr S3 = new GRBLinExpr();
            S3.addTerm(1.0, X3_1);
            S3.addTerm(1.0, X3_2);
            model.addConstr(S3, GRB.EQUAL, 1.0, "S3");
            GRBLinExpr S4 = new GRBLinExpr();
            S4.addTerm(1.0, X4_1);
            S4.addTerm(1.0, X4_2);
            model.addConstr(S4, GRB.EQUAL, 1.0, "S4");

            // Depending on mapping we need to activate the correct exec and comm times
            // ie. Te_1 = X1_1 * E1_1 + X1_2 * E1_2
            GRBQuadExpr E1 = new GRBQuadExpr();
            E1.addTerm(1.0, X1_1, E1_1);
            E1.addTerm(1.0, X1_2, E1_2);
            model.addQConstr(E1, GRB.EQUAL, Te1, "Te1");
            GRBQuadExpr E2 = new GRBQuadExpr();
            E2.addTerm(1.0, X2_1, E2_1);
            E2.addTerm(1.0, X2_2, E2_2);
            model.addQConstr(E2, GRB.EQUAL, Te2, "Te2");
            GRBQuadExpr E3 = new GRBQuadExpr();
            E3.addTerm(1.0, X3_1, E3_1);
            E3.addTerm(1.0, X3_2, E3_2);
            model.addQConstr(E3, GRB.EQUAL, Te3, "Te3");
            GRBQuadExpr E4 = new GRBQuadExpr();
            E4.addTerm(1.0, X4_1, E4_1);
            E4.addTerm(1.0, X4_2, E4_2);
            model.addQConstr(E4, GRB.EQUAL, Te4, "Te4");

            // Communication times
            // Tc_1 = X1_1 * C1_1 + X1_2 * C1_2
            // GRBQuadExpr C1 = new GRBQuadExpr();
            // C1.addTerm(1.0, X1_1, C1_1);
            // C1.addTerm(1.0, X1_2, C1_2);
            // model.addQConstr(C1, GRB.EQUAL, Tc1, "Tc1");
            // GRBQuadExpr C2 = new GRBQuadExpr();
            // C2.addTerm(1.0, X2_1, C2_1);
            // C2.addTerm(1.0, X2_2, C2_2);
            // model.addQConstr(C2, GRB.EQUAL, Tc2, "Tc2");
            // GRBQuadExpr C3 = new GRBQuadExpr();
            // C3.addTerm(1.0, X3_1, C3_1);
            // C3.addTerm(1.0, X3_2, C3_2);
            // model.addQConstr(C3, GRB.EQUAL, Tc3, "Tc3");
            // GRBQuadExpr C4 = new GRBQuadExpr();
            // C4.addTerm(1.0, X4_1, C4_1);
            // C4.addTerm(1.0, X4_2, C4_2);
            // model.addQConstr(C4, GRB.EQUAL, Tc4, "Tc4");

            // Communication times are comprised of a recieve time and a send time.
            // ie. If an application has to transition between resources for two sequential
            // tasks then there is a communication component where data is recieved from
            // the previous resource, then being sent to the next resource.

            // To track if a communication pair (recv + send) is required, a helper variables
            // track if two sequential tasks have the same resource mapping. If they do
            // then the communication cost pair is disabled. Thus, there is one helper
            // variable per pair of sequential tasks per resource, ie. number of edges
            // in the application graph * number of resources.

            // These helper variables make sure that the communication cost pair is only
            // disabled if BOTH mappings are the same,
            // ie. Z1_2_1 = 1 if X1_1 = X2_1 = 1
            // this is required for each resource,

            // Thus, via the constraints
            // Z1_2_1 <- Task edge 1->2 with respect to resource 1
            // Z1_2_1 <= X1_1 <- Task 1 mapped onto resource 1
            // Z1_2_1 <= X2_1 <- Task 2 mapped onto resource 1
            // C1_2_1 = (Cr1 + Cs2) * (1-Z1_2_1)
            // C1_2_1 = Cr1 - (Cr1 * Z1_2_1) + Cs2 - (Cs2 * Z1_2_1)

            // Thus, if both X1_1 and X2_1 are 1, ie. both tasks are on resource 1
            // Z1_2_1 resolves to 1 and the communication cost for C1_2_1 is dropped
            // to zero.

            // To sum the overall communication cost between task 1 and 2 for all resources
            // we again use the helper variables to stop us from getting multiple copies of
            // Cr1 + Cs2
            // C1_2 = (Cr1 + Cs2) * (1 - sum(Z1_2 over all r))
            // eg. 
            // C1_2 = (Cr1 + Cs2) * (1 - Z1_2_1 - Z1_2_2)
            // C1_2 = Cr1 + Cs2 - (Z1_2_1 * Cr1) - (Z1_2_1 * Cs2) - (Z1_2_2 * Cr1) - (Z1_2_2 * Cs2)

            // Helper variables for each pair of sequential tasks and each resource
            GRBVar Z1_2_1 =  model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Z1_2_1");
            GRBVar Z1_2_2 =  model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Z1_2_2");
            GRBVar Z3_4_1 =  model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Z3_4_1");
            GRBVar Z3_4_2 =  model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Z3_4_2");

            // Constrain Z to be less than X such that Z is only 1 if BOTH X are 1 for the same r
            GRBLinExpr Z1_1_a = new GRBLinExpr();
            Z1_1_a.addTerm(1.0, Z1_2_1);
            model.addConstr(Z1_1_a, GRB.LESS_EQUAL, X1_1, "Z1_1_a");
            GRBLinExpr Z1_1_b = new GRBLinExpr();
            Z1_1_b.addTerm(1.0, Z1_2_1);
            model.addConstr(Z1_1_b, GRB.LESS_EQUAL, X2_1, "Z1_1_b");
            GRBLinExpr Z2_1_a = new GRBLinExpr();
            Z2_1_a.addTerm(1.0, Z1_2_2);
            model.addConstr(Z2_1_a, GRB.LESS_EQUAL, X2_1, "Z2_1_a");
            GRBLinExpr Z2_1_b = new GRBLinExpr();
            Z2_1_b.addTerm(1.0, Z1_2_2);
            model.addConstr(Z2_1_b, GRB.LESS_EQUAL, X2_2, "Z2_1_b");


            // Resource constraints for each resource and each pair of tasks
            // N1 + N2
            // Resource 1
            ilps.addResourceMappingAllPairConstraint(Ts1, Te1, Tc1, Ts2, Te2, Tc2, Y1, X1_1, X2_1,
                    K, model);
            // Resource 2
            ilps.addResourceMappingAllPairConstraint(Ts1, Te1, Tc1, Ts2, Te2, Tc2, Y2, X1_2, X2_2,
                    K, model);
            // N1 + N3
            ilps.addResourceMappingAllPairConstraint(Ts1, Te1, Tc1, Ts3, Te3, Tc3, Y3, X1_1, X3_1,
                    K, model);
            ilps.addResourceMappingAllPairConstraint(Ts1, Te1, Tc1, Ts3, Te3, Tc3, Y4, X1_2, X3_2,
                    K, model);
            // N1 + N4
            ilps.addResourceMappingAllPairConstraint(Ts1, Te1, Tc1, Ts4, Te4, Tc4, Y5, X1_1, X4_1,
                    K, model);
            ilps.addResourceMappingAllPairConstraint(Ts1, Te1, Tc1, Ts4, Te4, Tc4, Y6, X1_2, X4_2,
                    K, model);
            // N2 + N3
            ilps.addResourceMappingAllPairConstraint(Ts2, Te2, Tc2, Ts3, Te3, Tc3, Y7, X2_1, X3_1,
                    K, model);
            ilps.addResourceMappingAllPairConstraint(Ts2, Te2, Tc2, Ts3, Te3, Tc3, Y8, X2_2, X3_2,
                    K, model);
            // N2 + N4
            ilps.addResourceMappingAllPairConstraint(Ts2, Te2, Tc2, Ts4, Te4, Tc4, Y9, X2_1, X4_1,
                    K, model);
            ilps.addResourceMappingAllPairConstraint(Ts2, Te2, Tc2, Ts4, Te4, Tc4, Y10, X2_2, X4_2,
                    K, model);
            // N3 + N4
            ilps.addResourceMappingAllPairConstraint(Ts3, Te3, Tc3, Ts4, Te4, Tc4, Y11, X3_1, X4_1,
                    K, model);
            ilps.addResourceMappingAllPairConstraint(Ts3, Te3, Tc3, Ts4, Te4, Tc4, Y12, X3_2, X4_2,
                    K, model);

            // Finish times
            // Tf = Ts + Ts + Tc
            ilps.addFinishTimeConstraint(Tf1, Ts1, Te1, Tc1, model);
            ilps.addFinishTimeConstraint(Tf2, Ts2, Te2, Tc2, model);
            ilps.addFinishTimeConstraint(Tf3, Ts3, Te3, Tc3, model);
            ilps.addFinishTimeConstraint(Tf4, Ts4, Te4, Tc4, model);

            // Set objective
            GRBLinExpr obj = new GRBLinExpr();

            // // Minimize the maximum finish time
            // // Maximum finish time
            // GRBVar Tfmax = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "max");
            // model.addGenConstrMax(Tfmax, finish_times, 0.0, "max");
            // obj.addTerm(1.0, Tfmax);

            // Minimize the average finish time of all end tasks
            double[] coeffs = new double[end_finish_times.length];
            Arrays.fill(coeffs, 1.0 / end_finish_times.length);
            obj.addTerms(coeffs, end_finish_times);
            model.setObjective(obj, GRB.MINIMIZE);

            try {
                model.optimize();
            } catch (GRBException e) {
                System.out.println("Model optimization failed");
            }

            System.out.println(obj.getValue());

            System.out.println(String.format("Task 1: %s -> %s: E: %s C: %s",
                    Ts1.get(GRB.DoubleAttr.X), Tf1.get(GRB.DoubleAttr.X), Te1.get(GRB.DoubleAttr.X),
                    Tc1.get(GRB.DoubleAttr.X)));
            System.out.println(String.format("Task 2: %s -> %s: E: %s C: %s",
                    Ts2.get(GRB.DoubleAttr.X), Tf2.get(GRB.DoubleAttr.X), Te2.get(GRB.DoubleAttr.X),
                    Tc2.get(GRB.DoubleAttr.X)));
            System.out.println(String.format("Task 3: %s -> %s: E: %s C: %s",
                    Ts3.get(GRB.DoubleAttr.X), Tf3.get(GRB.DoubleAttr.X), Te3.get(GRB.DoubleAttr.X),
                    Tc3.get(GRB.DoubleAttr.X)));
            System.out.println(String.format("Task 4: %s -> %s: E: %s C: %s",
                    Ts4.get(GRB.DoubleAttr.X), Tf4.get(GRB.DoubleAttr.X), Te4.get(GRB.DoubleAttr.X),
                    Tc4.get(GRB.DoubleAttr.X)));
            for (GRBVar var : mapping_vars)
                System.out.println(String.format("%s: %s", var.get(GRB.StringAttr.VarName),
                        var.get(GRB.DoubleAttr.X)));
            for (GRBVar var : exec_times)
                System.out.println(String.format("%s: %s", var.get(GRB.StringAttr.VarName),
                        var.get(GRB.DoubleAttr.X)));
            for (GRBVar var : comm_times)
                System.out.println(String.format("%s: %s", var.get(GRB.StringAttr.VarName),
                        var.get(GRB.DoubleAttr.X)));

            // Dispose of model and environment
            model.dispose();
            env.dispose();

        } catch (GRBException e) {
            System.out.println("Error code: " + e.getErrorCode() + ". " + e.getMessage());
        }
    }
}
