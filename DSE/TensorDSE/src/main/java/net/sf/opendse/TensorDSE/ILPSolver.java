package net.sf.opendse.TensorDSE;

import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import gurobi.*;
import net.sf.opendse.model.Application;
import net.sf.opendse.model.Architecture;
import net.sf.opendse.model.Dependency;
import net.sf.opendse.model.Link;
import net.sf.opendse.model.Mappings;
import net.sf.opendse.model.Resource;
import net.sf.opendse.model.Routings;
import net.sf.opendse.model.Specification;
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

    public void gurobiILPExample() {

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

            // GRBVar finish_times[] = new GRBVar[] {Tf1, Tf2, Tf3, Tf4};
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
            ilps.addSchedulingDependencyConstraint(Ts1, E1, C1, Ts2, model);
            ilps.addSchedulingDependencyConstraint(Ts3, E3, C3, Ts4, model);

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
            GRBVar X1_1 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "X1,1");
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

            // Summ constraints
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

            // Resource constraints for each resource and each pair of tasks
            // N1 + N2
            ilps.addResourceMappingAllPairConstraint(Ts1, E1, C1, Ts2, E2, C2, Y1, X1_1, X2_1, K,
                    model);
            ilps.addResourceMappingAllPairConstraint(Ts1, E1, C1, Ts2, E2, C2, Y2, X1_2, X2_2, K,
                    model);
            // N1 + N3
            ilps.addResourceMappingAllPairConstraint(Ts1, E1, C1, Ts3, E3, C3, Y3, X1_1, X3_1, K,
                    model);
            ilps.addResourceMappingAllPairConstraint(Ts1, E1, C1, Ts3, E3, C3, Y4, X1_2, X3_2, K,
                    model);
            // N1 + N4
            ilps.addResourceMappingAllPairConstraint(Ts1, E1, C1, Ts4, E4, C4, Y5, X1_1, X4_1, K,
                    model);
            ilps.addResourceMappingAllPairConstraint(Ts1, E1, C1, Ts4, E4, C4, Y6, X1_2, X4_2, K,
                    model);
            // N2 + N3
            ilps.addResourceMappingAllPairConstraint(Ts2, E2, C2, Ts3, E3, C3, Y7, X2_1, X3_1, K,
                    model);
            ilps.addResourceMappingAllPairConstraint(Ts2, E2, C2, Ts3, E3, C3, Y8, X2_2, X3_2, K,
                    model);
            // N2 + N4
            ilps.addResourceMappingAllPairConstraint(Ts2, E2, C2, Ts4, E4, C4, Y9, X2_1, X4_1, K,
                    model);
            ilps.addResourceMappingAllPairConstraint(Ts2, E2, C2, Ts4, E4, C4, Y10, X2_2, X4_2, K,
                    model);
            // N3 + N4
            ilps.addResourceMappingAllPairConstraint(Ts3, E3, C3, Ts4, E4, C4, Y11, X3_1, X4_1, K,
                    model);
            ilps.addResourceMappingAllPairConstraint(Ts3, E3, C3, Ts4, E4, C4, Y12, X3_2, X4_2, K,
                    model);

            // Finish times
            ilps.addFinishTimeConstraint(Tf1, Ts1, E1, C1, model);
            ilps.addFinishTimeConstraint(Tf2, Ts2, E2, C2, model);
            ilps.addFinishTimeConstraint(Tf3, Ts3, E3, C3, model);
            ilps.addFinishTimeConstraint(Tf4, Ts4, E4, C4, model);

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

    public void solveILP(SpecificationDefinition specification){
        

    }
}
