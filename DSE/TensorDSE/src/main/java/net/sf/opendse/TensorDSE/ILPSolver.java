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

    // Scheduling dependency of the form Ts_j >= Tf_i
    // where task i is a dependency of task j
    public Void addSchedulingDependencyConstraint(GRBVar precursor_task_finish, GRBVar dependent_task_start,
            GRBModel model) {
        GRBLinExpr exp = new GRBLinExpr();
        exp.addTerm(1.0, precursor_task_finish);

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
    // Ts_i >= Tf_j * Ys
    // Ts_j >= Tf_i * (1 - Ys) >= Tf_i - (Tf_i * Ys)

    public Void addResourceMappingPairConstraint(GRBVar task_one_start, GRBVar task_one_finish,
            GRBVar task_two_start, GRBVar task_two_finish, GRBVar Y, GRBModel model) {

        GRBQuadExpr exp1 = new GRBQuadExpr();
        exp1.addTerm(1.0, task_two_finish, Y);

        try {
            model.addQConstr(exp1, GRB.LESS_EQUAL, task_one_start, "");
        } catch (GRBException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        GRBQuadExpr exp2 = new GRBQuadExpr();
        exp2.addTerm(1.0, task_one_finish);
        exp2.addTerm(-1.0, task_one_finish, Y);

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
    // Ts_i >= (Tf_j * y_i_j_r) - K (2 - x_i_r - x_j_r)
    // Ts_j >= (Tf_i * y_i_j_r) - 2K + (K * x_i_r) + (K * x_j_r)

    public Void addResourceMappingAllPairConstraint(GRBVar task_one_start, GRBVar task_one_finish,
            GRBVar task_two_start, GRBVar task_two_finish, GRBVar Y, GRBVar X_one, GRBVar X_two,
            Double K, GRBModel model) {

        GRBQuadExpr exp1 = new GRBQuadExpr();
        exp1.addTerm(1.0, task_two_finish, Y);
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
        exp2.addTerm(1.0, task_one_finish);
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
            // 2.1 ts >= 0
            GRBVar ts1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Ts1");
            GRBVar ts2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Ts2");
            GRBVar ts3 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Ts3");
            GRBVar ts4 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Ts4");

            // Finish times for each task
            GRBVar tf1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tf1");
            GRBVar tf2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tf2");
            GRBVar tf3 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tf3");
            GRBVar tf4 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tf4");

            GRBVar finish_times[] = new GRBVar[] {tf1, tf2, tf3, tf4};
            GRBVar end_finish_times[] = new GRBVar[] {tf2, tf4};

            // Benchmaked execution times, min and max values are the same to create constants
            GRBVar E1_1 = model.addVar(1.0, 1.0, 0.0, GRB.CONTINUOUS, "E1_1");
            GRBVar E1_2 = model.addVar(1.0, 1.0, 0.0, GRB.CONTINUOUS, "E1_2");
            GRBVar E2_1 = model.addVar(1.0, 1.0, 0.0, GRB.CONTINUOUS, "E2_1");
            GRBVar E2_2 = model.addVar(1.0, 1.0, 0.0, GRB.CONTINUOUS, "E2_2");
            GRBVar E3_1 = model.addVar(1.0, 1.0, 0.0, GRB.CONTINUOUS, "E3_1");
            GRBVar E3_2 = model.addVar(1.0, 1.0, 0.0, GRB.CONTINUOUS, "E3_2");
            GRBVar E4_1 = model.addVar(1.0, 1.0, 0.0, GRB.CONTINUOUS, "E4_1");
            GRBVar E4_2 = model.addVar(1.0, 1.0, 0.0, GRB.CONTINUOUS, "E4_2");

            // Total execution times
            GRBVar te1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc1");
            GRBVar te2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc2");
            GRBVar te3 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc3");
            GRBVar te4 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc4");

            // Benchmarked edge communication times
            GRBVar Cs1_1 = model.addVar(1.0, 1.0, 0.0, GRB.CONTINUOUS, "Cs1_1");
            GRBVar Cs1_2 = model.addVar(1.2, 1.2, 0.0, GRB.CONTINUOUS, "Cs1_2");
            GRBVar Cr1_1 = model.addVar(0.8, 0.8, 0.0, GRB.CONTINUOUS, "Cr1_1");
            GRBVar Cr1_2 = model.addVar(1.0, 1.0, 0.0, GRB.CONTINUOUS, "Cr1_2");
            GRBVar Cs2_1 = model.addVar(2.3, 2.3, 0.0, GRB.CONTINUOUS, "Cs2_1");
            GRBVar Cs2_2 = model.addVar(2.0, 2.0, 0.0, GRB.CONTINUOUS, "Cs2_2");
            GRBVar Cr2_1 = model.addVar(0.4, 0.4, 0.0, GRB.CONTINUOUS, "Cr2_1");
            GRBVar Cr2_2 = model.addVar(0.5, 0.5, 0.0, GRB.CONTINUOUS, "Cr2_2");
            GRBVar Cs3_1 = model.addVar(1.0, 1.0, 0.0, GRB.CONTINUOUS, "Cs3_1");
            GRBVar Cs3_2 = model.addVar(1.5, 1.5, 0.0, GRB.CONTINUOUS, "Cs3_2");
            GRBVar Cr3_1 = model.addVar(1.2, 1.2, 0.0, GRB.CONTINUOUS, "Cr3_1");
            GRBVar Cr3_2 = model.addVar(1.0, 1.0, 0.0, GRB.CONTINUOUS, "Cr3_2");
            GRBVar Cs4_1 = model.addVar(1.7, 1.7, 0.0, GRB.CONTINUOUS, "Cs4_1");
            GRBVar Cs4_2 = model.addVar(1.1, 1.1, 0.0, GRB.CONTINUOUS, "Cs4_2");
            GRBVar Cr4_1 = model.addVar(1.4, 1.4, 0.0, GRB.CONTINUOUS, "Cr4_1");
            GRBVar Cr4_2 = model.addVar(1.0, 1.0, 0.0, GRB.CONTINUOUS, "Cr4_2");

            // Communication times

            // Sending from tasks to next tasks, created from benchmarked results
            // and z helper variables that disable comm costs for same device comms
            // Labeled cs_i_j_r, in the example we have for example 1->2 on resources
            // 1 and 2, thus cs_1_2_1 and cs_1_2_2
            GRBVar cs_1_2_1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cs_1_2_1");
            GRBVar cs_1_2_2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cs_1_2_2");
            GRBVar cs_3_4_1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cs_3_4_1");
            GRBVar cs_3_4_2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cs_3_4_2");

            // Total communication times
            GRBVar tc_1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc1");
            GRBVar tc_2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc2");
            GRBVar tc_3 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc3");
            GRBVar tc_4 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc4");

            // Zi_j_r helper variables for when two sequential tasks are on the same resource
            // ie. x_i_r = x_j_r = 1
            GRBVar z_1_2_1 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "Z1_2_1");
            GRBVar z_1_2_2 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "Z1_2_1");
            GRBVar z_3_4_1 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "Z3_4_1");
            GRBVar z_3_4_2 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "Z3_4_1");

            // Add constraints

            // Finish times
            // 2.2 tf >= ts + te + tc
            addFinishTimeConstraint(tf1, ts1, te1, tc_1, model);
            addFinishTimeConstraint(tf2, ts2, te2, tc_2, model);
            addFinishTimeConstraint(tf3, ts3, te3, tc_3, model);
            addFinishTimeConstraint(tf4, ts4, te4, tc_4, model);

            // Execution times
            // 2.3 t3 = sum (E_i_r * x_i_r)

            // Scheduling dependencies of each task, ie. Ts2 >= Tf1
            addSchedulingDependencyConstraint(tf1, ts2, model);
            addSchedulingDependencyConstraint(tf3, ts4, model);

            // Resource mappings for each pair of tasks mapped to the same resource,
            // ie. n1,n2 : n1,n3 : n2,n3
            // Scheduling variables - One for each pair of tasks m apped to same resource
            GRBVar Y1 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y1");
            GRBVar Y2 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y2");
            GRBVar Y3 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y3");

            addResourceMappingPairConstraint(ts1, tf1, ts2, tf2, Y1, model);
            addResourceMappingPairConstraint(ts1, tf1, ts3, tf3, Y2, model);
            addResourceMappingPairConstraint(ts2, tf2, ts3, tf3, Y3, model);



            // Communication costs
            // For each node it's sending and recieving comm costs are the maximum costs from
            // all recieving and sending edges, respectivley.
            //
            // Only communication costs should be used that are being sent/recieved from resource
            // activated by the mapping
            //
            // cs_i >= Cr_j_r * x_j_r
            // cr_i >= Cs_j_r * x_j_r
            //
            // This constraint needs to be set for each task and each task it sends and recieves
            // from, for each resource
            

            
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
                    ts1.get(GRB.DoubleAttr.X), tf1.get(GRB.DoubleAttr.X), te1.get(GRB.DoubleAttr.X),
                    tc_1.get(GRB.DoubleAttr.X)));
            System.out.println(String.format("Task 2: %s -> %s: E: %s C: %s",
                    ts2.get(GRB.DoubleAttr.X), tf2.get(GRB.DoubleAttr.X), te2.get(GRB.DoubleAttr.X),
                    tc_2.get(GRB.DoubleAttr.X)));
            System.out.println(String.format("Task 3: %s -> %s: E: %s C: %s",
                    ts3.get(GRB.DoubleAttr.X), tf3.get(GRB.DoubleAttr.X), te3.get(GRB.DoubleAttr.X),
                    tc_3.get(GRB.DoubleAttr.X)));
            System.out.println(String.format("Task 4: %s -> %s: E: %s C: %s",
                    ts4.get(GRB.DoubleAttr.X), tf4.get(GRB.DoubleAttr.X), te4.get(GRB.DoubleAttr.X),
                    tc_4.get(GRB.DoubleAttr.X)));

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
            ilps.addSchedulingDependencyConstraint(Tf1, Ts2, model);
            ilps.addSchedulingDependencyConstraint(Tf3, Ts4, model);

            // Resource mappings for each pair of tasks and for each resource, Ti,r.
            // ie. n1,1,n2,1 : n1,2,n2,1
            // Scheduling variables - One for each pair of tasks mapped to same  resource
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
            ilps.addResourceMappingAllPairConstraint(Ts1, Tf1, Ts2, Tf2, Y1, X1_1, X2_1, K, model);
            ilps.addResourceMappingAllPairConstraint(Ts1, Tf1, Ts2, Tf2, Y2, X1_2, X2_2, K, model);
            ilps.addResourceMappingAllPairConstraint(Ts1, Tf1, Ts3, Tf3, Y3, X1_1, X3_1, K, model);
            ilps.addResourceMappingAllPairConstraint(Ts1, Tf1, Ts3, Tf3, Y4, X1_2, X3_2, K, model);
            ilps.addResourceMappingAllPairConstraint(Ts1, Tf1, Ts4, Tf4, Y5, X1_1, X4_1, K, model);
            ilps.addResourceMappingAllPairConstraint(Ts1, Tf1, Ts4, Tf4, Y6, X1_2, X4_2, K, model);
            ilps.addResourceMappingAllPairConstraint(Ts2, Tf2, Ts3, Tf3, Y7, X2_1, X3_1, K, model);
            ilps.addResourceMappingAllPairConstraint(Ts2, Tf2, Ts3, Tf3, Y8, X2_2, X3_2, K, model);
            ilps.addResourceMappingAllPairConstraint(Ts2, Tf2, Ts4, Tf4, Y9, X2_1, X4_1, K, model);
            ilps.addResourceMappingAllPairConstraint(Ts2, Tf2, Ts4, Tf4, Y10, X2_2, X4_2, K, model);
            ilps.addResourceMappingAllPairConstraint(Ts3, Tf3, Ts4, Tf4, Y11, X3_1, X4_1, K, model);
            ilps.addResourceMappingAllPairConstraint(Ts3, Tf3, Ts4, Tf4, Y12, X3_2, X4_2, K, model);

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
            GRBVar CR1_1 = model.addVar(0.5, 0.5, 0.0, GRB.CONTINUOUS, "C1_1");
            GRBVar CS1_1 = model.addVar(0.5, 0.5, 0.0, GRB.CONTINUOUS, "C1_1");
            // Communication time for task 1 on resource 2
            GRBVar CR1_2 = model.addVar(2.0, 2.0, 0.0, GRB.CONTINUOUS, "C1_2");
            GRBVar CS1_2 = model.addVar(2.0, 2.0, 0.0, GRB.CONTINUOUS, "C1_2");
            GRBVar CR2_1 = model.addVar(0.5, 0.5, 0.0, GRB.CONTINUOUS, "C2_1");
            GRBVar CS2_1 = model.addVar(0.5, 0.5, 0.0, GRB.CONTINUOUS, "C2_1");
            GRBVar CR2_2 = model.addVar(1.0, 1.0, 0.0, GRB.CONTINUOUS, "C2_2");
            GRBVar CS2_2 = model.addVar(1.0, 1.0, 0.0, GRB.CONTINUOUS, "C2_2");
            GRBVar CR3_1 = model.addVar(1.0, 1.0, 0.0, GRB.CONTINUOUS, "C3_1");
            GRBVar CS3_1 = model.addVar(1.0, 1.0, 0.0, GRB.CONTINUOUS, "C3_1");
            GRBVar CR3_2 = model.addVar(0.5, 0.5, 0.0, GRB.CONTINUOUS, "C3_2");
            GRBVar CS3_2 = model.addVar(0.5, 0.5, 0.0, GRB.CONTINUOUS, "C3_2");
            GRBVar CR4_1 = model.addVar(1.0, 1.0, 0.0, GRB.CONTINUOUS, "C4_1");
            GRBVar CS4_1 = model.addVar(1.0, 1.0, 0.0, GRB.CONTINUOUS, "C4_1");
            GRBVar CR4_2 = model.addVar(0.5, 0.5, 0.0, GRB.CONTINUOUS, "C4_2");
            GRBVar CS4_2 = model.addVar(0.5, 0.5, 0.0, GRB.CONTINUOUS, "C4_2");

            GRBVar[] comm_times = {CR1_1, CS1_1, CR1_2, CS1_2, CR2_1, CS2_1, CR2_2, CS2_2, CR3_1,
                    CS3_1, CR3_2, CS3_2, CR4_1, CS4_1, CR4_2, CS4_2};

            // Add constraints

            // Scheduling dependencies of each task, ie. Ts2 >= Ts1 + e1 + c1
            ilps.addSchedulingDependencyConstraint(Tf1, Ts2, model);
            ilps.addSchedulingDependencyConstraint(Tf3, Ts4, model);

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
            GRBVar Z1_2_1 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Z1_2_1");
            GRBVar Z1_2_2 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Z1_2_2");
            GRBVar Z3_4_1 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Z3_4_1");
            GRBVar Z3_4_2 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Z3_4_2");

            // Constrain Z to be less than X such that Z is only 1 if BOTH X are 1 for the same r
            // X variables have the format Xtask_resource

            // Edge T1->T2 T1 mapping on resource 1, (X1_1)
            GRBLinExpr Z12_1_1 = new GRBLinExpr();
            Z12_1_1.addTerm(1.0, Z1_2_1);
            model.addConstr(Z12_1_1, GRB.LESS_EQUAL, X1_1, "Z12_1_1");
            // Edge T1->T2 T2 mapping on resource 1, (X2_1)
            GRBLinExpr Z12_2_1 = new GRBLinExpr();
            Z12_2_1.addTerm(1.0, Z1_2_1);
            model.addConstr(Z12_2_1, GRB.LESS_EQUAL, X2_1, "Z12_2_1");
            // Edge T1->T2 T1 mapping on resource 2, (X1_2)
            GRBLinExpr Z12_2_2 = new GRBLinExpr();
            Z12_2_2.addTerm(1.0, Z1_2_2);
            model.addConstr(Z12_2_2, GRB.LESS_EQUAL, X1_2, "Z12_2_2");
            // Edge T1->T2 T2 mapping on resource 2, (X2_2)
            GRBLinExpr Z12_1_2 = new GRBLinExpr();
            Z12_1_2.addTerm(1.0, Z1_2_2);
            model.addConstr(Z12_1_2, GRB.LESS_EQUAL, X2_2, "Z12_1_2");

            // Edge T3->T4 T3 mapping on resource 1, (X3_1)
            GRBLinExpr Z34_3_1 = new GRBLinExpr();
            Z34_3_1.addTerm(1.0, Z3_4_1);
            model.addConstr(Z34_3_1, GRB.LESS_EQUAL, X3_1, "Z34_3_1");
            // Edge T3->T4 T4 mapping on resource 1, (X4_1)
            GRBLinExpr Z34_4_1 = new GRBLinExpr();
            Z34_4_1.addTerm(1.0, Z3_4_1);
            model.addConstr(Z34_4_1, GRB.LESS_EQUAL, X4_1, "Z34_4_1");
            // Edge T3->T4 T3 mapping on resource 2, (X3_2)
            GRBLinExpr Z34_3_2 = new GRBLinExpr();
            Z34_3_2.addTerm(1.0, Z3_4_2);
            model.addConstr(Z34_3_2, GRB.LESS_EQUAL, X3_2, "Z34_3_2");
            // Edge T3->T4 T4 mapping on resource 2, (X4_2)
            GRBLinExpr Z34_4_2 = new GRBLinExpr();
            Z34_4_2.addTerm(1.0, Z3_4_2);
            model.addConstr(Z34_4_2, GRB.LESS_EQUAL, X4_2, "Z34_4_2");

            // Communication times
            GRBQuadExpr C12 = new GRBQuadExpr();


            // Resource constraints for each resource and each pair of tasks
             ilps.addResourceMappingAllPairConstraint(Ts1, Tf1, Ts2, Tf2, Y1, X1_1, X2_1, K, model);
            ilps.addResourceMappingAllPairConstraint(Ts1, Tf1, Ts2, Tf2, Y2, X1_2, X2_2, K, model);
            ilps.addResourceMappingAllPairConstraint(Ts1, Tf1, Ts3, Tf3, Y3, X1_1, X3_1, K, model);
            ilps.addResourceMappingAllPairConstraint(Ts1, Tf1, Ts3, Tf3, Y4, X1_2, X3_2, K, model);
            ilps.addResourceMappingAllPairConstraint(Ts1, Tf1, Ts4, Tf4, Y5, X1_1, X4_1, K, model);
            ilps.addResourceMappingAllPairConstraint(Ts1, Tf1, Ts4, Tf4, Y6, X1_2, X4_2, K, model);
            ilps.addResourceMappingAllPairConstraint(Ts2, Tf2, Ts3, Tf3, Y7, X2_1, X3_1, K, model);
            ilps.addResourceMappingAllPairConstraint(Ts2, Tf2, Ts3, Tf3, Y8, X2_2, X3_2, K, model);
            ilps.addResourceMappingAllPairConstraint(Ts2, Tf2, Ts4, Tf4, Y9, X2_1, X4_1, K, model);
            ilps.addResourceMappingAllPairConstraint(Ts2, Tf2, Ts4, Tf4, Y10, X2_2, X4_2, K, model);
            ilps.addResourceMappingAllPairConstraint(Ts3, Tf3, Ts4, Tf4, Y11, X3_1, X4_1, K, model);
            ilps.addResourceMappingAllPairConstraint(Ts3, Tf3, Ts4, Tf4, Y12, X3_2, X4_2, K, model);

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
