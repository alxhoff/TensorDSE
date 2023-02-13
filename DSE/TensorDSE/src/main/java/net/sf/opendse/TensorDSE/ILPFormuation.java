package net.sf.opendse.TensorDSE;

import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import org.javatuples.Pair;
import gurobi.*;
import net.sf.opendse.model.Resource;
import net.sf.opendse.model.Task;

public class ILPFormuation {

    public Double K;

    public ILPFormuation() {
        this.K = 100.0;
    }

    public ILPFormuation(Double K) {
        this.K = K;
    }


    /**
     * @param x_vars
     * @param model
     */
    // For each potential mapping of a task to a resource an x variable contains the boolean
    // information for the mapping. For each task the sum of all x variables can be at most 1,
    // ie. a task can only be mapped to one resource.
    // for all i sum_r x_i_r = 1
    public void addResourceMappingConstraint(GRBVar[] x_vars, GRBModel model) {
        GRBLinExpr exp = new GRBLinExpr();

        for (int i = 0; i < x_vars.length; i++)
            exp.addTerm(1.0, x_vars[i]);

        try {
            model.addConstr(exp, GRB.EQUAL, 1.0, "");
        } catch (GRBException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }


    /**
     * @param resulting_cost
     * @param sending_cost
     * @param receiving_cost
     * @param model
     */
    public void addTotalCommunicationCostConstraint(GRBVar resulting_cost, GRBVar sending_cost,
            GRBVar receiving_cost, GRBModel model) {

        GRBLinExpr exp = new GRBLinExpr();

        exp.addTerm(1.0, sending_cost);
        exp.addTerm(1.0, receiving_cost);

        try {
            model.addConstr(exp, GRB.EQUAL, resulting_cost, "");
        } catch (GRBException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }


    /**
     * @param resulting_cost
     * @param cost
     * @param mapping_var
     * @param model
     */
    public void addCommunicationCostSelectionConstraint(GRBVar resulting_cost, GRBVar cost,
            GRBVar mapping_var, GRBModel model) {

        GRBQuadExpr exp = new GRBQuadExpr();

        exp.addTerm(1.0, cost, mapping_var);

        try {
            model.addQConstr(exp, GRB.LESS_EQUAL, resulting_cost, "");
        } catch (GRBException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    public void addCommunicationCostConstraint(GRBVar resulting_cost, GRBVar cost, GRBModel model) {

        GRBLinExpr exp = new GRBLinExpr();

        exp.addTerm(1.0, cost);

        try {
            model.addConstr(exp, GRB.LESS_EQUAL, resulting_cost, "");
        } catch (GRBException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }


    /**
     * @param resulting_cost
     * @param benchmarked_cost
     * @param z_helper
     * @param model
     */
    // Eg.
    // cs_i_j_r = Cs_i_r(1 - z_i_j_r)
    // cs_i_j_r = Cs_i_r - (Cs_i_r * z_i_j_r)
    public void addSameResourceCommunicationCostConstraint(GRBVar resulting_cost,
            GRBVar benchmarked_cost, GRBVar z_helper, GRBModel model) {

        GRBQuadExpr exp = new GRBQuadExpr();

        exp.addTerm(1.0, benchmarked_cost);
        exp.addTerm(-1.0, benchmarked_cost, z_helper);

        try {
            model.addQConstr(exp, GRB.EQUAL, resulting_cost, "");
        } catch (GRBException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }


    /**
     * @param precursor_task_finish
     * @param dependent_task_start
     * @param model
     */
    // Scheduling dependency of the form Ts_j >= Tf_i
    // where task i is a dependency of task j
    public void addTaskSchedulingDependencyConstraint(GRBVar precursor_task_finish,
            GRBVar dependent_task_start, GRBModel model) {
        GRBLinExpr exp = new GRBLinExpr();
        exp.addTerm(1.0, precursor_task_finish);

        try {
            model.addConstr(exp, GRB.LESS_EQUAL, dependent_task_start, "");
        } catch (GRBException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }


    /**
     * @param result_var
     * @param input_var_one
     * @param input_var_two
     * @param model
     */
    public void addPairAndConstrint(GRBVar result_var, GRBVar input_var_one, GRBVar input_var_two,
            GRBModel model) {
        // // y >= x1 + x2 - 1, y <= x1, y <= x2, 0 <= y <= 1
        // GRBLinExpr exp = new GRBLinExpr();
        // exp.addTerm(1.0, input_var_one);
        // exp.addTerm(1.0, input_var_two);
        // exp.addConstant(-1.0);
        // try {
        // model.addConstr(exp, GRB.LESS_EQUAL, result_var, "");
        // } catch (GRBException e) {
        // // TODO Auto-generated catch block
        // e.printStackTrace();
        // }

        // exp = new GRBLinExpr();
        // exp.addTerm(1.0, result_var);
        // try {
        // model.addConstr(exp, GRB.LESS_EQUAL, input_var_one, "");
        // } catch (GRBException e) {
        // // TODO Auto-generated catch block
        // e.printStackTrace();
        // }

        // exp = new GRBLinExpr();
        // exp.addTerm(1.0, result_var);
        // try {
        // model.addConstr(exp, GRB.LESS_EQUAL, input_var_two, "");
        // } catch (GRBException e) {
        // // TODO Auto-generated catch block
        // e.printStackTrace();
        // }

        GRBVar[] input_vars = {input_var_one, input_var_two};
        try {
            model.addGenConstrAnd(result_var, input_vars, "");
        } catch (GRBException e) {
            System.out.println("wait here");
            e.printStackTrace();
        }
    }


    /**
     * @param result_var
     * @param vars1
     * @param vars2
     * @param model
     */
    public void addSumOfVectorsConstraint(GRBVar result_var, GRBVar[] vars1, GRBVar[] vars2,
            GRBModel model) {

        if (vars1.length != vars2.length)
            return;

        GRBQuadExpr exp = new GRBQuadExpr();
        for (int i = 0; i < vars1.length; i++)
            exp.addTerm(1.0, vars1[i], vars2[i]);

        try {
            model.addQConstr(exp, GRB.EQUAL, result_var, "");
        } catch (GRBException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }


    /**
     * @param task_one_start
     * @param task_one_finish
     * @param task_two_start
     * @param task_two_finish
     * @param Y
     * @param model
     * @return Void
     */
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


    /**
     * @param task_one_start
     * @param task_one_finish
     * @param task_two_start
     * @param task_two_finish
     * @param Y
     * @param X_one
     * @param X_two
     * @param K
     * @param model
     * @return Void
     */
    // When solving the mapping via the ILP, there should be a pair-wise resource mapping
    // constraints placed on all possible pairs of tasks.
    //
    // Ts_i >= (Tf_j * y_i_j_r) - K (2 - x_i_r - x_j_r)
    // Ts_j >= Tf_i - (Tf_i * y_i_j_r) - 2K + (K * x_i_r) + (K * x_j_r)

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
        exp2.addTerm(-1.0, task_one_finish, Y);
        exp2.addConstant(-2.0 * K);
        exp2.addTerm(K, X_one);
        exp2.addTerm(K, X_two);

        try {
            model.addQConstr(exp2, GRB.LESS_EQUAL, task_two_start, "");
        } catch (GRBException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        return null;
    }

    public void addDirectExecutionTimeConstraint(GRBVar resulting_cost, GRBVar benchmarked_time,
            GRBModel model) {

        GRBLinExpr exp = new GRBLinExpr();

        exp.addTerm(1.0, benchmarked_time);

        try {
            model.addConstr(exp, GRB.EQUAL, resulting_cost, "");
        } catch (GRBException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }



    /**
     * @param task_finish
     * @param task_start
     * @param task_exec_time
     * @param task_comm_time
     * @param model
     * @return Void
     */
    // Finish times are simply the start time summed with the execution time and communication time
    // Tf = Ts + E1 + C1
    public void addFinishTimeConstraint(GRBVar task_finish, GRBVar task_start,
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
    }


    /**
     * @param task_id
     * @param model
     * @return ILPTask
     */
    private ILPTask initILPTaskBase(Task task, GRBModel model) {

        ILPTask ret = new ILPTask(model);

        ret.setID(task.getId());
        ret.setTask(task);

        return ret;
    }


    /**
     * @param task_id
     * @param target_resource
     * @param comm_cost
     * @param exec_cost
     * @param model
     * @return ILPTask
     */
    public ILPTask initILPTask(Task task, Resource target_resource, Pair<Double, Double> comm_cost,
            Double exec_cost, GRBModel model) {

        ILPTask ret = initILPTaskBase(task, model);

        ret.setSend_cost(comm_cost.getValue0());
        ret.setRecv_cost(comm_cost.getValue1());
        ret.setExecution_cost(exec_cost);
        ret.setTarget_resource(target_resource);

        return ret;
    }


    /**
     * @param task_id
     * @param target_resources
     * @param comm_costs
     * @param HashMap<Resource
     * @param exec_costs
     * @param model
     * @return ILPTask
     */
    public ILPTask initILPTask(Task task, ArrayList<Resource> target_resources,
            HashMap<Resource, Pair<Double, Double>> comm_costs,
            HashMap<Resource, Double> exec_costs, GRBModel model) {

        ILPTask ret = initILPTaskBase(task, model);

        HashMap<Resource, Double> send_costs = new HashMap<Resource, Double>();
        for (Map.Entry<Resource, Pair<Double, Double>> entry : comm_costs.entrySet()) {
            send_costs.put(entry.getKey(), entry.getValue().getValue0());
        }
        ret.setSend_costs(send_costs);

        HashMap<Resource, Double> recv_costs = new HashMap<Resource, Double>();
        for (Map.Entry<Resource, Pair<Double, Double>> entry : comm_costs.entrySet()) {
            recv_costs.put(entry.getKey(), entry.getValue().getValue1());
        }
        ret.setRecv_costs(recv_costs);

        ret.setExecution_costs(exec_costs);
        ret.setTarget_resources(target_resources);

        return ret;
    }


    public void gurobiDSEExampleSixTask() {
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

            // 2.1 Start times for each task
            // ts >= 0
            GRBVar ts_1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Ts1");
            GRBVar ts_2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Ts2");
            GRBVar ts_3 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Ts3");

            GRBVar ts_4 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Ts4");
            GRBVar ts_5 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Ts5");
            GRBVar ts_6 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Ts6");

            GRBVar[] start_times = {ts_1, ts_2, ts_3, ts_4, ts_5, ts_6};

            // 2.2 Total execution times
            GRBVar te_1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc1");
            GRBVar te_2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc2");
            GRBVar te_3 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc3");

            GRBVar te_4 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc4");
            GRBVar te_5 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc5");
            GRBVar te_6 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc6");

            GRBVar[] execution_times = {te_1, te_2, te_3, te_4, te_5, te_6};

            // 2.3 Total communication times
            GRBVar tc_1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc1");
            GRBVar tc_2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc2");
            GRBVar tc_3 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc3");

            GRBVar tc_4 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc4");
            GRBVar tc_5 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc5");
            GRBVar tc_6 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc6");

            GRBVar[] communication_times = {tc_1, tc_2, tc_3, tc_4, tc_5, tc_6};

            // 2.4 Finish times
            // tf >= ts + te + tc
            GRBVar tf_1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tf1");
            GRBVar tf_2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tf2");
            GRBVar tf_3 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tf3");

            GRBVar tf_4 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tf4");
            GRBVar tf_5 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tf5");
            GRBVar tf_6 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tf6");

            GRBVar finish_times[] = new GRBVar[] {tf_1, tf_2, tf_3, tf_4, tf_5, tf_6};
            GRBVar end_finish_times[] = new GRBVar[] {tf_3, tf_6};

            addFinishTimeConstraint(tf_1, ts_1, te_1, tc_1, model);
            addFinishTimeConstraint(tf_2, ts_2, te_2, tc_2, model);
            addFinishTimeConstraint(tf_3, ts_3, te_3, tc_3, model);

            addFinishTimeConstraint(tf_4, ts_4, te_4, tc_4, model);
            addFinishTimeConstraint(tf_5, ts_5, te_5, tc_5, model);
            addFinishTimeConstraint(tf_6, ts_6, te_6, tc_6, model);

            // 2.5 Mapping variables x_i_r, 1 if task i is mapped to resource r
            // Any mappings that are not possible should be set to 0.0
            GRBVar x_1_1 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "x_1_1");
            GRBVar x_1_2 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "x_1_2");
            GRBVar[] x_1 = {x_1_1, x_1_2};
            GRBVar x_2_1 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "x_2_1");
            GRBVar x_2_2 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "x_2_2");
            GRBVar[] x_2 = {x_2_1, x_2_2};
            GRBVar x_3_1 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "x_3_1");
            GRBVar x_3_2 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "x_3_2");
            GRBVar[] x_3 = {x_3_1, x_3_2};

            GRBVar x_4_1 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "x_4_1");
            GRBVar x_4_2 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "x_4_2");
            GRBVar[] x_4 = {x_4_1, x_4_2};
            GRBVar x_5_1 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "x_5_1");
            GRBVar x_5_2 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "x_5_2");
            GRBVar[] x_5 = {x_5_1, x_5_2};
            GRBVar x_6_1 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "x_6_1");
            GRBVar x_6_2 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "x_6_2");
            GRBVar[] x_6 = {x_6_1, x_6_2};

            GRBVar[][] mapping_vars = {x_1, x_2, x_3, x_4, x_5, x_6};

            // 2.6 Resource mapped execution times
            // tei = sum_r (E_i_r * x_i_r)
            // Benchmaked execution times, min and max values are the same to create constants
            GRBVar E_1_1 = model.addVar(1.0, 1.0, 0.0, GRB.CONTINUOUS, "E1_1");
            GRBVar E_1_2 = model.addVar(2.0, 2.0, 0.0, GRB.CONTINUOUS, "E1_2");
            GRBVar[] E_1 = {E_1_1, E_1_2};
            GRBVar E_2_1 = model.addVar(2.1, 2.1, 0.0, GRB.CONTINUOUS, "E2_1");
            GRBVar E_2_2 = model.addVar(1.1, 1.1, 0.0, GRB.CONTINUOUS, "E2_2");
            GRBVar[] E_2 = {E_2_1, E_2_2};
            GRBVar E_3_1 = model.addVar(1.6, 1.6, 0.0, GRB.CONTINUOUS, "E3_1");
            GRBVar E_3_2 = model.addVar(1.5, 1.5, 0.0, GRB.CONTINUOUS, "E3_2");
            GRBVar[] E_3 = {E_3_1, E_3_2};

            GRBVar E_4_1 = model.addVar(1.2, 1.2, 0.0, GRB.CONTINUOUS, "E4_1");
            GRBVar E_4_2 = model.addVar(2.2, 2.2, 0.0, GRB.CONTINUOUS, "E4_2");
            GRBVar[] E_4 = {E_4_1, E_4_2};
            GRBVar E_5_1 = model.addVar(2.3, 2.3, 0.0, GRB.CONTINUOUS, "E5_1");
            GRBVar E_5_2 = model.addVar(1.3, 1.3, 0.0, GRB.CONTINUOUS, "E5_2");
            GRBVar[] E_5 = {E_5_1, E_5_2};
            GRBVar E_6_1 = model.addVar(1.6, 1.6, 0.0, GRB.CONTINUOUS, "E6_1");
            GRBVar E_6_2 = model.addVar(1.5, 1.5, 0.0, GRB.CONTINUOUS, "E6_2");
            GRBVar[] E_6 = {E_6_1, E_6_2};

            GRBVar[][] benchmarked_execution_times = {E_1, E_2, E_3, E_4, E_5, E_6};

            addSumOfVectorsConstraint(te_1, E_1, x_1, model);
            addSumOfVectorsConstraint(te_2, E_2, x_2, model);
            addSumOfVectorsConstraint(te_3, E_3, x_3, model);

            addSumOfVectorsConstraint(te_4, E_4, x_4, model);
            addSumOfVectorsConstraint(te_5, E_5, x_5, model);
            addSumOfVectorsConstraint(te_6, E_6, x_6, model);

            // 2.7 Task scheduling dependencies
            // Ts2 >= Tf1
            addTaskSchedulingDependencyConstraint(tf_1, ts_2, model);
            addTaskSchedulingDependencyConstraint(tf_2, ts_3, model);

            addTaskSchedulingDependencyConstraint(tf_4, ts_5, model);
            addTaskSchedulingDependencyConstraint(tf_5, ts_6, model);

            // 2.8 Same resource communication costs
            // 2.8.1 Z helper variable
            // If a communication between tasks i and j happens on the same resource
            // then the Z helper variable should be set to 1.
            // For each communication edge and for each resoource we need a helper variable.
            // Zi_j_r helper variables for when two sequential tasks are on the same resource
            // ie. x_i_r = x_j_r = 1
            GRBVar z_1_2_1 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "z_1_2_1");
            GRBVar z_1_2_2 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "z_1_2_1");
            GRBVar z_2_3_1 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "z_2_3_1");
            GRBVar z_2_3_2 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "z_2_3_1");

            GRBVar z_4_5_1 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "z_4_5_1");
            GRBVar z_4_5_2 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "z_4_5_1");
            GRBVar z_5_6_1 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "z_5_6_1");
            GRBVar z_5_6_2 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "z_5_6_1");

            String[] z_labels = {"1->2 on R1", "1->2 on R2", "2->3 on R1", "2->3 on R2",
                    "4->5 on R1", "4->5 on R2", "5->6 on R1", "5->6 on R2"};
            GRBVar[] z_variables =
                    {z_1_2_1, z_1_2_2, z_2_3_1, z_2_3_2, z_4_5_1, z_4_5_2, z_5_6_1, z_5_6_2};

            // eg. between task 1 and 2 on resource 1, x_1_1 and x_2_1 must be the same if
            // z_1_2_1 is to be 1
            addPairAndConstrint(z_1_2_1, x_1_1, x_2_1, model);
            addPairAndConstrint(z_1_2_2, x_1_2, x_2_2, model);
            addPairAndConstrint(z_2_3_1, x_2_1, x_3_1, model);
            addPairAndConstrint(z_2_3_2, x_2_2, x_3_2, model);

            addPairAndConstrint(z_4_5_1, x_4_1, x_5_1, model);
            addPairAndConstrint(z_4_5_2, x_4_2, x_5_2, model);
            addPairAndConstrint(z_5_6_1, x_5_1, x_6_1, model);
            addPairAndConstrint(z_5_6_2, x_5_2, x_6_2, model);

            // 2.8.2 Same resource sending
            // cs_i_j_r = Cs_i_r(1 - sum_r z_i_j_r)

            // Benchmarked edge communication times
            // End tasks don't send to any further tasks

            GRBVar Cs_1_1 = model.addVar(0.5, 0.5, 0.0, GRB.CONTINUOUS, "Cs_1_1");
            GRBVar Cs_1_2 = model.addVar(0.6, 0.6, 0.0, GRB.CONTINUOUS, "Cs_1_2");
            GRBVar Cs_2_1 = model.addVar(0.33, 0.33, 0.0, GRB.CONTINUOUS, "Cs_2_1");
            GRBVar Cs_2_2 = model.addVar(0.44, 0.44, 0.0, GRB.CONTINUOUS, "Cs_2_2");
            GRBVar Cs_3_1 = model.addVar(0.33, 0.33, 0.0, GRB.CONTINUOUS, "Cs_3_1");
            GRBVar Cs_3_2 = model.addVar(0.44, 0.44, 0.0, GRB.CONTINUOUS, "Cs_3_2");

            GRBVar Cs_4_1 = model.addVar(0.4, 0.4, 0.0, GRB.CONTINUOUS, "Cs_4_1");
            GRBVar Cs_4_2 = model.addVar(0.5, 0.5, 0.0, GRB.CONTINUOUS, "Cs_4_2");
            GRBVar Cs_5_1 = model.addVar(0.64, 0.64, 0.0, GRB.CONTINUOUS, "Cs_5_1");
            GRBVar Cs_5_2 = model.addVar(0.34, 0.34, 0.0, GRB.CONTINUOUS, "Cs_5_2");
            GRBVar Cs_6_1 = model.addVar(0.64, 0.64, 0.0, GRB.CONTINUOUS, "Cs_6_1");
            GRBVar Cs_6_2 = model.addVar(0.34, 0.34, 0.0, GRB.CONTINUOUS, "Cs_6_2");

            GRBVar[][] benchmarked_sending_times = {{Cs_1_1, Cs_1_2}, {Cs_2_1, Cs_2_2},
                    {Cs_3_1, Cs_3_2}, {Cs_4_1, Cs_4_2}, {Cs_5_1, Cs_5_2}, {Cs_6_1, Cs_6_2}};

            // Sending from tasks to next tasks, created from benchmarked results
            // and z helper variables that disable comm costs for same device comms
            // Labeled cs_i_j_r, in the example we have for example 1->2 on resources
            // 1 and 2, thus cs_1_2_1 and cs_1_2_2
            GRBVar cs_1_2_1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cs_1_2_1");
            GRBVar cs_1_2_2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cs_1_2_2");
            GRBVar cs_2_3_1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cs_2_3_1");
            GRBVar cs_2_3_2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cs_2_3_2");
            GRBVar cs_3_x_1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cs_3_x_1");
            GRBVar cs_3_x_2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cs_3_x_2");

            GRBVar cs_4_5_1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cs_4_5_1");
            GRBVar cs_4_5_2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cs_4_5_2");
            GRBVar cs_5_6_1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cs_5_6_1");
            GRBVar cs_5_6_2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cs_5_6_2");
            GRBVar cs_6_x_1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cs_6_x_1");
            GRBVar cs_6_x_2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cs_6_x_2");

            GRBVar[][] same_resource_sending_times =
                    {{cs_1_2_1, cs_1_2_2}, {cs_2_3_1, cs_2_3_2}, {cs_3_x_1, cs_3_x_2},
                            {cs_4_5_1, cs_4_5_2}, {cs_5_6_1, cs_5_6_2}, {cs_6_x_1, cs_6_x_2}};

            // cs_i_j_r, Cs_i_r, z_i_j_r
            // Sending times are concerned with the benchmarked time from the sending task,
            // ie. for cs_1_2_1, task 1 is sending, thus Cs_1_1 is our benchmark
            addSameResourceCommunicationCostConstraint(cs_1_2_1, Cs_1_1, z_1_2_1, model);
            addSameResourceCommunicationCostConstraint(cs_1_2_2, Cs_1_2, z_1_2_2, model);
            addSameResourceCommunicationCostConstraint(cs_2_3_1, Cs_2_1, z_2_3_1, model);
            addSameResourceCommunicationCostConstraint(cs_2_3_2, Cs_2_2, z_2_3_2, model);

            addSameResourceCommunicationCostConstraint(cs_4_5_1, Cs_4_1, z_4_5_1, model);
            addSameResourceCommunicationCostConstraint(cs_4_5_2, Cs_4_2, z_4_5_2, model);
            addSameResourceCommunicationCostConstraint(cs_5_6_1, Cs_5_1, z_5_6_1, model);
            addSameResourceCommunicationCostConstraint(cs_5_6_2, Cs_5_2, z_5_6_2, model);

            // 2.8.3 Same resource receiving
            // cr_i_j_r = Cr_j_r(1 - z_i_j_r)

            // Benchmarked edge receive times
            // Starting tasks dont receive from any tasks
            GRBVar Cr_2_1 = model.addVar(0.35, 0.35, 0.0, GRB.CONTINUOUS, "Cr2_1");
            GRBVar Cr_2_2 = model.addVar(0.45, 0.45, 0.0, GRB.CONTINUOUS, "Cr2_2");
            GRBVar Cr_3_1 = model.addVar(0.36, 0.36, 0.0, GRB.CONTINUOUS, "Cr3_1");
            GRBVar Cr_3_2 = model.addVar(0.46, 0.46, 0.0, GRB.CONTINUOUS, "Cr3_2");

            GRBVar Cr_5_1 = model.addVar(0.25, 0.25, 0.0, GRB.CONTINUOUS, "Cr5_1");
            GRBVar Cr_5_2 = model.addVar(0.3, 0.3, 0.0, GRB.CONTINUOUS, "Cr5_2");
            GRBVar Cr_6_1 = model.addVar(0.35, 0.35, 0.0, GRB.CONTINUOUS, "Cr6_1");
            GRBVar Cr_6_2 = model.addVar(0.25, 0.25, 0.0, GRB.CONTINUOUS, "Cr6_2");

            GRBVar[][] benchmarked_receiving_times = {{}, {Cr_2_1, Cr_2_2}, {Cr_3_1, Cr_3_2}, {},
                    {Cr_5_1, Cr_5_2}, {Cr_6_1, Cr_6_2}};

            GRBVar cr_1_2_1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cr_1_2_1");
            GRBVar cr_1_2_2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cr_1_2_2");
            GRBVar cr_2_3_1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cr_2_3_1");
            GRBVar cr_2_3_2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cr_2_3_2");

            GRBVar cr_4_5_1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cr_4_5_1");
            GRBVar cr_4_5_2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cr_4_5_2");
            GRBVar cr_5_6_1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cr_5_6_1");
            GRBVar cr_5_6_2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cr_5_6_2");

            GRBVar[][] same_resource_receiving_times = {{}, {cr_1_2_1, cr_1_2_2},
                    {cr_2_3_1, cr_2_3_2}, {}, {cr_4_5_1, cr_4_5_2}, {cr_5_6_1, cr_5_6_2}};

            // cs_i_j_r, Cs_i_r, sum_r z_i_j_r
            // Receiving tasks are concerned with the benchmarked time from the receiving task,
            // ie. for cs_1_2_1, task 2 is receiving, this Cr_2_1 is our benchmark
            addSameResourceCommunicationCostConstraint(cr_1_2_1, Cr_2_1, z_1_2_1, model);
            addSameResourceCommunicationCostConstraint(cr_1_2_2, Cr_2_2, z_1_2_2, model);
            addSameResourceCommunicationCostConstraint(cr_2_3_1, Cr_3_1, z_2_3_1, model);
            addSameResourceCommunicationCostConstraint(cr_2_3_2, Cr_3_2, z_2_3_2, model);

            addSameResourceCommunicationCostConstraint(cr_4_5_1, Cr_5_1, z_4_5_1, model);
            addSameResourceCommunicationCostConstraint(cr_4_5_2, Cr_5_2, z_4_5_2, model);
            addSameResourceCommunicationCostConstraint(cr_5_6_1, Cr_6_1, z_5_6_1, model);
            addSameResourceCommunicationCostConstraint(cr_5_6_2, Cr_6_2, z_5_6_2, model);

            // 2.9 Communication cost selection

            // Assuming parallel communication is possible, the largest communication cost
            // from all incoming or outgoing tasks is what is taken as the communication cost.
            // This value changes depending on which resource the task of interest is mapped onto
            // they the mapping helper variables x are used to give us the cost from the
            // correct resource.

            GRBVar cs_1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cs_1");
            GRBVar cs_2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cs_2");
            GRBVar cs_3 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cs_3");

            GRBVar cs_4 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cs_4");
            GRBVar cs_5 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cs_5");
            GRBVar cs_6 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cs_6");

            GRBVar[] selected_sending_comm_costs = {cs_1, cs_2, cs_3, cs_4, cs_5, cs_6};

            // 2.9.1 Sending communication costs
            // For all edges j->i, cs_i = max_j sum_r cs_i_j_r * x_j_r

            // sending costs for a given edge on a given resource are toggled using
            // the x mapping variables
            // We can ignore the sending costs for the end tasks as that will be zero
            // Since it's sending, the sending task's x variable is used
            addCommunicationCostSelectionConstraint(cs_1, cs_1_2_1, x_1_1, model);
            addCommunicationCostSelectionConstraint(cs_1, cs_1_2_2, x_1_2, model);
            addCommunicationCostSelectionConstraint(cs_2, cs_2_3_1, x_2_1, model);
            addCommunicationCostSelectionConstraint(cs_2, cs_2_3_2, x_2_2, model);

            addCommunicationCostSelectionConstraint(cs_4, cs_4_5_1, x_4_1, model);
            addCommunicationCostSelectionConstraint(cs_4, cs_4_5_2, x_4_2, model);
            addCommunicationCostSelectionConstraint(cs_5, cs_5_6_1, x_5_1, model);
            addCommunicationCostSelectionConstraint(cs_5, cs_5_6_2, x_5_2, model);

            // 2.9.2 Receiving communication costs
            // For all edges j->i, cr_i = max_j sum_r cr_i_j_r * x_j_r
            // Linearized to: for all edges i->j and for all r, cr_i >= cr_i_j_r * x_j_r

            // Task 2 receives from task 1 and task 4 receives from task 3
            GRBVar cr_1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cr_1");
            GRBVar cr_2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cr_2");
            GRBVar cr_3 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cr_3");

            GRBVar cr_4 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cr_4");
            GRBVar cr_5 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cr_5");
            GRBVar cr_6 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cr_6");

            GRBVar[] selected_receiving_comm_costs = {cr_1, cr_2, cr_3, cr_4, cr_5, cr_6};

            // Receiving so the receiving task's x variable is used
            addCommunicationCostSelectionConstraint(cr_2, cr_1_2_1, x_2_1, model);
            addCommunicationCostSelectionConstraint(cr_2, cr_1_2_2, x_2_2, model);
            addCommunicationCostSelectionConstraint(cr_3, cr_2_3_1, x_3_1, model);
            addCommunicationCostSelectionConstraint(cr_3, cr_2_3_2, x_3_2, model);

            addCommunicationCostSelectionConstraint(cr_5, cr_4_5_1, x_5_1, model);
            addCommunicationCostSelectionConstraint(cr_5, cr_4_5_2, x_5_2, model);
            addCommunicationCostSelectionConstraint(cr_6, cr_5_6_1, x_6_1, model);
            addCommunicationCostSelectionConstraint(cr_6, cr_5_6_2, x_6_2, model);

            // 2.10 Total communication costs
            // Total communication time is the sum of sending and receiving times for each task
            // tc_i = cs_i + cr_i
            addTotalCommunicationCostConstraint(tc_1, cs_1, cr_1, model);
            addTotalCommunicationCostConstraint(tc_2, cs_2, cr_2, model);
            addTotalCommunicationCostConstraint(tc_3, cs_3, cr_3, model);

            addTotalCommunicationCostConstraint(tc_4, cs_4, cr_4, model);
            addTotalCommunicationCostConstraint(tc_5, cs_5, cr_5, model);
            addTotalCommunicationCostConstraint(tc_6, cs_6, cr_6, model);

            // 5.1 Resource mapping
            // A task can be mapped to exactly one resource
            addResourceMappingConstraint(x_1, model);
            addResourceMappingConstraint(x_2, model);
            addResourceMappingConstraint(x_3, model);

            addResourceMappingConstraint(x_4, model);
            addResourceMappingConstraint(x_5, model);
            addResourceMappingConstraint(x_6, model);

            // 5.2 Resource sharing
            // For each pair of tasks a helper variable is used to resolve both tasks having
            // conflicting
            // mappings onto the same resource. 6 tasks, therefore need n(n-1)/2 pairs * #r, ie. 30
            // pairs
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
            GRBVar Y13 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y13");
            GRBVar Y14 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y14");
            GRBVar Y15 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y15");

            // 1-2 r1
            addResourceMappingAllPairConstraint(ts_1, tf_1, ts_2, tf_2, Y1, x_1_1, x_2_1, this.K,
                    model);
            // 1-2 r2
            addResourceMappingAllPairConstraint(ts_1, tf_1, ts_2, tf_2, Y1, x_1_2, x_2_2, this.K,
                    model);
            // 1-3 r1
            addResourceMappingAllPairConstraint(ts_1, tf_1, ts_3, tf_3, Y2, x_1_1, x_3_1, this.K,
                    model);
            // 1-3 r2
            addResourceMappingAllPairConstraint(ts_1, tf_1, ts_3, tf_3, Y2, x_1_2, x_3_2, this.K,
                    model);
            // 1-4 r1
            addResourceMappingAllPairConstraint(ts_1, tf_1, ts_4, tf_4, Y3, x_1_1, x_4_1, this.K,
                    model);
            // 1-4 r2
            addResourceMappingAllPairConstraint(ts_1, tf_1, ts_4, tf_4, Y3, x_1_2, x_4_2, this.K,
                    model);
            // 1-5 r1
            addResourceMappingAllPairConstraint(ts_1, tf_1, ts_5, tf_5, Y4, x_1_1, x_5_1, this.K,
                    model);
            // 1-5 r2
            addResourceMappingAllPairConstraint(ts_1, tf_1, ts_5, tf_5, Y4, x_1_2, x_5_2, this.K,
                    model);
            // 1-6 r1
            addResourceMappingAllPairConstraint(ts_1, tf_1, ts_6, tf_6, Y5, x_1_1, x_6_1, this.K,
                    model);
            // 1-6 r2
            addResourceMappingAllPairConstraint(ts_1, tf_1, ts_6, tf_6, Y5, x_1_2, x_6_2, this.K,
                    model);
            // 2-3 r1
            addResourceMappingAllPairConstraint(ts_2, tf_2, ts_3, tf_3, Y6, x_2_1, x_3_1, this.K,
                    model);
            // 2-3 r2
            addResourceMappingAllPairConstraint(ts_2, tf_2, ts_3, tf_3, Y6, x_2_2, x_3_2, this.K,
                    model);
            // 2-4 r1
            addResourceMappingAllPairConstraint(ts_2, tf_2, ts_4, tf_4, Y7, x_2_1, x_4_1, this.K,
                    model);
            // 2-4 r2
            addResourceMappingAllPairConstraint(ts_2, tf_2, ts_4, tf_4, Y7, x_2_2, x_4_2, this.K,
                    model);
            // 2-5 r1
            addResourceMappingAllPairConstraint(ts_2, tf_2, ts_5, tf_5, Y8, x_2_1, x_5_1, this.K,
                    model);
            // 2-5 r2
            addResourceMappingAllPairConstraint(ts_2, tf_2, ts_5, tf_5, Y8, x_2_2, x_5_2, this.K,
                    model);
            // 2-6 r1
            addResourceMappingAllPairConstraint(ts_2, tf_2, ts_6, tf_6, Y9, x_2_1, x_6_1, this.K,
                    model);
            // 2-6 r2
            addResourceMappingAllPairConstraint(ts_2, tf_2, ts_6, tf_6, Y9, x_2_2, x_6_2, this.K,
                    model);
            // 3-4 r1
            addResourceMappingAllPairConstraint(ts_3, tf_3, ts_4, tf_4, Y10, x_3_1, x_4_1, this.K,
                    model);
            // 3-4 r2
            addResourceMappingAllPairConstraint(ts_3, tf_3, ts_4, tf_4, Y10, x_3_2, x_4_2, this.K,
                    model);
            // 3-5 r1
            addResourceMappingAllPairConstraint(ts_3, tf_3, ts_5, tf_5, Y11, x_3_1, x_5_1, this.K,
                    model);
            // 3-5 r2
            addResourceMappingAllPairConstraint(ts_3, tf_3, ts_5, tf_5, Y11, x_3_2, x_5_2, this.K,
                    model);
            // 3-6 r1
            addResourceMappingAllPairConstraint(ts_3, tf_3, ts_6, tf_6, Y12, x_3_1, x_6_1, this.K,
                    model);
            // 3-6 r2
            addResourceMappingAllPairConstraint(ts_3, tf_3, ts_6, tf_6, Y12, x_3_2, x_6_2, this.K,
                    model);
            // 4-5 r1
            addResourceMappingAllPairConstraint(ts_4, tf_4, ts_5, tf_5, Y13, x_4_1, x_5_1, this.K,
                    model);
            // 4-5 r2
            addResourceMappingAllPairConstraint(ts_4, tf_4, ts_5, tf_5, Y13, x_4_2, x_5_2, this.K,
                    model);
            // 4-6 r1
            addResourceMappingAllPairConstraint(ts_4, tf_4, ts_6, tf_6, Y14, x_4_1, x_6_1, this.K,
                    model);
            // 4-6 r2
            addResourceMappingAllPairConstraint(ts_4, tf_4, ts_6, tf_6, Y14, x_4_2, x_6_2, this.K,
                    model);
            // 5-6 r1
            addResourceMappingAllPairConstraint(ts_5, tf_5, ts_6, tf_6, Y15, x_5_1, x_6_1, this.K,
                    model);
            // 5-6 r2
            addResourceMappingAllPairConstraint(ts_5, tf_5, ts_6, tf_6, Y15, x_5_2, x_6_2, this.K,
                    model);


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
            System.out.println();
            System.out.println();

            for (int i = 0; i < z_variables.length; i++) {
                System.out.println(
                        String.format("%s: %f", z_labels[i], z_variables[i].get(GRB.DoubleAttr.X)));
            }
            System.out.println();

            for (int i = 0; i < start_times.length; i++) {
                System.out.print(String.format("Task %d on resource ", i + 1));
                for (int j = 0; j < mapping_vars[i].length; j++)
                    if (mapping_vars[i][j].get(GRB.DoubleAttr.X) > 0.0)
                        System.out.println(String.format("%d", j + 1));
                System.out.println(String.format("Start: %f, finish: %f, exec: %f",
                        start_times[i].get(GRB.DoubleAttr.X), finish_times[i].get(GRB.DoubleAttr.X),
                        execution_times[i].get(GRB.DoubleAttr.X)));
                System.out.println(String.format("Comm: %f, send: %f, recv: %f",
                        communication_times[i].get(GRB.DoubleAttr.X),
                        selected_sending_comm_costs[i].get(GRB.DoubleAttr.X),
                        selected_receiving_comm_costs[i].get(GRB.DoubleAttr.X)));

                System.out.println("------------------------------------------------");
                System.out.println("Benchmarks");
                System.out.print("Execution times,  ");
                for (int j = 0; j < benchmarked_execution_times[i].length; j++)
                    System.out.print(String.format("Resource %d: %f, ", j + 1,
                            benchmarked_execution_times[i][j].get(GRB.DoubleAttr.X)));
                System.out.println();
                System.out.print("Sending times,  ");
                if (benchmarked_sending_times[i].length == 0)
                    System.out.print("None");
                for (int j = 0; j < benchmarked_sending_times[i].length; j++)
                    if (benchmarked_sending_times[i].length > 0)
                        System.out.print(String.format("Resource %d: %f, ", j + 1,
                                benchmarked_sending_times[i][j].get(GRB.DoubleAttr.X)));
                System.out.println();
                System.out.print("Receiving times,  ");
                if (benchmarked_receiving_times[i].length == 0)
                    System.out.print("None");
                for (int j = 0; j < benchmarked_receiving_times[i].length; j++)
                    if (benchmarked_receiving_times[i].length > 0)
                        System.out.print(String.format("Resource %d: %f, ", j + 1,
                                benchmarked_receiving_times[i][j].get(GRB.DoubleAttr.X)));
                System.out.println();

                System.out.println("------------------------------------------------");
                System.out.println("Same resource sending times");
                if (same_resource_sending_times[i].length == 0)
                    System.out.print("None");
                for (int j = 0; j < same_resource_sending_times[i].length; j++)
                    if (same_resource_sending_times[i].length > 0)
                        System.out.print(String.format("Resource %d: %f, ", j + 1,
                                same_resource_sending_times[i][j].get(GRB.DoubleAttr.X)));
                System.out.println();

                System.out.println("Same resource receiving times");
                if (same_resource_receiving_times[i].length == 0)
                    System.out.print("None");
                for (int j = 0; j < same_resource_receiving_times[i].length; j++)
                    if (same_resource_receiving_times[i].length > 0)
                        System.out.print(String.format("Resource %d: %f, ", j + 1,
                                same_resource_receiving_times[i][j].get(GRB.DoubleAttr.X)));
                System.out.println();

                System.out.println();
                System.out.println("####################################################");
                System.out.println();
            }

            for (int i = 0; i < mapping_vars.length; i++)
                System.out.println(String.format("Task %d - x1:%f, x2:%f", i + 1,
                        mapping_vars[i][0].get(GRB.DoubleAttr.X),
                        mapping_vars[i][1].get(GRB.DoubleAttr.X)));

            // Dispose of model and environment
            model.dispose();
            env.dispose();

        } catch (GRBException e) {
            System.out.println("Error code: " + e.getErrorCode() + ". " + e.getMessage());
        }
    }

    public void gurobiDSEExampleFourTask() {
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

            // Create variables

            // 2.1 Start times for each task
            // ts >= 0
            GRBVar ts_1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Ts1");
            GRBVar ts_2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Ts2");
            GRBVar ts_3 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Ts3");
            GRBVar ts_4 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Ts4");

            GRBVar[] start_times = {ts_1, ts_2, ts_3, ts_4};

            // 2.2 Total execution times
            GRBVar te_1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc1");
            GRBVar te_2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc2");
            GRBVar te_3 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc3");
            GRBVar te_4 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc4");

            GRBVar[] execution_times = {te_1, te_2, te_3, te_4};

            // 2.3 Total communication times
            GRBVar tc_1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc1");
            GRBVar tc_2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc2");
            GRBVar tc_3 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc3");
            GRBVar tc_4 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tc4");

            GRBVar[] communication_times = {tc_1, tc_2, tc_3, tc_4};

            // 2.4 Finish times
            // tf >= ts + te + tc
            GRBVar tf_1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tf1");
            GRBVar tf_2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tf2");
            GRBVar tf_3 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tf3");
            GRBVar tf_4 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "Tf4");

            GRBVar finish_times[] = new GRBVar[] {tf_1, tf_2, tf_3, tf_4};
            GRBVar end_finish_times[] = new GRBVar[] {tf_2, tf_4};

            addFinishTimeConstraint(tf_1, ts_1, te_1, tc_1, model);
            addFinishTimeConstraint(tf_2, ts_2, te_2, tc_2, model);
            addFinishTimeConstraint(tf_3, ts_3, te_3, tc_3, model);
            addFinishTimeConstraint(tf_4, ts_4, te_4, tc_4, model);

            // 2.5 Mapping variables

            // Mapping variables x_i_r, 1 if task i is mapped to resource r
            GRBVar x_1_1 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "x_1_1");
            GRBVar x_1_2 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "x_1_2");
            GRBVar[] x_1 = {x_1_1, x_1_2};
            GRBVar x_2_1 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "x_2_1");
            GRBVar x_2_2 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "x_2_2");
            GRBVar[] x_2 = {x_2_1, x_2_2};
            GRBVar x_3_1 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "x_3_1");
            GRBVar x_3_2 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "x_3_2");
            GRBVar[] x_3 = {x_3_1, x_3_2};
            GRBVar x_4_1 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "x_4_1");
            GRBVar x_4_2 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "x_4_2");
            GRBVar[] x_4 = {x_4_1, x_4_2};
            GRBVar[][] mapping_vars = {x_1, x_2, x_3, x_4};

            // 5.1 Resource mapping
            // A task can be mapped to exactly one resource
            addResourceMappingConstraint(x_1, model);
            addResourceMappingConstraint(x_2, model);
            addResourceMappingConstraint(x_3, model);
            addResourceMappingConstraint(x_4, model);

            // 2.6 Resouce mapped execution times
            // tei = sum_r (E_i_r * x_i_r)
            // Benchmaked execution times, min and max values are the same to create constants
            GRBVar E_1_1 = model.addVar(1.0, 1.0, 0.0, GRB.CONTINUOUS, "E1_1");
            GRBVar E_1_2 = model.addVar(2.0, 2.0, 0.0, GRB.CONTINUOUS, "E1_2");
            GRBVar[] E_1 = {E_1_1, E_1_2};
            GRBVar E_2_1 = model.addVar(2.1, 2.1, 0.0, GRB.CONTINUOUS, "E2_1");
            GRBVar E_2_2 = model.addVar(1.1, 1.1, 0.0, GRB.CONTINUOUS, "E2_2");
            GRBVar[] E_2 = {E_2_1, E_2_2};
            GRBVar E_3_1 = model.addVar(1.2, 1.2, 0.0, GRB.CONTINUOUS, "E3_1");
            GRBVar E_3_2 = model.addVar(2.2, 2.2, 0.0, GRB.CONTINUOUS, "E3_2");
            GRBVar[] E_3 = {E_3_1, E_3_2};
            GRBVar E_4_1 = model.addVar(2.3, 2.3, 0.0, GRB.CONTINUOUS, "E4_1");
            GRBVar E_4_2 = model.addVar(1.3, 1.3, 0.0, GRB.CONTINUOUS, "E4_2");
            GRBVar[] E_4 = {E_4_1, E_4_2};

            GRBVar[][] benchmarked_execution_times = {E_1, E_2, E_3, E_4};

            addSumOfVectorsConstraint(te_1, E_1, x_1, model);
            addSumOfVectorsConstraint(te_2, E_2, x_2, model);
            addSumOfVectorsConstraint(te_3, E_3, x_3, model);
            addSumOfVectorsConstraint(te_4, E_4, x_4, model);

            // 2.7 Scheduling dependencies of each task
            // Ts2 >= Tf1
            addTaskSchedulingDependencyConstraint(tf_1, ts_2, model);
            addTaskSchedulingDependencyConstraint(tf_3, ts_4, model);

            // 2.8 Same resource communication costs
            // 2.8.1 Z helper variable
            // If a communication between tasks i and j happens on the same resource
            // then the Z helper variable should be set to 1.
            // For each communication edge and for each resoource we need a helper variable.
            // Zi_j_r helper variables for when two sequential tasks are on the same resource
            // ie. x_i_r = x_j_r = 1
            GRBVar z_1_2_1 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "Z1_2_1");
            GRBVar z_1_2_2 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "Z1_2_1");
            GRBVar z_3_4_1 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "Z3_4_1");
            GRBVar z_3_4_2 = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "Z3_4_1");

            GRBVar[] z_variables = {z_1_2_1, z_1_2_2, z_3_4_1, z_3_4_2};

            addPairAndConstrint(z_1_2_1, x_1_1, x_2_1, model);
            addPairAndConstrint(z_1_2_2, x_1_2, x_2_2, model);
            addPairAndConstrint(z_3_4_1, x_3_1, x_4_1, model);
            addPairAndConstrint(z_3_4_2, x_3_2, x_4_2, model);

            // 2.8.2 Same resource sending
            // cs_i_j_r = Cs_i_r(1 - sum_r z_i_j_r)

            // Benchmarked edge communication times
            // Task 2 and 4 don't send to any further tasks

            GRBVar Cs_1_1 = model.addVar(0.5, 0.5, 0.0, GRB.CONTINUOUS, "Cs1_1");
            GRBVar Cs_1_2 = model.addVar(0.6, 0.6, 0.0, GRB.CONTINUOUS, "Cs1_2");
            GRBVar Cs_3_1 = model.addVar(0.4, 0.4, 0.0, GRB.CONTINUOUS, "Cs3_1");
            GRBVar Cs_3_2 = model.addVar(0.5, 0.5, 0.0, GRB.CONTINUOUS, "Cs3_2");

            GRBVar[][] benchmarked_sending_times = {{Cs_1_1, Cs_1_2}, {}, {Cs_3_1, Cs_3_2}, {}};

            // Sending from tasks to next tasks, created from benchmarked results
            // and z helper variables that disable comm costs for same device comms
            // Labeled cs_i_j_r, in the example we have for example 1->2 on resources
            // 1 and 2, thus cs_1_2_1 and cs_1_2_2
            GRBVar cs_1_2_1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cs_1_2_1");
            GRBVar cs_1_2_2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cs_1_2_2");
            GRBVar cs_3_4_1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cs_3_4_1");
            GRBVar cs_3_4_2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cs_3_4_2");

            GRBVar[][] same_resource_sending_times =
                    {{cs_1_2_1, cs_1_2_2}, {}, {cs_3_4_1, cs_3_4_2}, {}};

            // cs_i_j_r, Cs_i_r, z_i_j_r
            addSameResourceCommunicationCostConstraint(cs_1_2_1, Cs_1_1, z_1_2_1, model);
            addSameResourceCommunicationCostConstraint(cs_1_2_2, Cs_1_2, z_1_2_2, model);
            addSameResourceCommunicationCostConstraint(cs_3_4_1, Cs_3_1, z_3_4_1, model);
            addSameResourceCommunicationCostConstraint(cs_3_4_2, Cs_3_2, z_3_4_2, model);

            // 2.8.3 Same resource receiving
            // cr_i_j_r = Cr_j_r(1 - z_i_j_r)

            // Benchmarked edge receive times
            // Tasks 1 and 3 dont receive from any tasks as they are the starting tasks
            GRBVar Cr_2_1 = model.addVar(0.35, 0.35, 0.0, GRB.CONTINUOUS, "Cr2_1");
            GRBVar Cr_2_2 = model.addVar(0.45, 0.45, 0.0, GRB.CONTINUOUS, "Cr2_2");
            GRBVar Cr_4_1 = model.addVar(0.25, 0.25, 0.0, GRB.CONTINUOUS, "Cr4_1");
            GRBVar Cr_4_2 = model.addVar(0.3, 0.3, 0.0, GRB.CONTINUOUS, "Cr4_2");

            GRBVar[][] benchmarked_receiving_times = {{}, {Cr_2_1, Cr_2_2}, {}, {Cr_4_1, Cr_4_2}};

            GRBVar cr_1_2_1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cr_1_2_1");
            GRBVar cr_1_2_2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cr_1_2_2");
            GRBVar cr_3_4_1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cr_3_4_1");
            GRBVar cr_3_4_2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cr_3_4_2");

            GRBVar[][] same_resource_receiving_times =
                    {{cr_1_2_1, cr_1_2_2}, {}, {cr_3_4_1, cr_3_4_2}, {}};

            // cs_i_j_r, Cs_i_r, sum_r z_i_j_r
            addSameResourceCommunicationCostConstraint(cr_1_2_1, Cr_2_1, z_1_2_1, model);
            addSameResourceCommunicationCostConstraint(cr_1_2_2, Cr_2_2, z_1_2_2, model);
            addSameResourceCommunicationCostConstraint(cr_3_4_1, Cr_4_1, z_3_4_1, model);
            addSameResourceCommunicationCostConstraint(cr_3_4_2, Cr_4_2, z_3_4_2, model);

            // 2.9 Communication cost selection

            // Assuming parallel communication is possible, the largest communication cost
            // from all incoming or outgoing tasks is what is taken as the communication cost.
            // This value changes depending on which resource the task of interest is mapped onto
            // they the mapping helper variables x are used to give us the cost from the
            // correct resource.

            // 2.9.1 Sending communication costs
            // For all edges j->i, cs_i = max_j sum_r cs_i_j_r * x_j_r

            // Task 1 sends to task 2 and task 3 sends to task 4
            GRBVar cs_1 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cs_1");
            GRBVar cs_2 = model.addVar(0.0, 0.0, 0.0, GRB.CONTINUOUS, "cs_2");
            GRBVar cs_3 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cs_3");
            GRBVar cs_4 = model.addVar(0.0, 0.0, 0.0, GRB.CONTINUOUS, "cs_4");

            GRBVar[] selected_sending_comm_costs = {cs_1, cs_2, cs_3, cs_4};

            addCommunicationCostSelectionConstraint(cs_1, cs_1_2_1, x_1_1, model);
            addCommunicationCostSelectionConstraint(cs_1, cs_1_2_2, x_1_2, model);
            addCommunicationCostSelectionConstraint(cs_3, cs_3_4_1, x_3_1, model);
            addCommunicationCostSelectionConstraint(cs_3, cs_3_4_2, x_3_2, model);

            // 2.9.2 Receiving communication costs
            // For all edges j->i, cr_i = max_j sum_r cr_i_j_r * x_j_r
            // Linearized to: for all edges i->j and for all r, cr_i >= cr_i_j_r * x_j_r

            // Task 2 receives from task 1 and task 4 receives from task 3
            GRBVar cr_1 = model.addVar(0.0, 0.0, 0.0, GRB.CONTINUOUS, "cr_1");
            GRBVar cr_2 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cr_2");
            GRBVar cr_3 = model.addVar(0.0, 0.0, 0.0, GRB.CONTINUOUS, "cr_3");
            GRBVar cr_4 = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "cr_4");

            GRBVar[] selected_receiving_comm_costs = {cr_1, cr_2, cr_3, cr_4};

            addCommunicationCostSelectionConstraint(cr_2, cr_1_2_1, x_2_1, model);
            addCommunicationCostSelectionConstraint(cr_2, cr_1_2_2, x_2_2, model);
            addCommunicationCostSelectionConstraint(cr_4, cr_3_4_1, x_4_1, model);
            addCommunicationCostSelectionConstraint(cr_4, cr_3_4_2, x_4_2, model);

            // 2.10 Total communication costs
            // Total communication time is the sum of sending and receiving times for each task
            // tc_i = cs_i + cr_i
            addTotalCommunicationCostConstraint(tc_1, cs_1, cr_1, model);
            addTotalCommunicationCostConstraint(tc_2, cs_2, cr_2, model);
            addTotalCommunicationCostConstraint(tc_3, cs_3, cr_3, model);
            addTotalCommunicationCostConstraint(tc_4, cs_4, cr_4, model);

            // 5.2 Resource sharing
            // For each pair of tasks a helper variable is used to resolve both tasks having
            // conflicting
            // mappings onto the same resource. 4 tasks, therefore need n(n-1)/2 pairs, ie. 6 pairs
            //
            GRBVar Y1 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y1");
            GRBVar Y2 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y2");
            GRBVar Y3 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y3");
            GRBVar Y4 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y4");
            GRBVar Y5 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y5");
            GRBVar Y6 = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "Y6");


            // 1-2 r1
            addResourceMappingAllPairConstraint(ts_1, tf_1, ts_2, tf_2, Y1, x_1_1, x_2_1, this.K,
                    model);
            // 1-2 r2
            addResourceMappingAllPairConstraint(ts_1, tf_1, ts_2, tf_2, Y1, x_1_2, x_2_2, this.K,
                    model);
            // 1-3 r1
            addResourceMappingAllPairConstraint(ts_1, tf_1, ts_3, tf_3, Y2, x_1_1, x_3_1, this.K,
                    model);
            // 1-3 r2
            addResourceMappingAllPairConstraint(ts_1, tf_1, ts_3, tf_3, Y2, x_1_2, x_3_2, this.K,
                    model);
            // 1-4 r1
            addResourceMappingAllPairConstraint(ts_1, tf_1, ts_4, tf_4, Y3, x_1_1, x_4_1, this.K,
                    model);
            // 1-4 r2
            addResourceMappingAllPairConstraint(ts_1, tf_1, ts_4, tf_4, Y3, x_1_2, x_4_2, this.K,
                    model);
            // 2-3 r1
            addResourceMappingAllPairConstraint(ts_2, tf_2, ts_3, tf_3, Y4, x_2_1, x_3_1, this.K,
                    model);
            // 2-3 r2
            addResourceMappingAllPairConstraint(ts_2, tf_2, ts_3, tf_3, Y4, x_2_2, x_3_2, this.K,
                    model);
            // 2-4 r1
            addResourceMappingAllPairConstraint(ts_2, tf_2, ts_4, tf_4, Y5, x_2_1, x_4_1, this.K,
                    model);
            // 2-4 r2
            addResourceMappingAllPairConstraint(ts_2, tf_2, ts_4, tf_4, Y5, x_2_2, x_4_2, this.K,
                    model);
            // 3-4 r1
            addResourceMappingAllPairConstraint(ts_3, tf_3, ts_4, tf_4, Y6, x_3_1, x_4_1, this.K,
                    model);
            // 3-4 r2
            addResourceMappingAllPairConstraint(ts_3, tf_3, ts_4, tf_4, Y6, x_3_2, x_4_2, this.K,
                    model);

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
            System.out.println();
            System.out.println();

            for (int i = 0; i < start_times.length; i++) {
                System.out.println(String.format("Task %d, start: %f, finish: %f, exec: %f", i + 1,
                        start_times[i].get(GRB.DoubleAttr.X), finish_times[i].get(GRB.DoubleAttr.X),
                        execution_times[i].get(GRB.DoubleAttr.X)));
                System.out.println(String.format("Comm: %f, send: %f, recv: %f",
                        communication_times[i].get(GRB.DoubleAttr.X),
                        selected_sending_comm_costs[i].get(GRB.DoubleAttr.X),
                        selected_receiving_comm_costs[i].get(GRB.DoubleAttr.X)));
                for (int j = 0; j < mapping_vars[i].length; j++)
                    if (mapping_vars[i][j].get(GRB.DoubleAttr.X) > 0.0)
                        System.out.println(String.format("Mapped to resource %d", j + 1));

                System.out.println("------------------------------------------------");
                System.out.println("Benchmarks");
                System.out.print("Execution times,  ");
                for (int j = 0; j < benchmarked_execution_times[i].length; j++)
                    System.out.print(String.format("Resource %d: %f, ", j,
                            benchmarked_execution_times[i][j].get(GRB.DoubleAttr.X)));
                System.out.println();
                System.out.print("Sending times,  ");
                if (benchmarked_sending_times[i].length == 0)
                    System.out.print("None");
                for (int j = 0; j < benchmarked_sending_times[i].length; j++)
                    if (benchmarked_sending_times[i].length > 0)
                        System.out.print(String.format("Resource %d: %f, ", j,
                                benchmarked_sending_times[i][j].get(GRB.DoubleAttr.X)));
                System.out.println();

                System.out.print("Receiving times,  ");
                if (benchmarked_receiving_times[i].length == 0)
                    System.out.print("None");
                for (int j = 0; j < benchmarked_receiving_times[i].length; j++)
                    if (benchmarked_receiving_times[i].length > 0)
                        System.out.print(String.format("Resource %d: %f, ", j,
                                benchmarked_receiving_times[i][j].get(GRB.DoubleAttr.X)));
                System.out.println();

                System.out.println("------------------------------------------------");
                System.out.println("Same resource sending times");
                if (same_resource_sending_times[i].length == 0)
                    System.out.print("None");
                for (int j = 0; j < same_resource_sending_times[i].length; j++)
                    if (same_resource_sending_times[i].length > 0)
                        System.out.print(String.format("Resource %d: %f, ", j,
                                same_resource_sending_times[i][j].get(GRB.DoubleAttr.X)));
                System.out.println();

                System.out.println("Same resource receiving times");
                if (same_resource_receiving_times[i].length == 0)
                    System.out.print("None");
                for (int j = 0; j < same_resource_receiving_times[i].length; j++)
                    if (same_resource_receiving_times[i].length > 0)
                        System.out.print(String.format("Resource %d: %f, ", j,
                                same_resource_receiving_times[i][j].get(GRB.DoubleAttr.X)));
                System.out.println();
            }

            for (int i = 0; i < mapping_vars.length; i++)
                System.out.println(String.format("Task %d - x1:%f, x2:%f", i + 1,
                        mapping_vars[i][0].get(GRB.DoubleAttr.X),
                        mapping_vars[i][1].get(GRB.DoubleAttr.X)));

            // Dispose of model and environment
            model.dispose();
            env.dispose();

        } catch (GRBException e) {
            System.out.println("Error code: " + e.getErrorCode() + ". " + e.getMessage());
        }
    }

    public void gurobiILPExampleGroupedComm() {

        try {

            ILPFormuation ilps = new ILPFormuation();

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
            ilps.addTaskSchedulingDependencyConstraint(Tf1, Ts2, model);
            ilps.addTaskSchedulingDependencyConstraint(Tf3, Ts4, model);

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

            ILPFormuation ilps = new ILPFormuation();

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
            ilps.addTaskSchedulingDependencyConstraint(Tf1, Ts2, model);
            ilps.addTaskSchedulingDependencyConstraint(Tf3, Ts4, model);

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
