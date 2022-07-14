cd $(sudo find -type d -name OptimizationTensorflow | head -1) 
touch results_mobilenet.csv
echo "Iteration,solution_name,generations,alpha,Mu,Lambda,CrossoverRate,Cost_of_mapping" >> results_mobilenet.csv
declare -a gen_array
gen_array=(400 500)
declare -a alpha_array 
alpha_array=(50 100)
declare -a mu_array
mu_array=(25 50)
declare -a lambda_array
lambda_array=(25 50)
declare -a crossRate_array
crossRate_array=(0.95 0.9)
for gen in "${gen_array[@]}"; do echo "$gen"; done
for gen in "${gen_array[@]}"; 
do
	for alpha in "${alpha_array[@]}";
       	do
		for mu in "${mu_array[@]}";
	       	do
			for lambda in "${lambda_array[@]}";
		       	do
				for cR in "${crossRate_array[@]}";
			       	do
					echo "$gen"
					sudo gradle run -Dexec.args='src/main/resources/models_summaries/data_analysismodel_summary.csv src/main/resources/test_mobilenet_ex results_mobilenet.csv 10000 '${gen}' '${alpha}' '${mu}' '${lambda}' '${cR}

				done
			done
		done
	done
done

