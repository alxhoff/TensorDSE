package net.sf.opendse.TensorDSE;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonIOException;
import com.google.gson.stream.JsonReader;
import net.sf.opendse.TensorDSE.JSON.Architecture.ArchitectureJSON;
import net.sf.opendse.TensorDSE.JSON.Model.Layer;
import net.sf.opendse.TensorDSE.JSON.Model.Model;
import net.sf.opendse.TensorDSE.JSON.Model.ModelJSON;
import net.sf.opendse.io.SpecificationWriter;
import net.sf.opendse.model.Application;
import net.sf.opendse.model.Architecture;
import net.sf.opendse.model.Communication;
import net.sf.opendse.model.Dependency;
import net.sf.opendse.model.Link;
import net.sf.opendse.model.Mapping;
import net.sf.opendse.model.Mappings;
import net.sf.opendse.model.Resource;
import net.sf.opendse.model.Specification;
import net.sf.opendse.model.Task;
// import net.sf.opendse.visualization.SpecificationViewer;

/**
 * The {@code SpecificationDefinition} is the class defining the Specification
 * that corresponds to
 * the graph of the deep learning model to which we want to optimize the
 * performance.
 *
 * @author Ines Ben Hmida
 * @author Alex Hoffman
 *
 * @param modelsummary user entry of the file path a csv file that summarized
 *                     the graph of the
 *                     considered deep learning model
 * @param costsfile    A csv file that contains the costs deducted from running
 *                     the benchmarks
 *
 */
public class SpecificationDefinition {

	/**
	 *
	 */
	public final List<String> supported_layers = Arrays.asList("conv_2d", "depthwise_conv_2d", "max_pool_2d", "average_pool_2d", "reshape", "fully_connected",
			"softmax", "add");

	/**
	 *
	 */
	private Specification specification = null;

	/**
	 *
	 */
	private HashMap<String, OperationCosts> operation_costs = new HashMap<String, OperationCosts>();

	private String profiling_costs_directory_path = null;

	/**
	 * Top level HashMap takes the model's index as the key, then the target layer's
	 * index is used
	 * to access the layer's Task
	 */
	private HashMap<Integer, HashMap<Integer, Task>> application_graphs = new HashMap<Integer, HashMap<Integer, Task>>();

	/**
	 *
	 */
	private HashMap<String, List<Resource>> resources = new HashMap<String, List<Resource>>();

	/**
	 *
	 */
	private ArrayList<Task> starting_tasks = new ArrayList<Task>();

	public List<Model> json_models = null;

	/**
	 * @param model_summary_path
	 * @param profiling_costs_directory_path
	 * @throws Exception
	 */
	public SpecificationDefinition(String model_summary_path, String profiling_costs_directory_path,
			String architecture_summary_path) throws Exception {

		Gson gson = new Gson();
		ModelJSON model_json = null;
		JsonReader jr;

		try {
			jr = new JsonReader(new FileReader(model_summary_path));
			model_json = gson.fromJson(jr, ModelJSON.class);
			this.json_models = model_json.getModels();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		for (Model model : this.json_models) {
			this.operation_costs.put(model.getName(),
					new OperationCosts(profiling_costs_directory_path, model.getName()));
		}
		this.profiling_costs_directory_path = profiling_costs_directory_path;
		this.specification = GetSpecificationFromTFLiteModel(model_summary_path, architecture_summary_path);
	}

	public void WriteJSONModelsToFile(String filename) {

		if (this.json_models != null) {

			Gson gson = new GsonBuilder().setPrettyPrinting().create();
			try {
				System.out.println(String.format("Storing mapping results: %s", filename));
				FileWriter file = new FileWriter(filename);
				ModelJSON model_json = new ModelJSON();
				model_json.setModels(this.json_models);
				gson.toJson(model_json, file);
				file.close();
			} catch (JsonIOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

	/**
	 * @brief Creates the OpenDSE specification which is the combination of the
	 *        architecture and
	 *        application graph with the set of possible mappings. The architecture
	 *        and application
	 *        graphs are generated from the provided
	 * @param models_description_path   Path to the JSON file containing a summary
	 *                                  of the NN models
	 *                                  from which the application graph is to be
	 *                                  built
	 * @param hardware_description_path Path to the JSON file containing the number
	 *                                  of execution
	 *                                  units on the target hardware platform
	 * @return Specification
	 */
	public Specification GetSpecificationFromTFLiteModel(String models_description_path,
			String hardware_description_path) {

		Architecture<Resource, Link> architecture = GetArchitectureFromArchitectureSummary(hardware_description_path);
		Application<Task, Dependency> application = GetApplicationFromModelSummary();
		Mappings<Task, Resource> mappings = CreateMappingOptions();

		Specification specification = new Specification(application, architecture, mappings);

		// For debugging
		// SpecificationViewer.view(specification);

		SpecificationWriter writer = new SpecificationWriter();
		writer.write(specification, "src/main/resources/generated_specs/spec_"
				+ new SimpleDateFormat("yyyy-MM--dd_hh-mm-ss").format(new Date()) + ".xml");

		return specification;
	}

	/**
	 * @param task_name
	 * @param inputs
	 * @param output_shape
	 * @param task_type
	 * @return
	 */
	private static Task CreateTaskNode(Layer layer, Integer model_index) {
		Task ret = new Task(String.format("%s-index%d_model%d", layer.getType(), layer.getIndex(),
				model_index));
		// TODO Input size?
		ret.setAttribute("type", layer.getType());
		// TODO 0 tensor for dtype?
		ret.setAttribute("dtype", layer.getInputs().get(0).getType());
		ret.setAttribute("input_tensors", layer.getInputTensorString());
		ret.setAttribute("output_tensors", layer.getOutputTensorString());
		ret.setAttribute("cost", 0.0);
		ret.setAttribute("start_time", 0.0);
		ret.setAttribute("end_time", 0.0);

		return ret;
	}

	/**
	 * @param model_summary_path
	 * @return
	 */
	private Application<Task, Dependency> GetApplicationFromModelSummary() {

		Application<Task, Dependency> application = new Application<Task, Dependency>();

		if (this.json_models != null)
			// For each model in the application graph
			for (int k = 0; k < this.json_models.size(); k++) {

				// Python structure of model JSON
				Model model = this.json_models.get(k);
				List<Layer> layers = model.getLayers();

				// Hashmap for each graph in application graph, containing a
				// hashmap of each layer in specific graph
				application_graphs.put(k, new HashMap<Integer, Task>());

				// Create task nodes in Application and populate hashmap
				for (int i = 0; i < layers.size(); i++) {
					Task t = CreateTaskNode(layers.get(i), k);

					// We want to keep track of our entry point tasks to our models such
					// that they can be traveresed more easily
					if (i == 0)
						this.starting_tasks.add(t);

					application_graphs.get(k).put(i, t);
					application.addVertex(t);
				}

				// Create connections between verticies
				for (int i = 0; i < layers.size(); i++) {
					// JSON parsed layer object
					Layer sending_layer = layers.get(i);
					Task sending_layer_task = application_graphs.get(k).get(sending_layer.getIndex());

					// Step through all output tensors
					List<Integer> sending_layer_output_tensor_indices = sending_layer.getOutputTensorArray();

					// Task to create communication from, ie. sending layer
					for (Integer sending_layer_output_tensor_index : sending_layer_output_tensor_indices) {

						// Output tensor doesn't send to anything
						if (sending_layer_output_tensor_index != model.getFinishing_tensor()) {

							// Get OpenDSE task and set input shape
							Integer target_task_index = 0;
							try {
								target_task_index = model
										.getLayerWithInputTensor(sending_layer_output_tensor_index)
										.getIndex();
							} catch (Exception ex) {
								ex.printStackTrace();
								System.exit(1);
							}

							Task target_task = application_graphs.get(k).get(target_task_index);
							target_task.setAttribute("input_shape", sending_layer.getOutputs()
									.get(sending_layer.getOutputs().size() - 1).getShapeProduct());

							// Comm task between two layers
							Communication comm = new Communication(String.format("comm_%s_TO_%s",
									sending_layer_task.getId(), target_task.getId()));
							application.addVertex(comm);

							// Create dependencies from:
							// - sending task -> communication
							// - communication -> target task
							application
									.addEdge(
											new Dependency(String.format("%s[%s] -> comm_%s",
													sending_layer_task.getId(),
													sending_layer.getIndex(), comm.getId())),
											sending_layer_task, comm);
							application.addEdge(
									new Dependency(String.format("comm_%s -> %s[%d]", comm.getId(),
											target_task.getId(), target_task_index)),
									comm, target_task);
						}
					}
				}
			}

		return application;
	}

	/**
	 * @param architecture_summary_path
	 * @return
	 */
	private Architecture<Resource, Link> GetArchitectureFromArchitectureSummary(
			String architecture_summary_path) {

		Architecture<Resource, Link> architecture = new Architecture<Resource, Link>();

		Gson gson = new Gson();
		ArchitectureJSON architecture_json = null;

		try {
			JsonReader jr = new JsonReader(new FileReader(architecture_summary_path));
			architecture_json = gson.fromJson(jr, ArchitectureJSON.class);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return null;
		}

		// System only has 2 busses independant of the number of execution units
		Resource bus_pci = new Resource("pci");
		Resource bus_usb = new Resource("usb");

		// Create a full-duplex link between PCI and USB busses
		Link link_pci_usb = new Link("link_pci_usb");
		link_pci_usb.setAttribute("cost", 0.0);
		architecture.addEdge(link_pci_usb, bus_pci, bus_usb);
		Link link_usb_pci = new Link("link_usb_pci");
		link_usb_pci.setAttribute("cost", 0.0);
		architecture.addEdge(link_usb_pci, bus_usb, bus_pci);

		// Create CPU Cores
		if (architecture_json.getCPU_count() > 0) {
			List<Resource> cpu_cores = new ArrayList<Resource>();
			for (int i = 0; i < architecture_json.getCPU_count(); i++) {
				String r_id = String.format("cpu%d", i);
				Resource core = new Resource(r_id);
				architecture.addVertex(core);
				cpu_cores.add(core);
			}
			resources.put("cpu", cpu_cores);

			// Link to bus
			cpu_cores.forEach((core) -> {
				Link link_to_dev = new Link(String.format("link_pci_%s", core.getId()));
				link_to_dev.setAttribute("cost", 0.0);
				architecture.addEdge(link_to_dev, bus_pci, core);
				Link link_to_pci = new Link(String.format("link_%s_pci", core.getId()));
				link_to_pci.setAttribute("cost", 0.0);
				architecture.addEdge(link_to_pci, core, bus_pci);
			});
		}

		// Create GPUs
		if (architecture_json.getGPU_count() > 0) {
			List<Resource> gpus = new ArrayList<Resource>();
			for (int i = 0; i < architecture_json.getGPU_count(); i++) {
				String r_id = String.format("gpu%d", i);
				Resource gpu = new Resource(r_id);
				architecture.addVertex(gpu);
				gpus.add(gpu);
			}
			resources.put("gpu", gpus);

			gpus.forEach((gpu) -> {
				Link link_to_dev = new Link(String.format("link_pci_%s", gpu.getId()));
				link_to_dev.setAttribute("cost", 0.0);
				architecture.addEdge(link_to_dev, bus_pci, gpu);
				Link link_to_pci = new Link(String.format("link_%s_pci", gpu.getId()));
				link_to_pci.setAttribute("cost", 0.0);
				architecture.addEdge(link_to_pci, gpu, bus_pci);
			});
		}

		// Create TPUs
		if (architecture_json.getTPU_count() > 0) {
			List<Resource> tpus = new ArrayList<Resource>();
			for (int i = 0; i < architecture_json.getTPU_count(); i++) {
				String r_id = String.format("tpu%d", i);
				Resource tpu = new Resource(r_id);
				architecture.addVertex(tpu);
				tpus.add(tpu);
			}
			resources.put("tpu", tpus);

			tpus.forEach((tpu) -> {
				Link link_to_dev = new Link(String.format("link_usb_%s", tpu.getId()));
				link_to_dev.setAttribute("cost", 0.0);
				architecture.addEdge(link_to_dev, bus_usb, tpu);
				Link link_to_usb = new Link(String.format("link_%s_usb", tpu.getId()));
				link_to_usb.setAttribute("cost", 0.0);
				architecture.addEdge(link_to_usb, tpu, bus_usb);
			});
		}

		return architecture;
	}

	/**
	 * @brief Creates a set of possible mappings, ie. all tasks can be mapped onto
	 *        the CPU + GPU,
	 *        compatible tasks can be mapped to the TPU.
	 * @return Mappings<Task, Resource>
	 */
	private Mappings<Task, Resource> CreateMappingOptions() {

		Mappings<Task, Resource> mappings = new Mappings<Task, Resource>();

		Integer model_index = 0;

		for (HashMap<Integer, Task> map : application_graphs.values()) {

			for (Task task : map.values()) {
				String task_id = task.getId();

				// All tasks can be mapped to CPU + GPU
				if (resources.containsKey("cpu")) {
					List<Resource> cpus = resources.get("cpu");
					for (int i = 0; i < cpus.size(); i++) {
						Resource resource = cpus.get(i);
						Mapping<Task, Resource> m = new Mapping<Task, Resource>(
								String.format("%s:%s", task_id, resource.getId()), task, resource);
						m.setAttribute("cost",
								this.operation_costs.get(this.json_models.get(model_index).getName()).GetOpCost("cpu",
										task.getAttribute("type"), task.getAttribute("dtype")));
						mappings.add(m);
					}
				}

				if (resources.containsKey("gpu")) {
					List<Resource> gpus = resources.get("gpu");
					for (int i = 0; i < gpus.size(); i++) {
						Resource resource = gpus.get(i);
						Mapping<Task, Resource> m = new Mapping<Task, Resource>(
								String.format("%s:%s", task_id, resource.getId()), task, resource);
						m.setAttribute("cost",
								this.operation_costs.get(this.json_models.get(model_index).getName()).GetOpCost("gpu",
										task.getAttribute("type"), task.getAttribute("dtype")));
						mappings.add(m);
					}
				}

				// Compatible tasks can be mapped to TPU
				if (resources.containsKey("tpu")) {
					List<Resource> tpus = resources.get("tpu");
					if (supported_layers.contains(task.getAttribute("type")))
						for (int i = 0; i < tpus.size(); i++) {
							Resource resource = tpus.get(i);
							Mapping<Task, Resource> m = new Mapping<Task, Resource>(
									String.format("%s:%s", task_id, resource.getId()), task,
									resource);
							m.setAttribute("cost",
									this.operation_costs.get(this.json_models.get(model_index).getName()).GetOpCost(
											"tpu",
											task.getAttribute("type"), task.getAttribute("dtype")));
							mappings.add(m);
						}
				}
			}

			model_index += 1;
		}

		return mappings;
	}

	/**
	 * @return Specification
	 */
	public Specification getSpecification() {
		return (specification);
	}

	public void setSpecification(Specification specification) {
		this.specification = specification;
	}

	public HashMap<String, OperationCosts> getOperation_costs() {
		return operation_costs;
	}

	public void setOperation_costs(HashMap<String, OperationCosts> operation_costs) {
		this.operation_costs = operation_costs;
	}

	public HashMap<Integer, HashMap<Integer, Task>> getApplication_graphs() {
		return application_graphs;
	}

	public void setApplication_graphs(HashMap<Integer, HashMap<Integer, Task>> application_graphs) {
		this.application_graphs = application_graphs;
	}

	public HashMap<String, List<Resource>> getResources() {
		return resources;
	}

	public void setResources(HashMap<String, List<Resource>> resources) {
		this.resources = resources;
	}

	public List<String> getSupported_layers() {
		return supported_layers;
	}

	public ArrayList<Task> getStarting_tasks() {
		return starting_tasks;
	}

	public void setStarting_tasks(ArrayList<Task> starting_tasks) {
		this.starting_tasks = starting_tasks;
	}

	public List<Model> getJson_models() {
		return json_models;
	}

	public void setJson_models(List<Model> json_models) {
		this.json_models = json_models;
	}

	public List<Double> getDeadlines() {

        List<Double> ret = new ArrayList<Double>();

        for(Model m: this.json_models)
            ret.add(m.getDeadline());

        return ret;
    }

}
