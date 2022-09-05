package net.sf.opendse.visualization;

import java.awt.Color;
import java.util.HashMap;
import java.util.Map;

import net.sf.opendse.model.ICommunication;
import net.sf.opendse.model.Node;
import net.sf.opendse.visualization.GraphPanelFormatApplication.FunctionTask;

import com.google.inject.Singleton;

@Singleton
public class ColorModelApplication implements ColorModel {

	Map<String, Color> colors = new HashMap<String, Color>();

	@Override
	public Color get(Node node) {
		if (node instanceof FunctionTask) {
			return Graphics.GRAY;
		} else if (node instanceof ICommunication) {
			return Graphics.KHAKI;
		} else {
			return Graphics.TOMATO;
		}
	}

	/**
	 * Register a {@link Color} for a {@link Node} type.
	 * 
	 * @param type
	 *            the type of the node
	 * @param color
	 *            the color to paint the node
	 * 
	 * @see Node#getType()
	 */
	public void registerColor(String type, Color color) {
		colors.put(type, color);
	}
}
