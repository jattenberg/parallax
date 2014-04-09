package com.dsi.parallax.pipeline.text;

import java.lang.reflect.Type;

import org.apache.commons.lang.StringUtils;

import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.LinearVectorFactory;
import com.dsi.parallax.pipeline.AbstractPipe;
import com.dsi.parallax.pipeline.Context;
import com.google.gson.reflect.TypeToken;

/**
 * Pipe component for transforming strings containing examples compressed in the
 * SVMLight sparse format into a LinearVector with an appropriately parsed
 * label.
 */
public class SVMLightToLabeledVectorPipe extends
		AbstractPipe<String, LinearVector> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -6077484722656220248L;

	/** The number of dimensions in the problem space */
	private final int dimensions;

	/**
	 * Instantiates a new SVMLight to labeled vector pipe.
	 * 
	 * @param dimensions
	 *            number of dimensions in the problem space, required input in
	 *            order to maintain dimensional consistency throughout the
	 *            machine learning problem.
	 */
	public SVMLightToLabeledVectorPipe(int dimensions) {
		super();
		this.dimensions = dimensions;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.pipeline.Pipe#getType()
	 */
	@Override
	public Type getType() {
		return new TypeToken<SVMLightToLabeledVectorPipe>() {
		}.getType(); //used for serialization
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.pipeline.AbstractPipe#operate(com.parallax.pipeline.Context)
	 */
	@Override
	protected Context<LinearVector> operate(Context<String> context) {
		String rawInput = context.getData(); //extract the raw string SVMLight example
		int toIndex = rawInput.indexOf("#"); //identify if there is a comment, and the index into the string if so

		//grabs the part of the string before the comment, trims (removing any leading or training white space)
		//and splits on spaces
		String[] parts = StringUtils.split(rawInput.substring(0,
				toIndex > 0 ? toIndex : rawInput.length()).trim());

		//in SVMLight, the first entry is always the example's label
		String label = parts[0];
		//initialize the data structure to store the numerical info for future data processing
		LinearVector vect = LinearVectorFactory.getVector(dimensions);

		//iterate through the feature / value pairs in the SVMLight string format, 
		//set the LinearVector's feature values appropriately
		for (int i = 1; i < parts.length; i++) {
			String[] featureValue = parts[i].split(":");
			vect.resetValue(Integer.parseInt(featureValue[0]),
					Double.parseDouble(featureValue[1]));
		}

		//set the context used to pass data through the pipeline appropriately with 
		//the vector as payload. 
		Context<LinearVector> out = Context.createContext(context, vect);
		//set the label of the output context
		out.setLabel(label);
		return out;

	}

}
