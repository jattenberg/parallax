package com.dsi.parallax.pipeline.text;

import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.LinearVectorFactory;
import com.dsi.parallax.pipeline.AbstractPipe;
import com.dsi.parallax.pipeline.Context;
import com.google.gson.reflect.TypeToken;
import org.apache.commons.lang.StringUtils;

import java.lang.reflect.Type;

/**
 * Pipe to convert a string containing a labeled sample in the Numeric VW format 
 * into a LinearVector with an appropriately parsed label.
 * 
 * @author Sumit Chopra
 */
public class NumericVWToLabeledVectorPipe extends
		AbstractPipe<String, LinearVector> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 6766098968708435991L;
	
	/** The maximum number of dimensions in the problem space */
	private final int dimensions;

	/**
	 * Instantiates a new NumericVW to labeled vector pipe.
	 * 
	 * @param dimensions
	 *            maximum number of dimensions in the problem space. This is required 
	 *            in order to maintain dimensional consistency throughout the
	 *            machine learning problem.
	 */
	public NumericVWToLabeledVectorPipe(int dimensions) {
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
		return new TypeToken<NumericVWToLabeledVectorPipe>() {
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
		// extract the raw contents of the string
		String rawInput = context.getData();
		// get the index of the '|' separator
		int toIndex = rawInput.indexOf("|");

		// grab the part of the string before the separator (i.e., label part)
		String[] labelParts = StringUtils.split(rawInput.substring(0, toIndex).trim());
		
		// grab the part of the string after the separator (i.e., data part)
		String[] dataParts = StringUtils.split(rawInput.substring(toIndex + 1, rawInput.length()).trim());

		// extract the label (which should be the only entry in labelParts)
		String label = labelParts[0];

		// initialize the data structure to store the numerical values of the vector
		LinearVector vect = LinearVectorFactory.getVector(dimensions);

		// iterate over the feature:value pairs in the NumericVW string format, and 
		// store the result in the LinearVector
		for (int i = 0; i < dataParts.length; i++) {
			String[] featureValue = dataParts[i].split(":");
			vect.resetValue(Integer.parseInt(featureValue[0]),
					Double.parseDouble(featureValue[1]));
		}

		// create a context with the newly created LinearVector as the payload, and the 
		// corresponding label
		Context<LinearVector> out = Context.createContext(context, vect);
		//set the label of the output context
		out.setLabel(label);
		return out;

	}

}
