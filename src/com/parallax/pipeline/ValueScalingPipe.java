/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline;

import java.lang.reflect.Type;
import java.util.Set;

import com.google.gson.reflect.TypeToken;
import com.parallax.ml.vector.LinearVector;
import com.parallax.ml.vector.LinearVectorFactory;
import com.parallax.ml.vector.util.ValueScaling;

/**
 * ValueScalingPipe scales the value of all the elements of the input LinearVector. 
 * The type of scaling is specified at the time of creating an instance of this 
 * pipe. 
 *
 * @author Josh Attenberg
 */
public class ValueScalingPipe extends AbstractPipe<LinearVector, LinearVector> {

	private static final long serialVersionUID = -1786610491732670544L;
	private ValueScaling scaling = ValueScaling.UNSCALED;
	private Set<Integer> toScale;

    /**
     * Class constructor specifying Value Scaling to create
     * @param scaling Value Scaling
     */
	public ValueScalingPipe(ValueScaling scaling) {
		super();
		this.scaling = scaling;
	}

    /**
     * Class constructor specifying Value Scaling and Scale set to create
     * @param scaling Value Scaling
     * @param toScale scale set
     */
	public ValueScalingPipe(ValueScaling scaling, Set<Integer> toScale) {
		this(scaling);
		this.toScale = toScale;
	}

    /**
     * The method returns the class's Type "ValueScalingPipe"
     * @return Type
     */
	@Override
	public Type getType() {
		return new TypeToken<ValueScalingPipe>() {
		}.getType();
	}

	@Override
	protected Context<LinearVector> operate(Context<LinearVector> context) {
		// get the input linear vector
		LinearVector in = context.getData();
		// make a new output linear vector
		LinearVector out = LinearVectorFactory.getVector(in.size());
		// scale all the values by the scaling type
		for (int x_i : in) {
			if (null == toScale || toScale.contains(x_i)) {
				double y_i = in.getValue(x_i);
				out.updateValue(x_i, scaling.scaleValue(y_i));
			}
		}
		return Context.createContext(context, out);
	}
}
