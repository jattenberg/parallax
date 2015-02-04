/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline;

import com.dsi.parallax.ml.vector.LinearVector;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;

/**
 * LinearVectorLengthFilterPipe
 *
 * @author Josh Attenberg
 */
public class LinearVectorLengthFilterPipe extends
		AbstractFilteringPipe<LinearVector> {

	private static final long serialVersionUID = -8330381548120123587L;
	private int length = 0;

    /**
     * Class constructor specifying number of length to create
     * @param length number of length
     */
	public LinearVectorLengthFilterPipe(int length) {
		super();
		this.length = length;
	}

    /**
     * The method returns the class's Type "LinearVectorLengthFilterPipe"
     * @return Type
     */
	@Override
	public Type getType() {
		return new TypeToken<LinearVectorLengthFilterPipe>(){}.getType();
	}

	@Override
	protected boolean operate(Context<LinearVector> context) {
		return Math.round(context.getData().L1Norm()) > length;
	}
}
