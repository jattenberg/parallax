/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline;

import com.dsi.parallax.ml.util.ScaledNormalizing;
import com.dsi.parallax.ml.vector.LinearVector;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;

/**
 * VectorNormalizingPipe scales each element of the input vector by dividing it with the 
 * norm of the vector (whose type L0/L1/L2/L-infinity) is specified.
 *
 * @author Josh Attenberg
 */
public class VectorNormalizingPipe extends AbstractPipe<LinearVector, LinearVector> {

	private static final long serialVersionUID = -5337253185016628355L;
	private final ScaledNormalizing normalizing;

    /**
     * Class constructor specifying Scaled Normalizing to create
     *
     * @param normalizing  Scaled Normalizing
     */
	public VectorNormalizingPipe(ScaledNormalizing normalizing) {
		this.normalizing = normalizing;
	}

    /**
     * The method returns the class's Type "VectorNormalizingPipe"
     * @return Type
     */
	@Override
	public Type getType() {
		return new TypeToken<VectorNormalizingPipe>() {
		}.getType();
	}

	@Override
	protected Context<LinearVector> operate(Context<LinearVector> context) {
		LinearVector in = context.getData();
		LinearVector out = normalizing.scaledNormalizing(in);
		
		return Context.createContext(context, out);
	}
}
