/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.optimization;

import com.dsi.parallax.ml.vector.LinearVector;

public class Gradient extends WeightedGradient {

	/**
	 * @param weight
	 * @param gradientVector
	 * @param loss
	 */
	public Gradient(LinearVector gradientVector, double loss) {
		super(1., gradientVector, loss);
	}

	/**
	 * @param weight
	 * @param gradientVector
	 */
	public Gradient(LinearVector gradientVector) {
		super(1., gradientVector, 0);
	}

	@Override
	public String toString() {
		return "loss: " + this.getLoss() + " grad: " + this.getGradientVector();
	}

    @Override
    public Gradient copy() {
        return new Gradient(getGradientVector().copy());
    }
}
