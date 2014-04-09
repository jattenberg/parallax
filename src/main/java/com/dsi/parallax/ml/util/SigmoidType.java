/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.util;

public enum SigmoidType
{
	TRUNCATION{@Override public double sigmoid(double input){return input>1?1:input<0?0:input;}},
	LOGIT{@Override public double sigmoid(double input){return 1. / (1. + Math.exp(-input));}},
	TANH{@Override public double sigmoid(double input){return Math.tanh(input);}};
	
	public abstract double sigmoid(double input);
}

