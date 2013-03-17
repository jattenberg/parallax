/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.classifier.linear;

import java.util.List;

import com.parallax.ml.classifier.Classifier;
import com.parallax.ml.vector.LinearVector;

/**
 * Interface describing linear classifiers, those classifiers which make
 * predictions according to f(x) = s(w'x) where s is some activation function
 * eg, a sigmoid
 * 
 * @param <C>
 *            the generic type
 */
public interface LinearClassifier<C extends LinearClassifier<C>> extends
		Classifier<C> {

	/**
	 * Inits the parameter vector
	 * 
	 * @param initVal
	 *            the initial value of the parameter vector
	 */
	public void initW(double initVal);

	/**
	 * Inits the parameter vector
	 */
	public void initW();

	/**
	 * Gets the parameter vector as an array of doubles
	 * 
	 * @return the parameter vector as an array of doubles
	 */
	public double[] getW();

	/**
	 * Gets the parameter at a particular index
	 * 
	 * @param index
	 *            dimension of the requested parameter value
	 * @return the param value requested
	 */
	public double getParam(int index);

	/**
	 * Sets the parameters to the input parameter values
	 * 
	 * @param W
	 *            the new w
	 */
	public void setW(List<Double> W);

	/**
	 * Sets the parameter values to the input parameter values
	 * 
	 * @param W
	 *            the new w
	 */
	public void setW(double[] W);

	/**
	 * Update the value of the specified parameter by adding the given value
	 * @param index
	 *            the index of the parameter to be updated
	 * @param wi
	 *            the wi value to be added to the specified parameter
	 */
	public void updateParam(int index, double wi);

	/**
	 * Sets the value of the specified parameter
	 * 
	 * @param wi
	 *            the new value of the specified parameter
	 * @param index
	 *            the index of the parameter to be set
	 */
	public void setParam(double wi, int index);

	/**
	 * Gets the parameter vector as a {@link LinearVector}
	 * 
	 * @return the parameter vector as a {@link LinearVector}
	 */
	public LinearVector getVector();
	
	/**
	 * prints the non-zero params of the internal {@link LinearVector}
	 * @return a string representing the internal parameters
	 */
	public String prettyPrint();
}
