/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.projection;

import java.io.Serializable;

import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.vector.LinearVector;

// TODO: Auto-generated Javadoc
/**
 * The Interface Projection.
 */
public interface Projection extends Serializable
{
	
	/**
	 * Project.
	 *
	 * @param x the x
	 * @return the linear vector
	 */
	public LinearVector project(LinearVector x);
	
	/**
	 * Project.
	 *
	 * @param <I> the generic type
	 * @param x the x
	 * @return the i
	 */
	public <I extends Instance<?>> I project(I x);
	
	/**
	 * Gets the input dimension.
	 *
	 * @return the input dimension
	 */
	public int getInputDimension();
	
	/**
	 * Gets the output dimension.
	 *
	 * @return the output dimension
	 */
	public int getOutputDimension();
}
