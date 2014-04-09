/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.objective;

import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.model.Model;
import com.dsi.parallax.ml.target.Target;

// TODO: Auto-generated Javadoc
/**
 * function object measuring the utility of a
 * particular model on a particular dataset
 * 
 * often used in conjunction with crossvalidation.
 *
 * @param <T> the generic type
 * @author jattenberg
 */
public interface Objective<T extends Target> {
	
	/**
	 * returns a double representing the performance of the current model
	 * on the supplied instances
	 * 
	 * generally, higher is better.
	 *
	 * @param <I> the generic type
	 * @param <E> the element type
	 * @param <M> the generic type
	 * @param instances the instances
	 * @param model the model
	 * @return the double
	 */
	public <I extends Instance<T>, E extends Iterable<I>, M extends Model<T, M>> double evaluate(E instances, M model);
}
