/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.instance;

/**
 * Bag in the statistical sense. From a group of instances, a sub-group are
 * sampled with replacement. This class also wraps the remaining, unselected
 * instances in order to aid with out-of-bag estimates.
 * 
 * @param <I>
 *            the type of instance used.
 */
public class Bag<I extends Instance<?>> {

	/**
	 * The in-bag instances to be considered; those instances selected with
	 * replacement
	 */
	private final Instances<I> bag;
	/**
	 * The out-of-bag instances; those instances from the initial set that were
	 * not selected during samping
	 */
	private final Instances<I> out;

	/**
	 * Instantiates a new bag.
	 * 
	 * @param bag
	 *            The in-bag instances to be considered; those instances
	 *            selected with replacement
	 * @param out
	 *            The out-of-bag instances; those instances from the initial set
	 *            that were not selected during samping
	 */
	public Bag(Instances<I> bag, Instances<I> out) {
		this.bag = bag;
		this.out = out;
	}

	/**
	 * Adds Instances to the first (in bag) split
	 * 
	 * @param inst
	 *            the instance to be added to the first split.
	 */
	public void addInBag(I inst) {
		bag.addInstance(inst);
	}

	/**
	 * Adds instances to the second (out-of-bag) split
	 * 
	 * @param inst
	 *            the instance to be added to the second split
	 */
	public void addOutOfBag(I inst) {
		out.addInstance(inst);
	}

	/**
	 * Gets the first (in bag) split.
	 * 
	 * @return the bag of instances (the first split)
	 */
	public Instances<I> getBagInstances() {
		return bag;
	}

	/**
	 * Gets the second (out-of-bag) split.
	 * 
	 * @return the out-of-bag instances (second split) 
	 */
	public Instances<I> getOutOfBagInstances() {
		return out;
	}
}
