/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.instance;

import static com.google.common.base.Preconditions.checkArgument;

import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import com.dsi.parallax.ml.util.MLUtils;
import com.google.common.collect.Lists;

/**
 * Collection for individual {@link BinaryClassificationInstance}'s. Provides
 * some additional utilities based on label distribution, etc.
 */
public class BinaryClassificationInstances extends
		Instances<BinaryClassificationInstance> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -4578942571464067344L;

	/** The number of positive examples in the collection */
	protected int posN = 0;

	/** The number of negative examples in the collection */
	protected int negN = 0;

	/**
	 * The threshold on a label to be positive or negative; labels >= thresh are
	 * positive
	 */
	protected double thresh = 0.5;

	/**
	 * Instantiates a new binary classification instances.
	 * 
	 * @param instances
	 *            the instances to be included in this collection
	 * @param dimensions
	 *            number of dimensionss in the instance.
	 */
	public BinaryClassificationInstances(
			List<BinaryClassificationInstance> instances, int dimensions) {
		super(dimensions);
		for (BinaryClassificationInstance inst : instances)
			addInstance(inst);
	}

	/**
	 * Instantiates a new binary classification instances.
	 * 
	 * @param dimensions
	 *            number of dimensionss in the instance.
	 * @param instances
	 *            the instances to be included in this collection
	 */
	public BinaryClassificationInstances(int dimensions,
			Collection<BinaryClassificationInstance> instances) {
		super(dimensions);
		for (BinaryClassificationInstance inst : instances)
			addInstance(inst);
	}

	/**
	 * Instantiates a new binary classification instances.
	 * 
	 * @param dimensions
	 *            number of dimensionss in the instance.
	 * @param input
	 *            the instances to be included in this collection
	 */
	public BinaryClassificationInstances(int dimensions,
			BinaryClassificationInstance... input) {
		super(dimensions);
		for (BinaryClassificationInstance inst : input)
			addInstance(inst);
	}

	public BinaryClassificationInstances(int dimensions,
			BinaryClassificationInstances... instanceses) {
		super(dimensions);
		for (BinaryClassificationInstances insts : instanceses) {
			addInstances(insts);
		}
	}

	/**
	 * Instantiates a new binary classification instances.
	 * 
	 * @param dimensions
	 *            the number of dimensionss in the instance.
	 */
	public BinaryClassificationInstances(int dimensions) {
		super(dimensions);
	}

	/**
	 * Instantiates a new binary classification instances.
	 * 
	 * @param dimensions
	 *            the number of dimensionss in the instance.
	 * @param insts
	 *            the instances to be included in this collection
	 */
	public BinaryClassificationInstances(int dimensions,
			Iterator<BinaryClassificationInstance> insts) {
		this(dimensions);
		while (insts.hasNext())
			addInstance(insts.next());
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.instance.Instances#makeIstances(int,
	 * java.util.Iterator)
	 */
	@Override
	protected Instances<BinaryClassificationInstance> makeIstances(
			int dimensions, Iterator<BinaryClassificationInstance> insts) {
		return new BinaryClassificationInstances(dimensions, insts);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.instance.Instances#makeIstances(int)
	 */
	@Override
	public Instances<BinaryClassificationInstance> makeIstances(int dimensions) {
		return new BinaryClassificationInstances(dimensions);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.instance.Instances#makeIstances(int, I[])
	 */
	@Override
	public Instances<BinaryClassificationInstance> makeIstances(int dimensions,
			BinaryClassificationInstance... input) {
		return new BinaryClassificationInstances(dimensions, input);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.instance.Instances#makeIstances(int, java.util.List)
	 */
	@Override
	public Instances<BinaryClassificationInstance> makeIstances(int dimensions,
			List<BinaryClassificationInstance> instances) {
		return new BinaryClassificationInstances(dimensions, instances);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.instance.Instances#remove(com.parallax.ml.instance.Instanze
	 * )
	 */
	@Override
	public boolean remove(BinaryClassificationInstance x) {
		boolean check = super.remove(x);
		if (check) {
			if (x.getLabel().getValue() > 0.5)
				posN = posN > 0 ? posN - 1 : 0;
			else
				negN = negN > 0 ? negN - 1 : 0;
		}
		return check;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.instance.Instances#addInstance(com.parallax.ml.instance
	 * .Instanze)
	 */
	@Override
	public void addInstance(BinaryClassificationInstance x) {
		if (x.getLabel().getValue() > 0.5)
			posN++;
		else
			negN++;
		this.instances.add(x);
	}

	public void addInstances(BinaryClassificationInstances insts) {
		for (BinaryClassificationInstance inst : insts) {
			addInstance(inst);
		}
	}

	/**
	 * Gets the num of positive examples in this collection
	 * 
	 * @return the num of positive examples in this collection
	 */
	public int getNumPos() {
		return this.posN;
	}

	/**
	 * Gets the num of negative examples in this collection
	 * 
	 * @return the num of negative examples in this collection
	 */
	public int getNumNeg() {
		return this.negN;
	}

	/**
	 * Gets the distribution of negative (first index) and positive (second
	 * index) examples in this collection. Note that that this array is a
	 * probability distribution; eg, sums to 1.
	 * 
	 * @return a size 2 double array containing [% negative, % positive]
	 */
	public double[] getClassDist() {
		double[] dist = new double[2];
		double tot = (double) this.getNumPos() + (double) this.getNumNeg();
		if (tot != 0.0) {
			dist[0] = (double) negN / tot;
			dist[1] = (double) posN / tot;
		}
		return dist;
	}

	/**
	 * Gets the distribution of negative (first index) and positive (second
	 * index) examples in this collection. Note that this array is a
	 * probability distribution; eg, sums to 1. Uses laplace (+1) smoothing to
	 * the internal counts
	 * 
	 * @return a size 2 double array containing [% negative, % positive]
	 */
	public double[] getClassDistLaplace() {
		double[] dist = new double[2];
		double tot = negN + posN + 2.0;
		if (tot != 0.0) {
			dist[0] = ((double) negN + 1.0) / tot;
			dist[1] = ((double) posN + 1.0) / tot;
		}
		return dist;
	}

	/**
	 * Returns the positive instances in the collection
	 * 
	 * @return all positive instances in the collection
	 */
	public BinaryClassificationInstances positiveInstances() {
		BinaryClassificationInstances out = new BinaryClassificationInstances(
				dimensions);
		for (BinaryClassificationInstance x : this)
			if (x.getLabel().getValue() > thresh)
				out.addInstance(x);
		return out;
	}

	/**
	 * Returns negative instances in the collection
	 * 
	 * @return all negative instances in the collection
	 */
	public BinaryClassificationInstances negativeInstances() {
		BinaryClassificationInstances out = new BinaryClassificationInstances(
				dimensions);
		for (BinaryClassificationInstance x : this)
			if (x.getLabel().getValue() <= thresh)
				out.addInstance(x);
		return out;
	}

	/**
	 * Induce skew; generate a {@link BinaryClassificationInstances} with the
	 * specified label skew of positive to negative instances. This data set is
	 * generated by sampling from the positive and negative instances in this
	 * collection
	 * 
	 * @param ratio
	 *            the desired % of instances that are positive
	 * @return a new collection generated by sampling from the instances in this
	 *         collection with the specified class distribution
	 */
	public BinaryClassificationInstances induceSkew(double ratio) {
		checkArgument(ratio <= 1 && ratio >= 0,
				"input value (%f) outside options bounds [0, 1]", ratio);
		if (ratio < 0 || ratio > 1)
			throw new IllegalArgumentException(
					"ratio must be a probability. input: " + ratio);

		BinaryClassificationInstances out = new BinaryClassificationInstances(
				dimensions);
		double thisRatio = getPositiveRatio();

		if (thisRatio == ratio)
			out = this;
		else if (thisRatio < ratio) {
			// down sample the majority instances
			double p = (1. - ratio) * posN / (negN * ratio);
			Random rand = new Random(System.currentTimeMillis());
			for (BinaryClassificationInstance x : this)
				if (x.getLabel().getValue() <= 0.5) {
					if (rand.nextDouble() < p) {
						out.addInstance(x);
					}
				} else {
					out.addInstance(x);
				}
		} else {
			// down sample the minority instances
			double p = ratio == 1 ? 1 : ratio * negN / (posN * (1. - ratio));
			for (BinaryClassificationInstance x : this)
				if (x.getLabel().getValue() > 0.5) {
					if (MLUtils.GENERATOR.nextDouble() < p) {
						out.addInstance(x);
					}
				} else {
					out.addInstance(x);
				}
		}
		// make sure there are at least 1 pos and neg instance in out
		for (BinaryClassificationInstance x : this) {
			if (out.negN == 0 && x.getLabel().getValue() <= 0.5)
				out.addInstance(x);
			if (out.posN == 0 && x.getLabel().getValue() > 0.5)
				out.addInstance(x);
		}
		return out;
	}

	/**
	 * get the % of instances that have a positive label
	 * 
	 * @return
	 */
	public double getPositiveRatio() {
		return (posN + negN == 0 ? 0 : (double) (posN) / (double) (posN + negN));
	}

	/**
	 * Merge several collections of instances into a new
	 * {@link BinaryClassificationInstances}.
	 * 
	 * @param dimensions
	 *            number of dimensionss in the instance.
	 * @param input
	 *            collections of instances to be merged
	 * @return a merged collection of instances containing all the specified
	 *         input
	 */
	public static BinaryClassificationInstances mergeInstances(int dimensions,
			BinaryClassificationInstances... input) {
		BinaryClassificationInstances out = new BinaryClassificationInstances(
				dimensions);

		for (BinaryClassificationInstances inst : input) {
			out.addInstance(inst.getInstances());
		}
		return out;
	}

	public BinaryClassificationInstances getStratifiedTraining(int fold,
			int folds) {
		return new BinaryClassificationInstances(dimensions,
				(BinaryClassificationInstances) positiveInstances()
						.getTraining(fold, folds),
				(BinaryClassificationInstances) negativeInstances()
						.getTraining(fold, folds));
	}

	public BinaryClassificationInstances getStratifiedTesting(int fold,
			int folds) {
		return new BinaryClassificationInstances(dimensions,
				(BinaryClassificationInstances) positiveInstances().getTesting(
						fold, folds),
				(BinaryClassificationInstances) negativeInstances().getTesting(
						fold, folds));
	}

	@Override
	public void clear() {
		negN = 0;
		posN = 0;
		instances = Lists.newArrayList();
	}

}
