/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.classifier.kernelmethods;

import com.dsi.parallax.ml.classifier.ClassifierEvaluation;
import com.dsi.parallax.ml.classifier.kernelmethods.KernelMethodBuilder.ForgetronBuilder;
import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.instance.Instances;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.util.MLUtils;
import com.dsi.parallax.ml.util.pair.PrimitivePair;

import java.util.LinkedHashMap;
import java.util.Map;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Forgetron model for kernel classifiers with a fixed number of support
 * vectors. See
 * {@link <a href="http://books.nips.cc/papers/files/nips18/NIPS2005_0192.pdf">The Forgetron: A Kernel-Based Perceptron on a Fixed Budget</a>}
 * by Ofer Dekel, Shai Shalev-Shwartz and Yoram Singer
 * 
 * @author jattenberg
 */
public class Forgetron extends AbstractUpdateableKernelClassifier<Forgetron> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -295243405175989314L;

	/**
	 * A mapping from instances to their respective lagrange multiplier and
	 * multiplier sign
	 */
	private Map<Instance<?>, PrimitivePair> instanceToLagrangeAndSign;

	/** The budget for support vectors */
	private double budget = 1000;

	/**
	 * Instantiates a new forgetron.
	 * 
	 * @param dimension
	 *            the number of features in the instantiated classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 */
	public Forgetron(int dimension, boolean bias) {
		super(dimension, bias);
		initialize();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.UpdateableClassifier#update(com.parallax.ml
	 * .instance.Instanze)
	 */
	@Override
	public <I extends Instance<BinaryClassificationTarget>> void updateModel(I x) {
		double ip = kernelInnerProduct(x);
		double label = MLUtils.probToSVMInterval(x.getLabel().getValue());
		if (label * ip <= 0)
			instanceToLagrangeAndSign.put(x, new PrimitivePair(label, 1.));
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.Model#initialize()
	 */
	@Override
	public Forgetron initialize() {
		instanceToLagrangeAndSign = new LinkedHashMap<Instance<?>, PrimitivePair>() {
			private static final long serialVersionUID = -9996535572044538L;

			@Override
			protected boolean removeEldestEntry(
					Map.Entry<Instance<?>, PrimitivePair> p) {
				return size() > budget ? true : false;
			}
		};
		return model;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.AbstractModel#getModel()
	 */
	@Override
	protected Forgetron getModel() {
		return this;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.UpdateableClassifier#update(java.util.Collection
	 * )
	 */
	@Override
	public <I extends Instances<? extends Instance<BinaryClassificationTarget>>> void updateModel(
			I instst) {
		for (Instance<BinaryClassificationTarget> inst : instst)
			updateModel(inst);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.AbstractClassifier#regress(com.parallax.ml
	 * .instance.Instanze)
	 */
	@Override
	public double regress(Instance<?> x) {
		return kernelInnerProduct(x);
	}

	/**
	 * Kernel inner product; the inner product with the set of support vectors
	 * in the kernel space
	 * 
	 * @param x
	 *            instance who's inner product is desired
	 * @return the kernel inner product
	 */
	private double kernelInnerProduct(Instance<?> x) {
		double out = 0;
		for (Instance<?> y : instanceToLagrangeAndSign.keySet()) {
			PrimitivePair p = instanceToLagrangeAndSign.get(y);
			out += kernel.InnerProduct(x, y) * p.first * p.second;
		}
		return out;
	}

	/**
	 * Sets the budget.
	 * 
	 * @param budget
	 *            the budget
	 * @return the forgetron
	 */
	public Forgetron setBudget(int budget) {
		checkArgument(budget >= 1, "budget must be positive, given: %s", budget);
		this.budget = budget;
		return model;
	}

	/**
	 * Builder.
	 * 
	 * @param dimension
	 *            the dimension
	 * @param bias
	 *            the bias
	 * @return the forgetron builder
	 */
	public static ForgetronBuilder builder(int dimension, boolean bias) {
		return new ForgetronBuilder(dimension, bias);
	}

	/**
	 * The main method.
	 * 
	 * @param args
	 *            the arguments
	 * @throws Exception
	 *             the exception
	 */
	public static void main(String[] args) throws Exception {
		ClassifierEvaluation.evaluate(args, Forgetron.class);
	}

}
