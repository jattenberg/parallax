/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.classifier.kernelmethods;

import static com.google.common.base.Preconditions.checkArgument;

import java.util.Map;

import com.dsi.parallax.ml.classifier.ClassifierEvaluation;
import com.dsi.parallax.ml.classifier.kernelmethods.KernelMethodBuilder.KernelSGDBuilder;
import com.dsi.parallax.ml.evaluation.LossGradientType;
import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.instance.Instances;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.util.MLUtils;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Maps;
import com.google.common.collect.Table;

/**
 * A kernel-based model capable of optimizing several loss functions using SGD
 * 
 * @author jattenberg
 */
public class KernelSGD extends AbstractUpdateableKernelClassifier<KernelSGD> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -539120141511355493L;

	/** The loss function to be optimized. */
	private LossGradientType lossType = LossGradientType.HINGELOSS;

	/** The margin describing tolerance to errors. */
	private double margin = 1.;

	/** Eta describing the aggressiveness of updates */
	private double eta = 1;
	/**
	 * Lambda describing the importance of regularization on the lagrange
	 * multipliers
	 */
	private double lambda = 0.01;

	/** The kernel matrix. */
	private transient Table<Instance<?>, Instance<?>, Double> kernelMatrix;

	/** The lagrange multipliers for the instance keys. */
	private Map<Instance<?>, Double> alphas;

	/**
	 * Instantiates a new kernel sgd.
	 * 
	 * @param dimension
	 *            the number of features in the instantiated classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 */
	public KernelSGD(int dimension, boolean bias) {
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
	public <I extends Instance<BinaryClassificationTarget>> void updateModel(
			I instst) {
		double pred = kernelInnerProduct(instst);
		double update = lossType.computeLossUpdate(pred,
				MLUtils.probToSVMInterval(instst.getLabel().getValue()));
		if (!MLUtils.floatingPointEquals(0, update)) {
			double updateVal = (alphas.containsKey(instst) ? alphas.get(instst)
					: 0) + (update * eta);
			alphas.put(instst, updateVal);
		}
		for (Instance<?> inst : alphas.keySet()) {
			alphas.put(inst, alphas.get(inst) * (1. - eta * lambda));
		}

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
	protected double regress(Instance<?> inst) {
		double kip = kernelInnerProduct(inst);
		return MLUtils.svmIntervalToProb(kip);
	}

	/**
	 * sum_i alpha_i k(x_i, x), the model's "score" for the input example
	 * 
	 * @param inst
	 *            the instance to be scored
	 * @return the model's raw score for the given example
	 */
	private double kernelInnerProduct(Instance<?> inst) {
		double val = 0;
		for (Instance<?> x : alphas.keySet()) {
			val += kernelInnerProduct(x, inst) * alphas.get(x);
		}
		return val;
	}

	/**
	 * Kernel inner product; the inner product between two instances in the
	 * kernel space.
	 * 
	 * @param x
	 *            the x first example in the inner product
	 * @param y
	 *            the y second example in the inner product
	 * @return the the kernel inner product between the two given examples
	 */
	private double kernelInnerProduct(Instance<?> x, Instance<?> y) {
		int xHashCode = x.hashCode(), yHashCode = y.hashCode();
		Instance<?> first = null, second = null;
		if (xHashCode < yHashCode) {
			first = x;
			second = y;
		} else {
			first = y;
			second = x;
		}

		if (kernelMatrix.contains(first, second))
			return kernelMatrix.get(first, second);
		else {
			double ip = kernel.InnerProduct(first, second);
			kernelMatrix.put(first, second, ip);
			return ip;
		}
	}

	// model setup and configuration stuff below.

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.Model#initialize()
	 */
	@Override
	public KernelSGD initialize() {
		kernelMatrix = HashBasedTable.create();
		alphas = Maps.newHashMap();
		return model;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.AbstractModel#getModel()
	 */
	@Override
	protected KernelSGD getModel() {
		return this;
	}

	/**
	 * Gets the loss function to be optimized.
	 * 
	 * @return the loss function to be optimized
	 */
	public LossGradientType getLossType() {
		return lossType;
	}

	/**
	 * Gets the margin describing tolerance to errors.
	 * 
	 * @return the margin describing tolerance to errors
	 */
	public double getMargin() {
		return margin;
	}

	/**
	 * Gets the Eta describing the aggressiveness of updates.
	 * 
	 * @return the Eta describing the aggressiveness of updates
	 */
	public double getEta() {
		return eta;
	}

	/**
	 * Gets the Lambda describing the importance of regularization on the
	 * lagrange multipliers.
	 * 
	 * @return the Lambda describing the importance of regularization on the
	 *         lagrange multipliers
	 */
	public double getLambda() {
		return lambda;
	}

	/**
	 * Sets the loss function to be optimized
	 * 
	 * @param gradType
	 *            the loss function to be optimized
	 * @return the kernel sgd
	 */
	public KernelSGD setLossGradientType(LossGradientType gradType) {
		this.lossType = gradType;
		return model;
	}

	/**
	 * Sets the margin describing tolerance to errors.
	 * 
	 * @param margin
	 *            the margin describing tolerance to errors
	 * @return the kernel sgd used for method chaining
	 */
	public KernelSGD setMargin(double margin) {
		checkArgument(margin > 0, "margin must be > 0, given: %s", margin);
		this.margin = margin;
		return model;
	}

	/**
	 * Sets the Eta describing the aggressiveness of updates. <br>
	 * must be positive
	 * 
	 * @param eta
	 *            the Eta describing the aggressiveness of updates
	 * @return the kernel sgd used for method chaining
	 */
	public KernelSGD setEta(double eta) {
		checkArgument(eta > 0, "eta must be > 0, given: %s", eta);
		this.eta = eta;
		return model;
	}

	/**
	 * Sets the Lambda describing the importance of regularization on the
	 * lagrange multipliers. <br>
	 * (must be positive)
	 * 
	 * @param lambda
	 *            the Lambda describing the importance of regularization on the
	 *            lagrange multipliers
	 * @return the kernel sgd used for method chaining
	 */
	public KernelSGD setLambda(double lambda) {
		checkArgument(lambda > 0, "lambda must be > 0, given: %s", lambda);
		this.lambda = lambda;
		return model;
	}

	/**
	 * Generate a builder for KernelSGD classifiers
	 * 
	 * @param dimension
	 *            the number of features in the instantiated classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 * @return the kernel sgd builder
	 */
	public static KernelSGDBuilder builder(int dimension, boolean bias) {
		return new KernelSGDBuilder(dimension, bias);
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
		ClassifierEvaluation.evaluate(args, KernelSGD.class);
	}

}
