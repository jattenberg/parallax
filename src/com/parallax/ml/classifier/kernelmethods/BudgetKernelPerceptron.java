/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.classifier.kernelmethods;

import static com.google.common.base.Preconditions.checkArgument;

import java.util.ArrayList;
import java.util.List;

import com.parallax.ml.classifier.ClassifierEvaluation;
import com.parallax.ml.classifier.kernelmethods.KernelMethodBuilder.BudgetKernelPerceptronBuilder;
import com.parallax.ml.instance.Instance;
import com.parallax.ml.instance.Instances;
import com.parallax.ml.target.BinaryClassificationTarget;
import com.parallax.ml.util.MLUtils;

/**
 * An online kernel-based classifier with a fixed budget for support vectors.
 * {@link <a href="http://homes.di.unimi.it/~cesabian/Pubblicazioni/J29.pdf">Tracking the best hyperplane with a simple budget Perceptron</a>}
 * by Giovanni Cavallanti, Nicolo Cesa-Bianchi & Claudio Gentile
 */
public class BudgetKernelPerceptron extends
		AbstractUpdateableKernelClassifier<BudgetKernelPerceptron> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 9060194914025549778L;

	/** The budget for support vectors */
	private int N = 100;

	/** The margin for acceptable errors */
	private double margin = 0.1;

	/** The pool of support vectors */
	protected List<Instance<?>> pool = null;

	/** The largrange multipliers associated with the support vectors */
	protected List<Double> alpha = null;

	/**
	 * Instantiates a new budget kernel perceptron.
	 * 
	 * @param dimension
	 *            the number of features in the instantiated classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 */
	public BudgetKernelPerceptron(int dimension, boolean bias) {
		super(dimension, bias);
		initialize();
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
	 * com.parallax.ml.classifier.UpdateableClassifier#updateModel(com.parallax.ml
	 * .instance.Instanze)
	 */
	@Override
	public <I extends Instance<BinaryClassificationTarget>> void updateModel(I x) {
		if (pool.size() == 0) {
			pool.add(x);
			alpha.add(MLUtils.probToSVMInterval(x.getLabel().getValue()));
		} else {
			double pred = regress(x);
			double act = MLUtils.probToSVMInterval(x.getLabel().getValue());
			if (pred * act <= margin) {
				pool.add(x);
				alpha.add(act * x.getWeight());
				prune(x, pred, act * x.getWeight());
			}
		}
	}

	/**
	 * Prune the set of support vectors to adhere to the budget.
	 * 
	 * @param <I>
	 *            the type of instance used.
	 * @param x
	 *            candidate example to be added to the pool
	 * @param pred
	 *            the predicted label
	 * @param act
	 *            the actual label
	 */
	private <I extends Instance<BinaryClassificationTarget>> void prune(I x,
			double pred, double act) {
		if (pool.size() > N) {
			double val = act
					* (pred - alpha.get(0)
							* kernel.InnerProduct(x, pool.get(0)));
			int index = 0;
			for (int i = 1; i < pool.size(); i++) {
				double diff = act
						* (pred - alpha.get(i)
								* kernel.InnerProduct(x, pool.get(i)));
				if (diff > val) {
					val = diff;
					index = i;
				}
			}
			pool.remove(index);
			alpha.remove(index);
		}
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
		double label = 0.0;
		for (int i = 0; i < pool.size(); i++) {
			label += alpha.get(i) * kernel.InnerProduct(x, pool.get(i));
		}
		return label;
	}

	/**
	 * Gets the pool size; the budget for supprot vectors.
	 * 
	 * @return the pool size; the budget for support vectors
	 */
	public int getPoolSize() {
		return N;
	}

	/**
	 * Gets the margin for acceptable errors
	 * 
	 * @return the margin for acceptable errors
	 */
	public double getMargin() {
		return margin;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.Model#initialize()
	 */
	@Override
	public BudgetKernelPerceptron initialize() {
		pool = new ArrayList<Instance<?>>();
		alpha = new ArrayList<Double>();
		return model;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.AbstractModel#getModel()
	 */
	@Override
	protected BudgetKernelPerceptron getModel() {
		return this;
	}

	/**
	 * margin for correct classification- errors within the boundary don't
	 * produce an update. larger values lead to less frequent updates, but
	 * models with less influence from noise
	 * 
	 * @param margin
	 *            input margin, [0,1]
	 * @return the budget kernel perceptron
	 */
	public BudgetKernelPerceptron setMargin(double margin) {
		checkArgument(margin >= 0 && margin <= 1,
				"margin must be between 0 and 1. input: %s", margin);
		this.margin = margin;
		return model;
	}

	/**
	 * size of pool of support vectors. large values lead to more detailed
	 * models but require more system resources
	 * 
	 * @param n
	 *            the n
	 * @return the budget kernel perceptron
	 */
	public BudgetKernelPerceptron setPoolSize(int n) {
		checkArgument(N > 0, "number of instances must be positive, given %s",
				N);
		this.N = n;
		return model;
	}

	/**
	 * Builder for {@link BudgetKernelPerceptron}
	 * 
	 * @param dimension
	 *            the number of features in the instantiated classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 * @return the budget kernel perceptron builder
	 */
	public static BudgetKernelPerceptronBuilder builder(int dimension,
			boolean bias) {
		return new BudgetKernelPerceptronBuilder(dimension, bias);
	}

	/**
	 * The main method.
	 * 
	 * @see {@link ClassifierEvaluation#evaluate(String[], Class)}
	 * 
	 * @param args
	 *            the arguments
	 * @throws Exception
	 *             the exception
	 */
	public static void main(String[] args) throws Exception {
		ClassifierEvaluation.evaluate(args, BudgetKernelPerceptron.class);
	}
}
