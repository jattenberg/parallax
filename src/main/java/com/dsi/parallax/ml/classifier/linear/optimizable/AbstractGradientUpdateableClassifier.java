/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.classifier.linear.optimizable;

import com.dsi.parallax.ml.classifier.AbstractUpdateableClassifier;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.instance.Instances;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.LinearVectorFactory;
import com.dsi.parallax.optimization.Gradient;
import com.dsi.parallax.optimization.Optimizable;
import com.dsi.parallax.optimization.stochastic.GradientStochasticOptimizer;
import com.dsi.parallax.optimization.stochastic.SGDBuilder;
import com.dsi.parallax.optimization.stochastic.StochasticGradientOptimizationBuilder;
import com.dsi.parallax.optimization.stochastic.anneal.AnnealingScheduleConfigurableBuilder;

import java.util.Collection;
import java.util.List;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Base class for updateable classifiers that can be trained using the using the
 * gradient of their loss function using an arbitrary gradient-based
 * optimization technique.
 * 
 * @param <C>
 *            the concrete type of GradientUpdateableClassifier, used for method
 *            chaining.
 * @see {@link GradientStochasticOptimizer}
 */
public abstract class AbstractGradientUpdateableClassifier<C extends AbstractGradientUpdateableClassifier<C>>
		extends AbstractUpdateableClassifier<C> implements
		GradientUpdateableClassifier<C>, Optimizable {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -3487114825971258955L;

	/**
	 * Builder for stochastic optimization methods. @see
	 * {@link GradientStochasticOptimizer}
	 */
	protected StochasticGradientOptimizationBuilder<?> builder;

	/**
	 * The optimizer used for improving the model based on the gradient (or
	 * mini-batch gradient) of the loss function.
	 */
	protected GradientStochasticOptimizer optimizer;

	/**
	 * The parameters; the data structure representing the model, the w (as in
	 * w'x) in the case of linear models.
	 */
	protected LinearVector parameters;

	/**
	 * Number of passes over the data when doing batch training. @see
	 * {@link #train(Instances)}
	 */
	protected int passes = 3;

	/**
	 * The number of examples to build a gradient from when doing batch
	 * training, or to subsample from when doing given a collection of
	 * instances. @see {@link #train(Instances)},
	 * 
	 * @see {@link #update(Collection)}
	 */
	protected int miniBatchSize = 5;

	/**
	 * last observed set of instances, used for loss and gradient computations
	 * in certain situations.
	 */
	protected transient BinaryClassificationInstances instances;

	/**
	 * Instantiates a new abstract gradient updateable classifier.
	 * 
	 * @param builder
	 *            Builder for stochastic optimization methods
	 * @param dimension
	 *            the number of features in the instantiated classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 */
	public AbstractGradientUpdateableClassifier(
			StochasticGradientOptimizationBuilder<?> builder, int dimension,
			boolean bias) {
		super(dimension, bias);
		this.builder = builder;
		initW();
	}

	/**
	 * Instantiates a new abstract gradient updateable classifier. by default
	 * optimizes using a {@link SGDBuilder}
	 * 
	 * @param dimension
	 *            the number of features in the instantiated classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 */
	public AbstractGradientUpdateableClassifier(int dimension, boolean bias) {
		this(
				new SGDBuilder(dimension, bias)
						.setAnnealingScheduleConfigurableBuilder(AnnealingScheduleConfigurableBuilder
								.configureForConstantRate(0.001)), dimension,
				bias);
	}

	/**
	 * Initializes the internal parameters representing the model.
	 */
	private void initW() {
		parameters = LinearVectorFactory.getVector(dimension);
		;
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
		this.updateModel(collect(instst));

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
			I insts) {
		this.instances = (BinaryClassificationInstances) insts;
		optimizer.update(this);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.model.Model#train(com.parallax.ml.instance.Instances)
	 */
	@Override
	public <I extends Instances<? extends Instance<BinaryClassificationTarget>>> void modelTrain(
			I instances) {
		for (int i = 0; i < passes; i++) {

			List<? extends Instance<BinaryClassificationTarget>> instList = instances
					.getInstances();
			miniBatchify(instList);
		}
	}

	/**
	 * sub-sample from a list of instances to the internal mini-batch size.
	 * 
	 * @param instList
	 *            list of training data to be sampled.
	 */
	private void miniBatchify(
			List<? extends Instance<BinaryClassificationTarget>> instList) {
		int start = 0;
		while (start < instList.size()) {
			int end = Math.min(start + miniBatchSize, instList.size());
			updateModel(collect(instList.subList(start, end), dimension));
			start += miniBatchSize;
		}
	}

	/**
	 * Sets the passes; Number of passes over the data when doing batch
	 * training.
	 * 
	 * @param passes
	 *            number of passes over the data
	 * @return the model itself, used for method chaining
	 * @see {@link #train(Instances)}
	 */
	@Override
	public C setPasses(int passes) {
		checkArgument(passes > 0, "passes must be greater than 0. input: %d",
				passes);
		this.passes = passes;
		return model;
	}

	/**
	 * Gets the passes; Number of passes over the data when doing batch
	 * training.
	 * 
	 * @return the passes
	 */
	@Override
	public int getPasses() {
		return passes;
	}

	/**
	 * Gets the mini batch size; The number of examples to build a gradient from
	 * when doing batch training, or to subsample from when doing given a
	 * collection of instances. @see {@link #train(Instances)},
	 * 
	 * @return the preferred size of minibatches
	 * @see {@link #update(Collection)}
	 */
	public int getMiniBatchSize() {
		return miniBatchSize;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.Model#initialize()
	 */
	@Override
	public C initialize() {
		initializeOptimizer();
		return model;
	}

	/**
	 * Initialize the internal optimization method.
	 */
	private void initializeOptimizer() {
		optimizer = builder.build();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.optimization.Optimizable#getParameters()
	 */
	@Override
	public LinearVector getVector() {
		return parameters;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.optimization.Optimizable#getParameter(int)
	 */
	@Override
	public double getParameter(int index) {
		return parameters.getValue(index);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.optimization.Optimizable#setParameter(int, double)
	 */
	@Override
	public void setParameter(int index, double value) {
		parameters.resetValue(index, value);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.optimization.Optimizable#computeGradient()
	 */
	@Override
	public Gradient computeGradient() {
		return computeGradient(parameters);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.optimization.Optimizable#computeLoss()
	 */
	@Override
	public double computeLoss() {
		return computeLoss(parameters);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.optimization.Optimizable#getNumParameters()
	 */
	@Override
	public int getNumParameters() {
		return dimension + (bias ? 1 : 0);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.optimization.Optimizable#setParameters(com.parallax.ml.util
	 * .vector.LinearVector)
	 */
	@Override
	public void setParameters(LinearVector params) {
		this.parameters = params;
	}

	@Override
	public <O extends StochasticGradientOptimizationBuilder<O>> C setOptimizationBuilder(
			O optimizer) {
		checkArgument(
				optimizer.getDimension() == this.dimension,
				"optimization builder dimension (%s) should match model dimension (%s)",
				optimizer.getDimension(), dimension);
		checkArgument(optimizer.getBias() == this.bias,
				"optimization builder bias (%s) should match model bias (%s)",
				optimizer.getBias(), bias);
		this.builder = optimizer;
		return model;
	}

	@Override
	public StochasticGradientOptimizationBuilder<?> getOptimizationBuilder() {
		return this.builder;
	}
}
