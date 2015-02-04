/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.classifier.linear.updateable;

import com.dsi.parallax.ml.classifier.AbstractUpdateableClassifier;
import com.dsi.parallax.ml.classifier.linear.LinearClassifier;
import com.dsi.parallax.ml.classifier.linear.LinearModelPrinter;
import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.instance.Instances;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.util.MLUtils;
import com.dsi.parallax.ml.util.option.Configuration;
import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.LinearVectorFactory;
import com.dsi.parallax.ml.vector.VectorType;
import com.dsi.parallax.optimization.Gradient;
import com.dsi.parallax.optimization.regularization.*;
import com.dsi.parallax.optimization.stochastic.anneal.AnnealingSchedule;
import com.dsi.parallax.optimization.stochastic.anneal.ConstantAnnealingSchedule;
import com.google.common.collect.Maps;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Base class for all linear updateable classification models. Maintains common
 * datastructures (eg, the linear model itself), Vector truncation type and
 * info, regularization hyper-parameters, and utility methods common to linear
 * models.
 * 
 * @param <C>
 *            the concrete LinearUpdateableClassifier. Used for method chaining.
 * @author jattenberg
 */
public abstract class AbstractLinearUpdateableClassifier<C extends AbstractLinearUpdateableClassifier<C>>
		extends AbstractUpdateableClassifier<C> implements LinearClassifier<C> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -7694419099064518567L;

	/** The parameters representing the linear model */
	protected LinearVector vec;

	/**
	 * The epoch when each parameter in the linear model was updated. Used for
	 * lazy regularization
	 */
	protected transient int[] lastAccessed;

	/** The vector truncation to be used. Defaults to {@link NullTruncation} */
	protected GradientTruncation truncation;

	/** The builder for truncation types. */
	protected TruncationConfigurableBuilder truncationBuilder = new TruncationConfigurableBuilder();

	/**
	 * The type of vector used to represent the internal data structures. @see
	 * {@link VectorType}
	 */
	private final VectorType linType;

	/** should regularization be performed on the bias (intercept) term? */
	protected boolean regularizeIntercept = false;

	/** The epoch; number of updates that have been performed on the model. */
	protected int epoch; // how many updates have been performed?

	// regularization parameters
	/**
	 * The regularization weight; the combined effect multiplier on all
	 * regularizations
	 */
	protected double regularizationWeight = 1;

	/** The coefficient weights; multipliers on individual regularization types */
	private Map<LinearCoefficientLossType, Double> regularizationCoefficientWeights;

	private AnnealingSchedule annealingSchedule = new ConstantAnnealingSchedule(
			0.1d);

	/**
	 * Instantiates a new abstract linear updatable classifier.
	 * 
	 * @param dimension
	 *            the number of features in the instantiated classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 */
	protected AbstractLinearUpdateableClassifier(int dimension, boolean bias) {
		super(dimension, bias);
		linType = VectorType.DOUBLEARR;
		initW();
		intializeCoefficientLosses();
		truncation = initializeTruncation();
	}

	/**
	 * Instantiates a new abstract linear updatable classifier.
	 * 
	 * @param dimension
	 *            the number of features in the instantiated classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 * @param dense
	 *            should the model have a sparse or dense representation
	 */
	protected AbstractLinearUpdateableClassifier(int dimension, boolean bias,
			boolean dense) {
		super(dimension, bias);
		linType = dense ? VectorType.DOUBLEARR : VectorType.TROVEMAP;
		initW();
		intializeCoefficientLosses();
		truncation = initializeTruncation();
	}

	/**
	 * Instantiates a new abstract linear updateable classifier.
	 * 
	 * @param dimension
	 *            the number of features in the instantiated classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 * @param initValue
	 *            what value should the linear vector default to?
	 */
	protected AbstractLinearUpdateableClassifier(int dimension, boolean bias,
			double initValue) {
		super(dimension, bias);
		linType = (dimension <= 1024 ? VectorType.DOUBLEARR
				: VectorType.TROVEMAP);
		initW(initValue);
		intializeCoefficientLosses();
		truncation = initializeTruncation();
	}

	/**
	 * Instantiates a new abstract linear updateable classifier.
	 * 
	 * @param dimension
	 *            the number of features in the instantiated classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 * @param initValue
	 *            what value should the linear vector default to?
	 * @param dense
	 *            should the model have a sparse or dense representation?
	 */
	protected AbstractLinearUpdateableClassifier(int dimension, boolean bias,
			double initValue, boolean dense) {
		super(dimension, bias);
		linType = dense ? VectorType.DOUBLEARR : VectorType.TROVEMAP;
		initW(initValue);
		intializeCoefficientLosses();
		truncation = initializeTruncation();
	}

	/**
	 * applies the current truncation to the parameters
	 */
	protected void applyTruncation() {
		if (truncation.getType() != TruncationType.NONE) {
			truncation.truncateParameters(vec);
		}
	}

	/**
	 * impact of cauchy regularization.
	 * 
	 * @return squaredWeight
	 */
	public double getCauchyRegularizationWeight() {
		return getRegularizationTypeWeight(LinearCoefficientLossType.CAUCHY);
	}

	/**
	 * Gets the coefficient weights.
	 * 
	 * @return the coefficient weights
	 */
	public Map<LinearCoefficientLossType, Double> getCoefficientWeights() {
		return regularizationCoefficientWeights;
	}

	/**
	 * impact of gaussian regularization.
	 * 
	 * @return squaredWeight
	 */
	public double getGaussianRegularizationWeight() {
		return getRegularizationTypeWeight(LinearCoefficientLossType.GAUSSIAN);
	}

	/**
	 * impact of laplace regularization.
	 * 
	 * @return squaredWeight
	 */
	public double getLaplaceRegularizationWeight() {
		return getRegularizationTypeWeight(LinearCoefficientLossType.LAPLACE);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.classifier.LinearClassifier#getParam(int)
	 */
	@Override
	public double getParam(int index) {
		return vec.getValue(index);
	}

	/**
	 * Gets the weight of the specified regularization type.
	 * 
	 * @param lossType
	 *            the loss type requested.
	 * @return the regularization weight
	 */
	public double getRegularizationTypeWeight(LinearCoefficientLossType lossType) {
		return regularizationCoefficientWeights.containsKey(lossType) ? regularizationCoefficientWeights
				.get(lossType) : 0;
	}

	/**
	 * Gets the regularization weight.
	 * 
	 * @return the regularization weight
	 */
	public double getRegularizationWeight() {
		return regularizationWeight;
	}

	/**
	 * impact of squared regularization.
	 * 
	 * @return squaredWeight
	 */
	public double getSquaredRegularizationWeight() {
		return getRegularizationTypeWeight(LinearCoefficientLossType.SQUARED);
	}

	/**
	 * Gets the truncation builder.
	 * 
	 * @return the truncation builder
	 */
	public TruncationConfigurableBuilder getTruncationBuilder() {
		return truncationBuilder;
	}

	// getters
	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.classifier.LinearClassifier#getVector()
	 */
	@Override
	public LinearVector getVector() {
		return vec;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.classifier.LinearClassifier#getW()
	 */
	@Override
	public double[] getW() {
		return vec.getW();
	}

	/**
	 * Infinity norm of params.
	 * 
	 * @return ||w||_\infty
	 */
	protected double infinityNormOfParams() {
		return MLUtils.infinityNorm(vec.getW());
	}

	/**
	 * Initialize the truncation data structures used for shrinking the model
	 * parameters.
	 */
	protected GradientTruncation initializeTruncation() {
		epoch = 0;
		return truncationBuilder.build();
	}

	/**
	 * Initializes the weights (and a few other things) 
	 * of the LinearClassifier. The weights are set to zero. 
	 *  
	 * @see com.parallax.ml.classifier.LinearClassifier#initW()
	 */
	@Override
	public void initW() {
		vec = linType.getVector(dimension);
		lastAccessed = new int[dimension];
		Arrays.fill(lastAccessed, epoch - 1);
	}

	/**
	 * Initializes the weights (and a few other things) of the LinearClassifer. 
	 * The weights are set to a the value of "param" 
	 *  
	 * @see com.parallax.ml.classifier.LinearClassifier#initW(double)
	 */
	@Override
	public void initW(double param) {
		vec = linType.getVector(dimension, param);
		lastAccessed = new int[dimension];
		Arrays.fill(lastAccessed, epoch - 1);
	}

	/**
	 * Inner product; w'x for some input instance x
	 * 
	 * @param x
	 *            input for which the inner product with the model is requested.
	 * @return w'x for the input x
	 */
	protected double innerProduct(Instance<?> x) {
		double tot = 0.;
		for (int x_i : x) {
			tot += x.getFeatureValue(x_i) * getParam(x_i);
		}
		if (bias) {
			tot += getParam(dimension - 1);
		}
		return tot;
	}

	/**
	 * Initialize coefficient losses.
	 */
	private void intializeCoefficientLosses() {
		regularizationCoefficientWeights = Maps.newHashMap();
		regularizationCoefficientWeights.put(
				LinearCoefficientLossType.GAUSSIAN, 0.);
		regularizationCoefficientWeights.put(LinearCoefficientLossType.LAPLACE,
				0.);
		regularizationCoefficientWeights.put(LinearCoefficientLossType.CAUCHY,
				0.);
		regularizationCoefficientWeights.put(LinearCoefficientLossType.SQUARED,
				0.);
	}

	/**
	 * Checks if is regularize intercept.
	 * 
	 * @return true, if is regularize intercept
	 */
	public boolean isRegularizeIntercept() {
		return regularizeIntercept;
	}

	/**
	 * One norm of params.
	 * 
	 * @return ||w||_1
	 */
	protected double oneNormOfParams() {
		return MLUtils.oneNorm(vec.getW());
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.classifier.LinearClassifier#getVector()
	 */
	@Override
	public String prettyPrint() {
		return LinearModelPrinter.prettyPrintVector(vec);
	}

	/**
	 * returns the gradient of the regularization loss along a particular
	 * dimension.
	 * 
	 * @param dim
	 *            dimension of the model being regularized
	 * @return the amount of regularization to be performed.
	 */
	private double regularizationLoss(int dim) {

		double loss = 0;

		if ((dim == dimension - 1 && bias && !regularizeIntercept)
				|| !(regularizationWeight > 0))
			return loss;

		for (LinearCoefficientLossType lossType : regularizationCoefficientWeights
				.keySet()) {
			if (regularizationCoefficientWeights.get(lossType) > 0)
				loss += lossType.gradient(vec.getValue(dim),
						regularizationCoefficientWeights.get(lossType));
		}

		loss *= (regularizationWeight * (lastAccessed[dim] - epoch))
				/ dimension;
		lastAccessed[dim] = epoch;
		return loss;
	}

	/**
	 * impact of cauchy regularization. (>=0)
	 * 
	 * @param cauchyWeight
	 *            the cauchy weight
	 * @return the model itself, used for method chaining
	 */
	public C setCauchyRegularizationWeight(double cauchyWeight) {
		checkArgument(cauchyWeight >= 0,
				"cauchyWeight must be nonnegative. given: %s", cauchyWeight);
		regularizationCoefficientWeights.put(
				(LinearCoefficientLossType.CAUCHY), cauchyWeight);
		return model;
	}

	/**
	 * impact of gaussian regularization. (>=0)
	 * 
	 * @param gaussianWeight
	 *            the gaussian weight
	 * @return the model itself, used for method chaining
	 */
	public C setGaussianRegularizationWeight(double gaussianWeight) {
		checkArgument(gaussianWeight >= 0,
				"gaussianWeight must be nonnegative. given: %s", gaussianWeight);
		regularizationCoefficientWeights.put(
				(LinearCoefficientLossType.GAUSSIAN), gaussianWeight);
		return model;
	}

	/**
	 * impact of laplace regularization. (>=0)
	 * 
	 * @param laplaceWeight
	 *            the laplace weight
	 * @return the model itself, used for method chaining
	 */
	public C setLaplaceRegularizationWeight(double laplaceWeight) {
		checkArgument(laplaceWeight >= 0,
				"laplaceWeight must be nonnegative. given: %s", laplaceWeight);
		regularizationCoefficientWeights.put(
				(LinearCoefficientLossType.LAPLACE), laplaceWeight);
		return model;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.classifier.LinearClassifier#setParam(double, int)
	 */
	@Override
	public void setParam(double wi, int index) {
		vec.resetValue(index, wi);
	}

	/**
	 * a global weight on all types of regularization used when training the
	 * model.
	 * 
	 * @param regularizationWeight
	 *            the regularization weight
	 * @return the c
	 */
	public C setRegularizationWeight(double regularizationWeight) {
		checkArgument(regularizationWeight >= 0,
				"lambda must be >= 0. input: %s", regularizationWeight);
		this.regularizationWeight = regularizationWeight;
		return model;
	}

	/**
	 * should regularization be used on the intercept (bias) term when training
	 * the regression model?.
	 * 
	 * @param regularizeIntercept
	 *            the regularize intercept
	 * @return the c
	 */
	public C setRegularizeIntercept(boolean regularizeIntercept) {
		this.regularizeIntercept = regularizeIntercept;
		return model;
	}

	/**
	 * impact of squared regularization. (>=0)
	 * 
	 * @param squaredWeight
	 *            the squared weight
	 * @return the model itself, used for method chaining
	 */
	public C setSquaredRegularizationWeight(double squaredWeight) {
		checkArgument(squaredWeight >= 0,
				"squaredWeight must be nonnegative. given: %s", squaredWeight);
		regularizationCoefficientWeights.put(
				(LinearCoefficientLossType.SQUARED), squaredWeight);
		return model;
	}

	/**
	 * Sets the truncation builder.
	 * 
	 * @param truncationConfiguration
	 *            configuration for a truncation builder
	 * @return the model itself, used for method chaining.
	 */
	public C setTruncationBuilder(
			Configuration<TruncationConfigurableBuilder> truncationConfiguration) {
		return setTruncationBuilder(new TruncationConfigurableBuilder(
				truncationConfiguration));
	}

	/**
	 * Sets the truncation builder.
	 * 
	 * @param truncationBuilder
	 *            the truncation builder
	 * @return the model itself, used for method chaining.
	 */
	public C setTruncationBuilder(
			TruncationConfigurableBuilder truncationBuilder) {
		this.truncationBuilder = truncationBuilder;
		this.truncation = initializeTruncation();
		return model;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.classifier.LinearClassifier#setW(double[])
	 */
	@Override
	public void setW(double[] W) {
		vec = LinearVectorFactory.getVector(W);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.classifier.LinearClassifier#setW(java.util.List)
	 */
	@Override
	public void setW(List<Double> W) {
		vec.setW(W);
	}

	public AnnealingSchedule getAnnealingSchedule() {
		return annealingSchedule;
	}

	public C setAnnealingSchedule(AnnealingSchedule annealingSchedule) {
		this.annealingSchedule = annealingSchedule;
		return model;
	}

	@Override
	public String toString() {
		return prettyPrint();
	}

	/**
	 * Two norm of params.
	 * 
	 * @return ||w||_2
	 */
	protected double twoNormOfParams() {
		return MLUtils.twoNorm(vec.getW());
	}

	/**
	 * Generate the update that would occur from the unscaled incorporation of
	 * information from a specific instance.
	 * 
	 * TODO: use weighted gradient
	 * 
	 * @param <I>
	 *            the type of training data to be considered.
	 * @param inst
	 *            training data used to improve the model.
	 * 
	 * @return Gradient the ADDITIVE change that should result from an unscaled
	 *         incorporation of a single example information
	 */
	protected abstract <I extends Instance<BinaryClassificationTarget>> Gradient computeUpdateGradient(
			I inst);

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
		for (Instance<BinaryClassificationTarget> inst : insts) {
			updateModel(inst);
		}
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
			I instance) {
		Gradient grad = computeUpdateGradient(instance);
		scaledGradientUpdate(grad);
		scaledRegularizationUpdate(grad);

		applyTruncation();
		epoch++;
	}

	/**
	 * applies regularization to the linear model based on the most recent
	 * input.
	 * 
	 * @param gradient
	 *            corresponding to most recent model update
	 */
	private void scaledRegularizationUpdate(Gradient grad) {
		if (regularizationWeight <= 0
				|| regularizationCoefficientWeights.isEmpty()) {
			return;
		}
		for (int x_i : grad) {
			scaledRegularizationUpdate(x_i);
		}
	}

	/**
	 * applies regularization for a single dimension, scaled according to the
	 * current learning rate
	 * 
	 * @param x_i
	 */
	protected void scaledRegularizationUpdate(int x_i) {
		double regularizationChange = regularizationLoss(x_i)
				* updateScale(x_i);
		// TODO: this may not be sufficient
		// for more dynamic annealing schedules.
		// think more about it.
		double current = getParam(x_i); // dont want to "cross over" 0 when
										// regularizing, this may lead to
										// some thrashing.

		double difference = current - regularizationChange;
		if (Math.signum(difference) != Math.signum(current)) {
			updateParam(x_i, -current);
		} else {
			updateParam(x_i, -regularizationChange);
		}
	}

	/**
	 * computes the scaling factor according to the annealing schedule and
	 * updates the parameters of the linear model based on this.
	 * 
	 * @param grad
	 */
	private void scaledGradientUpdate(Gradient grad) {
		for (int x_i : grad) {
			double scale = updateScale(x_i);
			updateParam(x_i, grad.getValue(x_i) * scale);
		}
	}

	/**
	 * computes the scale according to the annealing schedule used. TODO:
	 * multidimensional annealing schedules.
	 * 
	 * @param x_i
	 *            dimension of the update
	 * @return
	 */
	private double updateScale(int x_i) {
		return getAnnealingSchedule().learningRate(epoch, x_i);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.classifier.LinearClassifier#updateParam(double, int)
	 */
	@Override
	public void updateParam(int index, double wi) {
		vec.updateValue(index, wi);
	}

}
