/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.classifier;

import com.dsi.parallax.ml.classifier.smoother.Smoother;
import com.dsi.parallax.ml.classifier.smoother.SmootherType;
import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.instance.Instances;
import com.dsi.parallax.ml.model.AbstractModel;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.util.pair.PrimitivePair;
import com.google.common.collect.Lists;

import java.io.Serializable;
import java.util.Collection;

/**
 * The base class for for any classifier. Handles the training and application
 * of probability smoothers- additional functions that transform raw
 * classifier scores into probability estimates.
 * 
 * @param <C>
 *            The type of classifier- used for method chaining.
 */
public abstract class AbstractClassifier<C extends AbstractClassifier<C>>
		extends AbstractModel<BinaryClassificationTarget, C> implements
		Classifier<C>, Serializable {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 6099958753798623365L;

	/** A default value representing a large number. */
	protected final double BIGDOUBLE = 10000000;

	/** The type of smoother to be used. */
	protected SmootherType smootherType = SmootherType.NONE;

	/**
	 * The smoothing implementation used for smoothing classifier scores
	 * estimates into probability.
	 */
	protected Smoother<?> smoother;

	/**
	 * when training smoother for probability calibration, use cross
	 * validation. number of folds used. (>=1)
	 */
	protected int crossValidateSmootherTraining = 1;

	/**
	 * Instantiates a new abstract classifier.
	 * 
	 * @param dimension
	 *            the number of features in the instantiated classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 */
	protected AbstractClassifier(int dimension, boolean bias) {
		super(dimension, bias);
	}

	/**
	 * Gets the type of probability smoother used.
	 * 
	 * @return the regtype
	 */
	public SmootherType getSmoothertype() {
		return smootherType;
	}

	/**
	 * Sets the type of probability Smoother to be used.
	 * 
	 * @param smootherType
	 *            the smootherType
	 * @return the classification model
	 */
	public C setSmoothertype(SmootherType smootherType) {
		this.smootherType = smootherType;
		return model;
	}

	/**
	 * determines if cross validation should be used when training
	 * 
	 * @return true if crossvalidation is used, else false.
	 */
	public int getCrossvalidateSmootherTraining() {
		return crossValidateSmootherTraining;
	}

	/**
	 * determines if cross validation should be used when training
	 * probability smoothing
	 * 
	 * @param crossvalidateSmootherTraining
	 * @return the classification model
	 */
	public C setCrossvalidateSmootherTraining(
			int crossvalidateSmootherTraining) {
		this.crossValidateSmootherTraining = crossvalidateSmootherTraining;
		return model;
	}

	/**
	 * Trains the probability smoother.
	 * 
	 * @param instances
	 *            labeled training data.
	 */
	private void trainSmoother(
			Instances<? extends Instance<BinaryClassificationTarget>> instances) {
		Collection<PrimitivePair> pairs = generateRawPairs(instances);
		smoother = smootherType.getAndTrainSmoother(pairs);
	}

	/**
	 * generate the raw prediction, label pairs used for training probability
	 * smoothers for instance, platt's smoothing
	 * 
	 * @param instances
	 *            used for creating training data
	 * @return a collection of prediction, label pairs
	 */
	private Collection<PrimitivePair> generateRawPairs(
			Instances<? extends Instance<BinaryClassificationTarget>> instances) {
		Collection<PrimitivePair> pairs = Lists.newLinkedList();
		for (Instance<BinaryClassificationTarget> inst : instances) {
			double prediction = regress(inst);
			double label = inst.getLabel().getValue();
			pairs.add(new PrimitivePair(prediction, label));
		}
		return pairs;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.model.Model#train(com.parallax.ml.instance.Instances)
	 */
	@SuppressWarnings("unchecked")
	@Override
	public <I extends Instances<? extends Instance<BinaryClassificationTarget>>> void train(
			I instances) {
		if (crossValidateSmootherTraining == 1) {
			modelTrain(instances);
			trainSmoother(instances);
		} else {
			Collection<PrimitivePair> pairs = Lists.newLinkedList();
			int folds = Math.min(instances.size(),
					crossValidateSmootherTraining);

			for (int fold = 0; fold < folds; fold++) {
				I training = (I) instances.getTraining(fold, folds);
				I testing = (I) instances.getTesting(fold, folds);

				initialize();
				modelTrain(training);
				pairs.addAll(generateRawPairs(testing));
			}
			smoother = smootherType.getAndTrainSmoother(pairs);
			initialize();
			modelTrain(instances);
		}
	}

	/**
	 * trains the predictive components of the model on the supplied instances
	 * 
	 * @param instances
	 *            the training data
	 */
	protected abstract <I extends Instances<? extends Instance<BinaryClassificationTarget>>> void modelTrain(
			I instances);

	/**
	 * Apply the probability smoother to transform the raw classifier score
	 * into a probability estimate.
	 * 
	 * @param inst
	 *            the instance being labeled
	 * @return the a probability estimate of class membership.
	 */
	protected double smooth(Instance<?> inst) {
		double prediction = predict(inst).getValue();
		return smooth(prediction);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.model.Model#predict(com.parallax.ml.instance.Instanze)
	 */
	@Override
	public BinaryClassificationTarget predict(Instance<?> x) {
		double ip = regress(x);
		return new BinaryClassificationTarget(smooth(ip));
	}

	/**
	 * Regress. Get the raw classifier score for a particular example
	 * 
	 * @param inst
	 *            the instance being labeled
	 * @return the raw classifier score
	 */
	protected abstract double regress(Instance<?> inst);

	/**
	 * Apply the probability smoothing to transform the raw classifier score
	 * into a probability estimate.
	 * 
	 * @param prediction
	 *            the raw classifier score
	 * @return the probability estimate of class membership
	 */
	protected double smooth(double prediction) {
		return (null == smoother) ? Smoother.DUMMY_SMOOTHER
				.smooth(prediction) : smoother.smooth(prediction);
	}

}
