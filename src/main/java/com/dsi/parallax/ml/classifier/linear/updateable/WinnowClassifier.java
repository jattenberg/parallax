/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.classifier.linear.updateable;

import com.dsi.parallax.ml.classifier.ClassifierEvaluation;
import com.dsi.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.WinnowClassifierBuilder;
import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.LinearVectorFactory;
import com.dsi.parallax.optimization.Gradient;
import com.dsi.parallax.optimization.stochastic.anneal.AnnealingSchedule;
import com.dsi.parallax.optimization.stochastic.anneal.ConstantAnnealingSchedule;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * a margin Winnow classifier- linear model using multiplicative updates<br>
 * 
 * @author jattenberg
 */
public class WinnowClassifier extends
		AbstractLinearUpdateableClassifier<WinnowClassifier> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -8537218210219031052L;

	/** The margin describing sensitivity to mistakes. */
	private double margin = 0.5;

	/** The weight describing the aggressiveness of updates. */
	private double weight = 2.;

	/**
	 * gradient updates are applied via addition; for multiplicitive updates
	 * like winnow we compute the difference, and just add it
	 */
	private static final AnnealingSchedule SCHEDULE = new ConstantAnnealingSchedule(
			1d);

	/**
	 * Instantiates a new winnow classifier.
	 * 
	 * @param dimension
	 *            the number of features in the instantiated classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 */
	public WinnowClassifier(int dimension, boolean bias) {
		super(dimension, bias);
		initialize();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.Model#initialize()
	 */
	@Override
	public WinnowClassifier initialize() {
		initW(1.);
		return model;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.classifier.linearupdateable.
	 * AbstractLinearUpdateableClassifier
	 * #computeUpdateGradient(com.parallax.ml.instance.Instanze)
	 */
	@Override
	protected <I extends Instance<BinaryClassificationTarget>> Gradient computeUpdateGradient(
			I x) {
		double prediction = innerProduct(x) > 0.5 ? 1 : 0;
		double label = x.getLabel().getValue();
		
		if (Math.abs(prediction - label) > margin) {
			LinearVector afterVector = LinearVectorFactory.getVector(dimension);

			if (prediction > label) { // demotion step
				for (int x_i : x) {
					afterVector.resetValue(x_i,
							getParam(x_i) / (weight * x.getWeight()));
				}
				if (bias) {
					afterVector.resetValue(dimension - 1,
							getParam(dimension - 1) / (weight * x.getWeight()));
				}
			} else { // promotion step
				for (int x_i : x) {
					afterVector.resetValue(x_i,
							getParam(x_i) * (weight * x.getWeight()));
				}
				if (bias) {
					afterVector.resetValue(dimension - 1,
							getParam(dimension - 1) * (weight * x.getWeight()));
				}
			}
			LinearVector gradientVector = afterVector.minus(vec);
			return new Gradient(gradientVector);
		}
		return new Gradient(LinearVectorFactory.getVector(dimension));
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
		return innerProduct(x);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.classifier.linearupdateable.
	 * AbstractLinearUpdateableClassifier
	 * #innerProduct(com.parallax.ml.instance.Instanze)
	 */
	@Override
	public double innerProduct(Instance<?> x) {
		double label = 0;
		for (int x_i : x)
			label += getParam(x_i);
		if (bias)
			label += getParam(dimension - 1);
		label /= x.L1Norm() + 1.0;
		return label;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.AbstractModel#getModel()
	 */
	@Override
	protected WinnowClassifier getModel() {
		return this;
	}

	/**
	 * Sets The margin describing sensitivity to mistakes. must be between 0 and
	 * 1 (inclusive)
	 * 
	 * @param margin
	 *            The margin describing sensitivity to mistakes
	 * @return the winnow classifier used for method chaining
	 */
	public WinnowClassifier setMargin(double margin) {
		checkArgument(margin >= 0 && margin <= 1,
				"margin must be between 0 and 1. input: %s", margin);
		this.margin = margin;
		return model;
	}

	/**
	 * Gets the margin describing sensitivity to mistakes
	 * 
	 * @return the margin describing sensitivity to mistakes
	 */
	public double getMargin() {
		return margin;
	}

	/**
	 * Sets the weight describing the aggressiveness of updates.
	 * 
	 * @param weight
	 *            The weight describing the aggressiveness of updates.
	 * @return the winnow classifier used for method chaining
	 */
	public WinnowClassifier setWeight(double weight) {
		checkArgument(weight > 0, "weight must be > 0, given: %s", weight);
		this.weight = weight;
		return model;
	}

	/**
	 * returns the weight describing the aggressiveness of updates
	 */
	public double getWeight() {
		return this.weight;
	}
	
	@Override
	public AnnealingSchedule getAnnealingSchedule() {
		return SCHEDULE;
	}

	/**
	 * Builder.
	 * 
	 * @param dimension
	 *            the dimension
	 * @param bias
	 *            the bias
	 * @return the winnow classifier builder
	 */
	public static WinnowClassifierBuilder builder(int dimension, boolean bias) {
		return new WinnowClassifierBuilder(dimension, bias);
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
		ClassifierEvaluation.evaluate(args, WinnowClassifier.class);
	}

}
