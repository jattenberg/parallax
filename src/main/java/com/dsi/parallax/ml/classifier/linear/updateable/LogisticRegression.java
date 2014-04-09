/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.classifier.linear.updateable;

import static com.google.common.base.Preconditions.checkArgument;

import com.dsi.parallax.ml.classifier.ClassifierEvaluation;
import com.dsi.parallax.ml.classifier.UpdateableType;
import com.dsi.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.LogisticRegressionBuilder;
import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.util.MLUtils;
import com.dsi.parallax.ml.util.SigmoidType;
import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.LinearVectorFactory;
import com.dsi.parallax.optimization.Gradient;


/**
 * The Class LogisticRegression. Incorporates several modifications beyond
 * vanilla logistic regression, including the notion of a:
 * </p>
 * "margin": <a href="http://qwone.com/~jason/writing/mmlr.pdf">Maximum-Margin Logistic Regression</a>
 * </p> and </p>
 * "sharpness": <a href="http://stat.rutgers.edu/home/tzhang/papers/ir01_textcat.pdf">Text categorization based on regularized linear classifcation methods</a>
 */
public class LogisticRegression extends
		AbstractLinearUpdateableClassifier<LogisticRegression> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -3335370006146357997L;

	/** The update. */
	private UpdateableType update = UpdateableType.COORDINATEWISE;

	/** The shift for max margin logistic regression. */
	private boolean shift = false;

	/** The sharpness factor of the generalized logistic regression: gamma.
	 * The higher the value the better the loss approximates the hinge loss. */
	private double gamma = 1;

	/**
	 * Instantiates a new logistic regression.
	 * 
	 * @param dimensions
	 *            the dimensions
	 * @param biasTerm
	 *            the bias term
	 */
	public LogisticRegression(int dimensions, boolean biasTerm) {
		super(dimensions, biasTerm);
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
		double innerProduct = innerProduct(x) * x.L2Norm();
		return SigmoidType.LOGIT.sigmoid(innerProduct);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.Model#initialize()
	 */
	@Override
	public LogisticRegression initialize() {
		initW();
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
		double y = MLUtils.probToSVMInterval(x.getLabel().getValue());
		double Z = innerProduct(x) * y;
		double internal = gamma * (-Z + (shift ? 1 : 0));
		double factor = x.L2Norm();

		LinearVector gradientVector = LinearVectorFactory.getVector(dimension);

		if (lastAccessed == null)
			lastAccessed = new int[dimension];
		if (internal >= -50 && internal <= 50) {
			double numerator = Math.exp(internal);
			double denominator = 1 + numerator;
			factor *= (numerator / denominator);
		} else if (internal < 50)
			factor = 0.;

		for (int x_i : x) {
			gradientVector.updateValue(x_i, y
					* (x.getFeatureValue(x_i) * factor));
		}

		if (bias) {
			gradientVector.updateValue(dimension - 1, y * factor);
		}
		return new Gradient(gradientVector);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.AbstractModel#getModel()
	 */
	@Override
	protected LogisticRegression getModel() {
		return this;
	}

	/**
	 * Sets the updateable type.
	 * 
	 * @param type
	 *            the type
	 * @return the logistic regression
	 */
	public LogisticRegression setUpdateableType(UpdateableType type) {
		this.update = type;
		return model;
	}

	/**
	 * Sets the shift.
	 * 
	 * @param useShift
	 *            the use shift
	 * @return the logistic regression
	 */
	public LogisticRegression setShift(boolean useShift) {
		this.shift = useShift;
		return model;
	}

	/**
	 * Sets the gamma.
	 * 
	 * @param gamma
	 *            the gamma
	 * @return the logistic regression
	 */
	public LogisticRegression setGamma(double gamma) {
		checkArgument(gamma > 0 && gamma <= 1,
				"gamma must be in (0, 1], given: %s", gamma);
		this.gamma = gamma;
		return model;
	}

	/**
	 * Gets the update.
	 * 
	 * @return the update
	 */
	public UpdateableType getUpdate() {
		return update;
	}

	/**
	 * Checks if is shift.
	 * 
	 * @return true, if is shift
	 */
	public boolean isShift() {
		return shift;
	}

	/**
	 * Gets the gamma.
	 * 
	 * @return the gamma
	 */
	public double getGamma() {
		return gamma;
	}

	/**
	 * Builder.
	 * 
	 * @param dimension
	 *            the dimension
	 * @param bias
	 *            the bias
	 * @return the logistic regression builder
	 */
	public LogisticRegressionBuilder builder(int dimension, boolean bias) {
		return new LogisticRegressionBuilder(dimension, bias);
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
		ClassifierEvaluation.evaluate(args, LogisticRegression.class);
	}

}
