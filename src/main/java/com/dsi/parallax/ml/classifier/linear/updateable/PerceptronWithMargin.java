/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.classifier.linear.updateable;

import static com.google.common.base.Preconditions.checkArgument;

import com.dsi.parallax.ml.classifier.ClassifierEvaluation;
import com.dsi.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.PerceptronWithMarginBuilder;
import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.util.MLUtils;
import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.LinearVectorFactory;
import com.dsi.parallax.optimization.Gradient;

// TODO: Auto-generated Javadoc
/*
 * computes y = w'x
 * also provides techniques for training w and representing x 
 * to do: read saved model file
 * 
 * note: this maintains a SINGLE classifier for binary classification. multi-class/etc must be implemented externally\
 * by combining several LinearClassifier classes
 * 
 */
/**
 * The Class PerceptronWithMargin.
 */
public class PerceptronWithMargin extends
		AbstractLinearUpdateableClassifier<PerceptronWithMargin> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 1L;

	/** The margin. */
	private double margin = 0.4;

	/**
	 * Instantiates a new perceptron with margin.
	 * 
	 * @param dimension
	 *            the dimension
	 * @param bias
	 *            the bias
	 */
	public PerceptronWithMargin(int dimension, boolean bias) {
		super(dimension, bias);
		initialize();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.Model#initialize()
	 */
	@Override
	public PerceptronWithMargin initialize() {
		initW(1. / Math.sqrt(dimension + (bias ? 1. : 0.)));
		return model;
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
		double xnorm = x.L2Norm();
		if (bias)
			xnorm = Math.sqrt(xnorm * xnorm + 1.);

		return innerProduct(x, xnorm);

	}

	/**
	 * Inner product.
	 * 
	 * @param x
	 *            the x
	 * @param xnorm
	 *            the xnorm
	 * @return the double
	 */
	private double innerProduct(Instance<?> x, double xnorm) {
		if (MLUtils.floatingPointEquals(xnorm, 0))
			return 0;
		double out = 0.;
		for (int x_i : x)
			out += x.getFeatureValue(x_i) * getParam(x_i);
		if (bias)
			out += getParam(dimension - 1);
		return out / xnorm;
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
		// w_(k+1) = w_(k) + eta*(y_k - w_k^T*x_k)*w_k

		double label = MLUtils.probToSVMInterval(x.getLabel().getValue());
		double xnorm = x.L2Norm();

		if (bias) {
			xnorm = Math.sqrt(xnorm * xnorm + 1.);
		}

		LinearVector gradientVector = LinearVectorFactory.getVector(dimension);

		double innerProduct = innerProduct(x, xnorm) / vec.L2Norm();
		
		if (label * innerProduct < margin / 2.
				&& !MLUtils.floatingPointEquals(0, xnorm)) {
			for (int x_i : x) {
				double y_i = x.getFeatureValue(x_i) / xnorm;
				double update = label * y_i;
				gradientVector.updateValue(x_i, update);

			}
			if (bias) {
				double y_i = 1. / xnorm;

				double update = label * y_i;
				gradientVector.updateValue(dimension - 1, update);
			}
		}
		
		return new Gradient(gradientVector);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.AbstractModel#getModel()
	 */
	@Override
	protected PerceptronWithMargin getModel() {
		return this;
	}

	/**
	 * Sets the margin.
	 * 
	 * @param margin
	 *            the margin
	 * @return the perceptron with margin
	 */
	public PerceptronWithMargin setMargin(double margin) {
		checkArgument(margin >= 0 && margin <= 1,
				"margin must be between 0 and 1. input: %f", margin);
		this.margin = margin;
		return model;
	}

	/**
	 * Gets the margin.
	 * 
	 * @return the margin
	 */
	public double getMargin() {
		return margin;
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
		return MLUtils.svmIntervalToProb(innerProduct(x));
	}

	/**
	 * Builder.
	 * 
	 * @param dimension
	 *            the dimension
	 * @param bias
	 *            the bias
	 * @return the perceptron with margin builder
	 */
	public static PerceptronWithMarginBuilder builder(int dimension,
			boolean bias) {
		return new PerceptronWithMarginBuilder(dimension, bias);
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
		ClassifierEvaluation.evaluate(args, PerceptronWithMargin.class);
	}

}
