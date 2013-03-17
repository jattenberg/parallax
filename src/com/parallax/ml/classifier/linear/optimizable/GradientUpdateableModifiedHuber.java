package com.parallax.ml.classifier.linear.optimizable;

import com.parallax.ml.instance.BinaryClassificationInstance;
import com.parallax.ml.instance.Instance;
import com.parallax.ml.util.MLUtils;
import com.parallax.ml.vector.LinearVector;
import com.parallax.ml.vector.LinearVectorFactory;
import com.parallax.ml.vector.util.VectorUtils;
import com.parallax.optimization.Gradient;
import com.parallax.optimization.stochastic.StochasticGradientOptimizationBuilder;

public class GradientUpdateableModifiedHuber extends
		AbstractGradientUpdateableClassifier<GradientUpdateableModifiedHuber> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 6137094313834585779L;

	/**
	 * Instantiates a new gradient updateable Modified Huber model.
	 * 
	 * @param builder
	 *            used to construct the optimization procedure used for model
	 *            training
	 * @param dimension
	 *            the number of features in the instantiated classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 */
	public GradientUpdateableModifiedHuber(
			StochasticGradientOptimizationBuilder<?> builder, int dimension,
			boolean bias) {
		super(builder, dimension, bias);
		initialize();
	}

	/**
	 * Instantiates a new gradient updateable Modified Huber model.
	 * 
	 * @param dimension
	 *            the number of features in the instantiated classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 */
	public GradientUpdateableModifiedHuber(int dimension, boolean bias) {
		super(dimension, bias);
		initialize();
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
		return VectorUtils.dotProduct(parameters, inst);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.optimizable.AbstractGradientUpdateableClassifier
	 * #initialize()
	 */
	@Override
	public GradientUpdateableModifiedHuber initialize() {
		super.initialize();
		return model;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.AbstractModel#getModel()
	 */
	@Override
	protected GradientUpdateableModifiedHuber getModel() {
		return this;
	}

	@Override
	public Gradient computeGradient(LinearVector params) {
		double denominator = instances.size();
		double loss = 0d;

		LinearVector gradVector = LinearVectorFactory.getVector(dimension);

		for (BinaryClassificationInstance inst : instances) {
			double iLabel = MLUtils.probToSVMInterval(inst.getLabel()
					.getValue());
			double yp = VectorUtils.dotProduct(params, inst)
					* MLUtils.probToSVMInterval(inst.getLabel().getValue());
			double lossPart;
			if (yp >= -1d) {
				lossPart = Math.pow(Math.max(0, 1 - yp), 2d);
				if (1 - yp > 0) {
					for (int x_i : inst) {
						double update = (1 - yp) * inst.getFeatureValue(x_i)
								* iLabel;
						gradVector.updateValue(x_i, -update);
					}
				}
			} else {
				lossPart = -4 * yp;
				for (int x_i : inst) {
					double update = -4. * inst.getFeatureValue(x_i) * iLabel;
					gradVector.updateValue(x_i, -update);
				}
			}
			loss += lossPart / denominator;

		}
		return new Gradient(gradVector, loss);
	}

	@Override
	public double computeLoss(LinearVector params) {
		double denominator = instances.size();
		double loss = 0d;
		for (BinaryClassificationInstance inst : instances) {
			double iLoss = modifiedHuberLoss(inst, params);
			loss += iLoss;
		}
		return loss / denominator;
	}

	private double modifiedHuberLoss(BinaryClassificationInstance inst,
			LinearVector params) {
		double yp = VectorUtils.dotProduct(params, inst)
				* MLUtils.probToSVMInterval(inst.getLabel().getValue());
		if (yp >= -1d) {
			return Math.pow(Math.max(0, 1 - yp), 2d);
		} else {
			return -4 * yp;
		}
	}
}
