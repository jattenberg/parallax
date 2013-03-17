package com.parallax.ml.classifier.linear.optimizable;

import com.parallax.ml.instance.BinaryClassificationInstance;
import com.parallax.ml.instance.Instance;
import com.parallax.ml.util.MLUtils;
import com.parallax.ml.util.SigmoidType;
import com.parallax.ml.vector.LinearVector;
import com.parallax.ml.vector.LinearVectorFactory;
import com.parallax.ml.vector.util.VectorUtils;
import com.parallax.optimization.Gradient;
import com.parallax.optimization.stochastic.StochasticGradientOptimizationBuilder;

/**
 * The Class GradientUpdateableLogisticRegression.
 */
public class GradientUpdateableLogisticRegression
		extends
		AbstractGradientUpdateableClassifier<GradientUpdateableLogisticRegression> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -5788153484326435791L;

	public GradientUpdateableLogisticRegression(int dimension, boolean bias) {
		super(dimension, bias);
		initialize();
	}

	public GradientUpdateableLogisticRegression(
			StochasticGradientOptimizationBuilder<?> builder, int dimension,
			boolean bias) {
		super(builder, dimension, bias);
		initialize();
	}

	@Override
	public Gradient computeGradient(LinearVector params) {
		double denominator = instances.size();
		double loss = 0d;

		LinearVector gradVector = LinearVectorFactory.getVector(dimension);

		for (BinaryClassificationInstance inst : instances) {
			double prediction = regress(inst);
			double label = inst.getLabel().getValue();
			double py = MLUtils.probToSVMInterval(prediction)
					* MLUtils.probToSVMInterval(label);
			loss += Math.log(1 + Math.exp(py)) / denominator;

			for (int x_i : inst) {
				double update = (label - prediction)
						* inst.getFeatureValue(x_i);
				// / (1 + Math.exp(py));
				gradVector.updateValue(x_i, update / denominator);
			}
		}
		Gradient grad = new Gradient(gradVector, loss);
		return grad;
	}

	@Override
	public double computeLoss(LinearVector params) {
		double loss = 0d;
		for (BinaryClassificationInstance inst : instances) {
			double prediction = regress(inst);
			double label = MLUtils
					.probToSVMInterval(inst.getLabel().getValue());
			double py = MLUtils.probToSVMInterval(prediction) * label;
			loss += Math.log(1 + Math.exp(py));
		}
		return instances.size() > 0 ? loss / (double) instances.size() : 0;
	}

	@Override
	protected double regress(Instance<?> inst) {
		double dot = VectorUtils.dotProduct(parameters, inst);
		return SigmoidType.LOGIT.sigmoid(-dot);
	}

	@Override
	protected GradientUpdateableLogisticRegression getModel() {
		return this;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.optimizable.AbstractGradientUpdateableClassifier
	 * #initialize()
	 */
	@Override
	public GradientUpdateableLogisticRegression initialize() {
		super.initialize();
		return model;
	}

}
