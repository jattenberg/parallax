package com.parallax.ml.classifier.smoother;

import static com.google.common.base.Preconditions.checkArgument;

import java.util.Collection;

import com.parallax.ml.classifier.linear.optimizable.GradientUpdateableLogisticRegression;
import com.parallax.ml.instance.BinaryClassificationInstance;
import com.parallax.ml.instance.BinaryClassificationInstances;
import com.parallax.ml.target.BinaryClassificationTarget;
import com.parallax.ml.util.pair.PrimitivePair;
import com.parallax.optimization.stochastic.StochasticLBFGSBuilder;
import com.parallax.optimization.stochastic.anneal.AnnealingScheduleConfigurableBuilder;

/**
 * The Class LogisticRegressionSmoother.
 */
public class LogisticRegressionSmoother extends
		AbstractSmoother<LogisticRegressionSmoother> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -5229213936451815755L;

	/** The lr. model backing the smoother */
	private final GradientUpdateableLogisticRegression lr;

	private int passes = 10;
	/**
	 * Instantiates a new logistic regression smoother.
	 */
	public LogisticRegressionSmoother() {
		StochasticLBFGSBuilder lbfgsBuilder = new StochasticLBFGSBuilder(1,
				true);
		lbfgsBuilder.setAnnealingScheduleConfigurableBuilder(AnnealingScheduleConfigurableBuilder.configureForConstantRate(0.01));
		lr = new GradientUpdateableLogisticRegression(lbfgsBuilder, 1, true);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.smoother.Smoother#train(java.util.Collection
	 * )
	 */
	@Override
	public void train(Collection<PrimitivePair> input) {
		lr.setPasses(passes);
		BinaryClassificationInstances insts = new BinaryClassificationInstances(
				1);
		for (PrimitivePair pair : input) {
			BinaryClassificationInstance inst = new BinaryClassificationInstance(
					1);
			inst.addFeature(0, pair.getFirst());
			inst.setLabel(new BinaryClassificationTarget(pair.getSecond()));
			insts.addInstance(inst);
		}
		lr.train(insts);

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.smoother.Smoother#smooth(double)
	 */
	@Override
	public double smooth(double prediction) {
		BinaryClassificationInstance inst = new BinaryClassificationInstance(1);
		inst.addFeature(0, prediction);
		return lr.predict(inst).getValue();
	}

	
	public LogisticRegressionSmoother setPasses(int passes) {
		checkArgument(passes > 0, "passes must be positive, given: %s", passes);
		this.passes = passes;
		return this;
	}
}
