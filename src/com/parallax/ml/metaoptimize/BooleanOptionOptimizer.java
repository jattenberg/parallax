package com.parallax.ml.metaoptimize;

import com.parallax.ml.classifier.Classifier;
import com.parallax.ml.classifier.ClassifierBuilder;
import com.parallax.ml.instance.BinaryClassificationInstances;
import com.parallax.ml.metaoptimize.OptionOptimizer.AbstractBooleanOptionOptimizer;
import com.parallax.ml.objective.FoldEvaluator;
import com.parallax.ml.objective.ObjectiveScorer;
import com.parallax.ml.target.BinaryClassificationTarget;
import com.parallax.ml.util.option.BooleanOption;
import com.parallax.ml.util.option.Configuration;

// TODO: option specific optimization is still baking
/**
 * The Class BooleanOptionOptimizer.
 *
 * @param <C> the generic type
 * @param <B> the generic type
 */
public class BooleanOptionOptimizer<C extends Classifier<C>, B extends ClassifierBuilder<C, B>>
		extends AbstractBooleanOptionOptimizer<C, B> {

	/** The best value. */
	private boolean bestValue = false;
	
	/** The best objective. */
	private double bestObjective = Double.MIN_VALUE;

	/**
	 * Instantiates a new boolean option optimizer.
	 *
	 * @param scorer the scorer
	 * @param builder the builder
	 * @param instances the instances
	 * @param opt the opt
	 * @param folds the folds
	 */
	public BooleanOptionOptimizer(
			ObjectiveScorer<BinaryClassificationTarget> scorer, B builder,
			BinaryClassificationInstances instances, BooleanOption opt,
			int folds) {
		super(scorer, builder, instances, opt, folds);
	}

	/**
	 * Instantiates a new boolean option optimizer.
	 *
	 * @param scorer the scorer
	 * @param builder the builder
	 * @param instances the instances
	 * @param opt the opt
	 */
	public BooleanOptionOptimizer(
			ObjectiveScorer<BinaryClassificationTarget> scorer, B builder,
			BinaryClassificationInstances instances, BooleanOption opt) {
		super(scorer, builder, instances, opt);
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.metaoptimize.OptionOptimizer#reset()
	 */
	@Override
	public void reset() {
		bestValue = false;
		bestObjective = Double.MIN_VALUE;

	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.metaoptimize.OptionOptimizer.AbstractBooleanOptionOptimizer#optimize()
	 */
	@Override
	public Boolean optimize() {
		Configuration<B> conf = builder.getConfiguration();
		FoldEvaluator evaluator = new FoldEvaluator(folds);

		for (Boolean bool : new Boolean[] { true, false }) {

			conf.addBooleanValueOnShortName(opt.getShortName(), bool);
			builder.configure(conf);

			double objective;
			objective = evaluator.evaluate(instances, scorer, builder);
			if (objective > bestObjective) {
				bestObjective = objective;
				bestValue = bool;
			}
		}
		return bestValue;
	}

}
