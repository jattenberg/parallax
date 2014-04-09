/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.metaoptimize;

import static com.google.common.base.Preconditions.checkArgument;

import com.dsi.parallax.ml.classifier.Classifier;
import com.dsi.parallax.ml.classifier.ClassifierBuilder;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.ml.objective.FoldEvaluator;
import com.dsi.parallax.ml.objective.MeanScorer;
import com.dsi.parallax.ml.objective.Objective;
import com.dsi.parallax.ml.objective.ObjectiveScorer;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.util.option.Configuration;

// TODO: Auto-generated Javadoc
/**
 * class for optimizing the configuration of a particular classifier classifier
 * should be reflected by an entry in the Classfiers enum.
 * 
 * @param <C>
 *            the generic type
 * @param <B>
 *            the generic type
 * @author jattenberg
 */
public class GridMetaClassifierOptimizer<C extends Classifier<C>, B extends ClassifierBuilder<C, B>> {

	/** The scorer. */
	private final ObjectiveScorer<BinaryClassificationTarget> scorer;

	/** The builder. */
	private final B builder;

	/** The iterations. */
	private final int iterations;

	/** The grid. */
	private final Grid<B> grid;

	/** The Constant DEFAULT_FOLDS. */
	private static final int DEFAULT_FOLDS = 5;

	/** The folds. */
	private final int folds;

	/** The best config. */
	private Configuration<B> bestConfig;

	/** The best objective. */
	private double bestObjective;

	/**
	 * Instantiates a new grid meta classifier optimizer.
	 * 
	 * @param objective
	 *            the objective
	 * @param builder
	 *            the builder
	 * @param iterations
	 *            the iterations
	 * @param grid
	 *            the grid
	 */
	public GridMetaClassifierOptimizer(
			Objective<BinaryClassificationTarget> objective, B builder,
			int iterations, Grid<B> grid) {
		this(objective, builder, iterations, grid, DEFAULT_FOLDS);
	}

	/**
	 * Instantiates a new grid meta classifier optimizer.
	 * 
	 * @param objective
	 *            the objective
	 * @param builder
	 *            the builder
	 * @param iterations
	 *            the iterations
	 * @param grid
	 *            the grid
	 * @param folds
	 *            the folds
	 */
	public GridMetaClassifierOptimizer(
			Objective<BinaryClassificationTarget> objective, B builder,
			int iterations, Grid<B> grid, int folds) {
		this(new MeanScorer<BinaryClassificationTarget>(objective), builder,
				iterations, grid, folds);
	}

	/**
	 * Instantiates a new grid meta classifier optimizer.
	 * 
	 * @param scorer
	 *            the scorer
	 * @param builder
	 *            the builder
	 * @param iterations
	 *            the iterations
	 * @param grid
	 *            the grid
	 * @param folds
	 *            the folds
	 */
	public GridMetaClassifierOptimizer(
			ObjectiveScorer<BinaryClassificationTarget> scorer, B builder,
			int iterations, Grid<B> grid, int folds) {
		this.scorer = scorer;
		this.builder = builder;
		checkArgument(iterations > 0, "iterations must be > 0, given %s",
				iterations);
		this.iterations = iterations;
		this.grid = grid;
		this.folds = folds;
	}

	/**
	 * Instantiates a new grid meta classifier optimizer.
	 * 
	 * @param scorer
	 *            the scorer
	 * @param builder
	 *            the builder
	 * @param iterations
	 *            the iterations
	 * @param grid
	 *            the grid
	 */
	public GridMetaClassifierOptimizer(
			ObjectiveScorer<BinaryClassificationTarget> scorer, B builder,
			int iterations, Grid<B> grid) {
		this(scorer, builder, iterations, grid, DEFAULT_FOLDS);
	}

	/**
	 * Optimize.
	 * 
	 * @param instances
	 *            the instances
	 * @return the configuration
	 */
	public Configuration<B> optimize(BinaryClassificationInstances instances) {
		bestObjective = Double.MIN_VALUE;
		bestConfig = null;
		for (int i = 0; i < iterations; i++) {
			FoldEvaluator evaluator = new FoldEvaluator(folds);
			Configuration<B> config = grid.next();

			builder.configure(config);

			double currentObjective = evaluator.evaluate(instances, scorer,
					builder);

			if (currentObjective > bestObjective) {
				bestObjective = currentObjective;
				bestConfig = config;
			}
		}

		return bestConfig;
	}
}
