/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.examples.pipeline;

import com.dsi.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.LogisticRegressionBuilder;
import com.dsi.parallax.ml.classifier.linear.updateable.LogisticRegression;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.ml.metaoptimize.Grid;
import com.dsi.parallax.ml.metaoptimize.GridMetaClassifierOptimizer;
import com.dsi.parallax.ml.metaoptimize.RandomGridSearch;
import com.dsi.parallax.ml.objective.AUCObjective;
import com.dsi.parallax.ml.util.option.Configuration;
import com.dsi.parallax.pipeline.precompiled.VWtoBinaryInstancesPipeline;

import java.util.Arrays;

/**
 * A simple example where the hyper-parameters of a logistic regression model
 * are tuned using grid search.
 * 
 * @author jattenberg
 */
public class BestModelFinder {

	/**
	 * The main method.
	 * 
	 * @param args
	 *            (none needed)
	 */
	public static void main(String[] args) {

		VWtoBinaryInstancesPipeline pipe = new VWtoBinaryInstancesPipeline(
				"data/science.small.vw", (int) Math.pow(2, 16));
		BinaryClassificationInstances insts = pipe.next();
		Grid<LogisticRegressionBuilder> grid = new RandomGridSearch<LogisticRegressionBuilder>(
				new Configuration<LogisticRegressionBuilder>(
						LogisticRegressionBuilder.getOptions()), new String[] {
						"LR", "f", "g", "T" });
		GridMetaClassifierOptimizer<LogisticRegression, LogisticRegressionBuilder> optimizer = new GridMetaClassifierOptimizer<LogisticRegression, LogisticRegressionBuilder>(
				new AUCObjective(), new LogisticRegressionBuilder(
						(int) Math.pow(2, 16), true), 30, grid);

		Configuration<LogisticRegressionBuilder> optimized = optimizer
				.optimize(insts);

		System.out.println(Arrays.toString(optimized.getArgumentsFromOpts()));
	}
}
