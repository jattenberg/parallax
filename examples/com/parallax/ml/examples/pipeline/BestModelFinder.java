/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.examples.pipeline;

import java.util.Arrays;

import com.parallax.ml.classifier.linear.updateable.LogisticRegression;
import com.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.LogisticRegressionBuilder;
import com.parallax.ml.instance.BinaryClassificationInstances;
import com.parallax.ml.metaoptimize.Grid;
import com.parallax.ml.metaoptimize.GridMetaClassifierOptimizer;
import com.parallax.ml.metaoptimize.RandomGridSearch;
import com.parallax.ml.objective.AUCObjective;
import com.parallax.ml.util.option.Configuration;
import com.parallax.pipeline.precompiled.VWtoBinaryInstancesPipeline;

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
