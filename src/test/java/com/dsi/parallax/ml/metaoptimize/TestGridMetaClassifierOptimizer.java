/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.metaoptimize;

import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.dsi.parallax.ml.classifier.Classifiers;
import com.dsi.parallax.ml.classifier.bayes.NaiveBayes;
import com.dsi.parallax.ml.classifier.bayes.NaiveBayesBuilder;
import com.dsi.parallax.ml.classifier.linear.updateable.LogisticRegression;
import com.dsi.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.LogisticRegressionBuilder;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.ml.metaoptimize.GridMetaClassifierOptimizer;
import com.dsi.parallax.ml.metaoptimize.RandomGridSearch;
import com.dsi.parallax.ml.metaoptimize.SequentialGridSearch;
import com.dsi.parallax.ml.objective.AUCObjective;
import com.dsi.parallax.ml.objective.FoldEvaluator;
import com.dsi.parallax.ml.objective.MeanScorer;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.util.option.Configuration;
import com.dsi.parallax.pipeline.precompiled.VWtoBinaryInstancesPipeline;
import com.google.common.collect.Sets;

/**
 * The Class TestGridMetaClassifierOptimizer.
 */
public class TestGridMetaClassifierOptimizer {

	/** The dim. */
	int dim = (int) Math.pow(2, 14);

	/**
	 * Test random grid search.
	 */
	@SuppressWarnings("unchecked")
	@Test
	public void testRandomGridSearch() {
		LogisticRegressionBuilder LRBuilder = new LogisticRegressionBuilder(
				dim, true);
		Configuration<LogisticRegressionBuilder> defConfig = LRBuilder
				.getConfiguration();
		GridMetaClassifierOptimizer<LogisticRegression, LogisticRegressionBuilder> optimizer = new GridMetaClassifierOptimizer<LogisticRegression, LogisticRegressionBuilder>(
				new MeanScorer<BinaryClassificationTarget>(new AUCObjective()),
				LRBuilder,
				200,
				new RandomGridSearch<LogisticRegressionBuilder>(
						(Configuration<LogisticRegressionBuilder>) Classifiers.LOGISTICREGRESSION
								.getConfiguration(), new String[] { "GR", "f",
								"u" }), 5);
		VWtoBinaryInstancesPipeline pipe = new VWtoBinaryInstancesPipeline(
				"data/science.small.vw", dim);
		assertTrue(pipe.hasNext());
		BinaryClassificationInstances insts = pipe.next();

		Configuration<LogisticRegressionBuilder> conf = optimizer
				.optimize(insts);
		assertTrue(null != conf);
		assertTrue(!Sets.newHashSet(conf.getArgumentsFromOpts()).equals(
				Sets.newHashSet(Classifiers.LOGISTICREGRESSION
						.getConfiguration().getArgumentsFromOpts())));
		// System.out.println(Arrays.toString(conf.getArgumentsFromOpts()));
		FoldEvaluator evaluator = new FoldEvaluator();

		LRBuilder.configure(defConfig);
		double origPerf = evaluator.evaluate(insts,
				new MeanScorer<BinaryClassificationTarget>(new AUCObjective()),
				LRBuilder);
		LRBuilder.configure(conf);
		double tunedPerf = evaluator.evaluate(insts,
				new MeanScorer<BinaryClassificationTarget>(new AUCObjective()),
				LRBuilder);
		// System.out.println("tuned: " + tunedPerf + " untuned: " + origPerf);
		assertTrue(tunedPerf > origPerf);
	}

	/**
	 * Test sequential grid search.
	 */
	@SuppressWarnings("unchecked")
	@Test
	public void testSequentialGridSearch() {
		LogisticRegressionBuilder LRBuilder = new LogisticRegressionBuilder(
				dim, true);
		Configuration<LogisticRegressionBuilder> defConfig = LRBuilder
				.getConfiguration();
		GridMetaClassifierOptimizer<LogisticRegression, LogisticRegressionBuilder> optimizer = new GridMetaClassifierOptimizer<LogisticRegression, LogisticRegressionBuilder>(
				new MeanScorer<BinaryClassificationTarget>(new AUCObjective()),
				LRBuilder,
				100,
				new SequentialGridSearch<LogisticRegressionBuilder>(
						(Configuration<LogisticRegressionBuilder>) Classifiers.LOGISTICREGRESSION
								.getConfiguration(), new String[] { "GR", "f",
								"u" }, 50), 5);
		VWtoBinaryInstancesPipeline pipe = new VWtoBinaryInstancesPipeline(
				"data/science.small.vw", dim);
		assertTrue(pipe.hasNext());
		BinaryClassificationInstances insts = pipe.next();

		Configuration<LogisticRegressionBuilder> conf = optimizer
				.optimize(insts);
		assertTrue(null != conf);
		assertTrue(!Sets.newHashSet(conf.getArgumentsFromOpts()).equals(
				Sets.newHashSet(Classifiers.LOGISTICREGRESSION
						.getConfiguration().getArgumentsFromOpts())));
		// System.out.println(Arrays.toString(conf.getArgumentsFromOpts()));
		FoldEvaluator evaluator = new FoldEvaluator();

		LRBuilder.configure(defConfig);
		double origPerf = evaluator.evaluate(insts,
				new MeanScorer<BinaryClassificationTarget>(new AUCObjective()),
				LRBuilder);
		LRBuilder.configure(conf);
		double tunedPerf = evaluator.evaluate(insts,
				new MeanScorer<BinaryClassificationTarget>(new AUCObjective()),
				LRBuilder);
		// System.out.println("tuned: " + tunedPerf + " untuned: " + origPerf);
		assertTrue(tunedPerf > origPerf);
	}

	/**
	 * Test optimizes nested configs.
	 */
	@Test
	public void testOptimizesNestedConfigs() {

		NaiveBayesBuilder nbb = new NaiveBayesBuilder(dim, false);
		Configuration<NaiveBayesBuilder> defConfig = nbb.getConfiguration();

		@SuppressWarnings("unchecked")
		GridMetaClassifierOptimizer<NaiveBayes, NaiveBayesBuilder> optimizer = new GridMetaClassifierOptimizer<NaiveBayes, NaiveBayesBuilder>(
				new MeanScorer<BinaryClassificationTarget>(new AUCObjective()),
				nbb,
				200,
				new SequentialGridSearch<NaiveBayesBuilder>(
						(Configuration<NaiveBayesBuilder>) Classifiers.NB
								.getConfiguration(), new String[] { "l", "n" }, 10),
				1);

		VWtoBinaryInstancesPipeline pipe = new VWtoBinaryInstancesPipeline(
				"data/science.small.vw", dim);
		assertTrue(pipe.hasNext());
		BinaryClassificationInstances insts = pipe.next();

		Configuration<NaiveBayesBuilder> conf = optimizer.optimize(insts);
		assertTrue(null != conf);
		assertTrue(!Sets.newHashSet(conf.getArgumentsFromOpts()).equals(
				Sets.newHashSet(Classifiers.NB.getConfiguration()
						.getArgumentsFromOpts())));

		FoldEvaluator evaluator = new FoldEvaluator();

		nbb.configure(defConfig);
		double origPerf = evaluator.evaluate(insts,
				new MeanScorer<BinaryClassificationTarget>(new AUCObjective()),
				nbb);
		nbb.configure(conf);
		double tunedPerf = evaluator.evaluate(insts,
				new MeanScorer<BinaryClassificationTarget>(new AUCObjective()),
				nbb);
		assertTrue(tunedPerf > origPerf);
	}

	/**
	 * Test optimizes nested configs using several simultanious threads
	 */
	@Test
	public void testOptimizesNestedConfigsMultiThread() {

		NaiveBayesBuilder nbb = new NaiveBayesBuilder(dim, false);
		Configuration<NaiveBayesBuilder> defConfig = nbb.getConfiguration();

		@SuppressWarnings("unchecked")
		GridMetaClassifierOptimizer<NaiveBayes, NaiveBayesBuilder> optimizer = new GridMetaClassifierOptimizer<NaiveBayes, NaiveBayesBuilder>(
				new MeanScorer<BinaryClassificationTarget>(new AUCObjective()),
				nbb,
				200,
				new RandomGridSearch<NaiveBayesBuilder>(
						(Configuration<NaiveBayesBuilder>) Classifiers.NB
								.getConfiguration(), new String[] { "l", "n" }),
				5);

		VWtoBinaryInstancesPipeline pipe = new VWtoBinaryInstancesPipeline(
				"data/science.small.vw", dim);
		assertTrue(pipe.hasNext());
		BinaryClassificationInstances insts = pipe.next();

		Configuration<NaiveBayesBuilder> conf = optimizer.optimize(insts);
		assertTrue(null != conf);
		assertTrue(!Sets.newHashSet(conf.getArgumentsFromOpts()).equals(
				Sets.newHashSet(Classifiers.NB.getConfiguration()
						.getArgumentsFromOpts())));
		// System.out.println(Arrays.toString(conf.getArgumentsFromOpts()));
		FoldEvaluator evaluator = new FoldEvaluator();

		nbb.configure(defConfig);
		double origPerf = evaluator.evaluate(insts,
				new MeanScorer<BinaryClassificationTarget>(new AUCObjective()),
				nbb);
		nbb.configure(conf);
		double tunedPerf = evaluator.evaluate(insts,
				new MeanScorer<BinaryClassificationTarget>(new AUCObjective()),
				nbb);
		// System.out.println("tuned: " + tunedPerf + " untuned: " + origPerf);
		assertTrue(tunedPerf > origPerf);
	}
}
