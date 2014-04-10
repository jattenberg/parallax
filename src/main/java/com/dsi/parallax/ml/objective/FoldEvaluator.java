/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.objective;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.apache.log4j.Logger;

import com.dsi.parallax.ml.classifier.Classifier;
import com.dsi.parallax.ml.classifier.ClassifierBuilder;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;

// TODO: Auto-generated Javadoc
/**
 * cross-validates the performance of a classifier on some specified data using
 * a given objective value.
 *
 * @author jattenberg
 */
public class FoldEvaluator {
	
	/** The Constant LOGGER. */
	private static final Logger LOGGER = Logger.getLogger(FoldEvaluator.class);
	
	/** The executor. */
	private final ExecutorService executor;
	
	/** The Constant DEFAULT_FOLDS. */
	private static final int DEFAULT_FOLDS = 5;
	
	/** The folds. */
	private int folds;

	/**
	 * Instantiates a new fold evaluator.
	 */
	public FoldEvaluator() {
		this(DEFAULT_FOLDS);
	}

	/**
	 * Instantiates a new fold evaluator.
	 *
	 * @param folds the folds
	 */
	public FoldEvaluator(int folds) {
		this.folds = folds;
		executor = Executors.newFixedThreadPool(Math.min(DEFAULT_FOLDS, folds));
	}

	/**
	 * Evaluate.
	 *
	 * @param inputData the input data
	 * @param objective the objective
	 * @param builder the builder
	 * @return the double
	 */
	public double evaluate(BinaryClassificationInstances inputData,
			Objective<BinaryClassificationTarget> objective,
			ClassifierBuilder<?, ?> builder) {
		return evaluate(inputData, new MeanScorer<BinaryClassificationTarget>(
				objective), builder);
	}

	/**
	 * Evaluate.
	 *
	 * @param inputData the input data
	 * @param scorer the scorer
	 * @param builder the builder
	 * @return the double
	 */
	public double evaluate(BinaryClassificationInstances inputData,
			ObjectiveScorer<BinaryClassificationTarget> scorer,
			ClassifierBuilder<?, ?> builder) {
		scorer.reset();
		List<Callable<Object>> todo = new ArrayList<Callable<Object>>(folds);

		for (int fold = 0; fold < folds; fold++) {
			BinaryClassificationInstances training = inputData.getTraining(
					fold, folds);
			BinaryClassificationInstances testing = inputData.getTesting(fold,
					folds);
			FoldRunnable runnable = new FoldRunnable(training, testing, scorer,
					builder);
			todo.add(Executors.callable(runnable));
		}
		try {
			executor.invokeAll(todo);
		} catch (InterruptedException e) {
			LOGGER.error("error executing threads: " + e.getLocalizedMessage());
			throw new RuntimeException(e);
		}
		return scorer.getScore();
	}

	/**
	 * The Class FoldRunnable.
	 */
	private class FoldRunnable implements Runnable {

		/** The testing. */
		final BinaryClassificationInstances training, testing;
		
		/** The scorer. */
		final ObjectiveScorer<BinaryClassificationTarget> scorer;
		
		/** The builder. */
		final ClassifierBuilder<?, ?> builder;

		/**
		 * Instantiates a new fold runnable.
		 *
		 * @param training the training
		 * @param testing the testing
		 * @param scorer the scorer
		 * @param builder the builder
		 */
		public FoldRunnable(BinaryClassificationInstances training,
				BinaryClassificationInstances testing,
				ObjectiveScorer<BinaryClassificationTarget> scorer,
				ClassifierBuilder<?, ?> builder) {
			super();
			this.training = training;
			this.testing = testing;
			this.builder = builder;
			this.scorer = scorer;
		}

		/* (non-Javadoc)
		 * @see java.lang.Runnable#run()
		 */
		@SuppressWarnings({ "rawtypes", "unchecked" })
		@Override
		public void run() {
			Classifier model = builder.build();
			model.train(training);
			synchronized (this) {
				scorer.evaluate(testing, model);
			}
		}
	}
}
