/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.examples.numeric;

import java.io.File;
import java.util.Iterator;
import java.util.Map;

import com.google.common.collect.Maps;
import com.parallax.ml.classifier.Classifiers;
import com.parallax.ml.classifier.UpdateableClassifier;
import com.parallax.ml.classifier.linear.updateable.AROWClassifier;
import com.parallax.ml.classifier.linear.updateable.LogisticRegression;
import com.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.LogisticRegressionBuilder;
import com.parallax.ml.evaluation.OnlineEvaluation;
import com.parallax.ml.target.BinaryTargetNumericParser;
import com.parallax.optimization.regularization.TruncationConfigurableBuilder;
import com.parallax.optimization.regularization.TruncationType;
import com.parallax.pipeline.Context;
import com.parallax.pipeline.FileSource;
import com.parallax.pipeline.Pipeline;
import com.parallax.pipeline.classifier.SequentialClassifierTrainingPipe;
import com.parallax.pipeline.csv.NumericCSVtoLabeledVectorPipe;
import com.parallax.pipeline.file.FileToLinesPipe;
import com.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;
import com.parallax.pipeline.text.RegexStringFilterPipe;

/**
 * An example for reading numeric data stored in a CSV and sequentially updating
 * a classifier using the resultant data
 */
public class CSVPipeline {

	/**
	 * The main method.
	 * 
	 * @param args
	 *            the arguments
	 */
	@SuppressWarnings("unchecked")
	public static void main(String[] args) {
		String file = "data/ad.data";
		int dimension = 1558;

		Pipeline<File, OnlineEvaluation> pipeline;

		Map<String, String> labelMap = Maps.newHashMap();
		labelMap.put("ad.", "" + 1);
		labelMap.put("nonad.", "" + 0);

		pipeline = Pipeline
				.newPipeline(new FileSource(file))
				.addPipe(new FileToLinesPipe())
				.addPipe(new RegexStringFilterPipe("\\?"))
				.addPipe(
						new NumericCSVtoLabeledVectorPipe(" *, *", dimension,
								labelMap))
				.addPipe(
						new BinaryInstancesFromVectorPipe(
								new BinaryTargetNumericParser()))
				.addPipe(
						new SequentialClassifierTrainingPipe<AROWClassifier>(
								(UpdateableClassifier<AROWClassifier>) Classifiers.AROW
										.getClassifier(dimension, true)));
		Iterator<Context<OnlineEvaluation>> it = pipeline.process();
		OnlineEvaluation eval = it.next().getData();
		while (it.hasNext())
			eval = it.next().getData();
		System.out.println(eval.computeAUC() + " " + eval.computeAccuracy()
				+ " " + eval.computeBrierScore());

		LogisticRegressionBuilder builder = new LogisticRegressionBuilder(
				dimension, true);
		builder.setGaussianWeight(2.)
				.setLaplaceWeight(1.)
				.setTruncationBuilder(
						new TruncationConfigurableBuilder()
								.setTruncationType(TruncationType.TRUNCATING));

		LogisticRegression model = builder.build();

		pipeline = Pipeline
				.newPipeline(new FileSource(file))
				.addPipe(new FileToLinesPipe())
				.addPipe(new RegexStringFilterPipe("\\?"))
				.addPipe(
						new NumericCSVtoLabeledVectorPipe(" *, *", dimension,
								labelMap))
				.addPipe(
						new BinaryInstancesFromVectorPipe(
								new BinaryTargetNumericParser()))
				.addPipe(
						new SequentialClassifierTrainingPipe<LogisticRegression>(
								model));
		it = pipeline.process();

		eval = it.next().getData();
		while (it.hasNext())
			eval = it.next().getData();
		System.out.println(eval.computeAUC() + " " + eval.computeAccuracy()
				+ " " + eval.computeBrierScore());

	}

}
