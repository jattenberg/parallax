/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.classifier;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;
import java.util.Iterator;
import java.util.Map;

import org.junit.Test;

import com.dsi.parallax.ml.classifier.bayes.NaiveBayes;
import com.dsi.parallax.ml.classifier.linear.updateable.AROWClassifier;
import com.dsi.parallax.ml.classifier.linear.updateable.LogisticRegression;
import com.dsi.parallax.ml.classifier.linear.updateable.WinnowClassifier;
import com.dsi.parallax.ml.evaluation.OnlineEvaluation;
import com.dsi.parallax.ml.target.BinaryTargetNumericParser;
import com.dsi.parallax.pipeline.Context;
import com.dsi.parallax.pipeline.FileSource;
import com.dsi.parallax.pipeline.Pipeline;
import com.dsi.parallax.pipeline.classifier.SequentialClassifierTrainingPipe;
import com.dsi.parallax.pipeline.csv.NumericCSVtoLabeledVectorPipe;
import com.dsi.parallax.pipeline.file.FileToLinesPipe;
import com.dsi.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;
import com.google.common.collect.Maps;

/**
 * The Class TestSequentialClassifierTrainingPipe.
 */
public class TestSequentialClassifierTrainingPipe {

	/** The file. */
	File file = new File("data/iris.data");

	/** The bins. */
	int bins = 5;

	/**
	 * Test nb train on iris.
	 * 
	 * @throws IOException
	 *             Signals that an I/O exception has occurred.
	 */
	@Test
	public void testNBTrainOnIris() throws IOException {
		Pipeline<File, OnlineEvaluation> pipeline;
		Map<String, String> labelMap = Maps.newHashMap();
		labelMap.put("Iris-setosa", 1 + "");
		labelMap.put("Iris-versicolor", 0 + "");
		labelMap.put("Iris-virginica", 0 + "");
		NaiveBayes model = new NaiveBayes(bins, true);
		pipeline = Pipeline
				.newPipeline(new FileSource(file))
				.addPipe(new FileToLinesPipe())
				.addPipe(new NumericCSVtoLabeledVectorPipe(-1, 4, labelMap))
				.addPipe(
						new BinaryInstancesFromVectorPipe(
								new BinaryTargetNumericParser()))
				.addPipe(
						new SequentialClassifierTrainingPipe<NaiveBayes>(model));
		Iterator<Context<OnlineEvaluation>> out = pipeline.process();
		assertTrue(out.hasNext());

		double tot = 0, ct = 0;

		while (out.hasNext()) {
			assertTrue(out.hasNext());
			double perf = out.next().getData().computeAUC();
			if (!Double.isNaN(perf)) {
				tot += perf;
				ct++;
			}
		}
		assertTrue(tot / ct > 0.5);
	}

	/**
	 * Test lr train on iris.
	 * 
	 * @throws IOException
	 *             Signals that an I/O exception has occurred.
	 */
	@Test
	public void testLRTrainOnIris() throws IOException {
		Pipeline<File, OnlineEvaluation> pipeline;
		Map<String, String> labelMap = Maps.newHashMap();
		labelMap.put("Iris-setosa", 1 + "");
		labelMap.put("Iris-versicolor", 0 + "");
		labelMap.put("Iris-virginica", 0 + "");
		LogisticRegression model = new LogisticRegression(bins, true);
		pipeline = Pipeline
				.newPipeline(new FileSource(file))
				.addPipe(new FileToLinesPipe())
				.addPipe(new NumericCSVtoLabeledVectorPipe(-1, 4, labelMap))
				.addPipe(
						new BinaryInstancesFromVectorPipe(
								new BinaryTargetNumericParser()))
				.addPipe(
						new SequentialClassifierTrainingPipe<LogisticRegression>(
								model));
		Iterator<Context<OnlineEvaluation>> out = pipeline.process();
		assertTrue(out.hasNext());
		double tot = 0, ct = 0;

		while (out.hasNext()) {
			assertTrue(out.hasNext());
			double perf = out.next().getData().computeAUC();
			if (!Double.isNaN(perf)) {
				tot += perf;
				ct++;
			}
		}
		assertTrue(tot / ct > 0.5);
	}

	/**
	 * Test arow train on iris.
	 * 
	 * @throws IOException
	 *             Signals that an I/O exception has occurred.
	 */
	@Test
	public void testAROWTrainOnIris() throws IOException {
		Pipeline<File, OnlineEvaluation> pipeline;
		Map<String, String> labelMap = Maps.newHashMap();
		labelMap.put("Iris-setosa", 1 + "");
		labelMap.put("Iris-versicolor", 0 + "");
		labelMap.put("Iris-virginica", 0 + "");
		AROWClassifier model = new AROWClassifier(bins, true);
		pipeline = Pipeline
				.newPipeline(new FileSource(file))
				.addPipe(new FileToLinesPipe())
				.addPipe(new NumericCSVtoLabeledVectorPipe(-1, 4, labelMap))
				.addPipe(
						new BinaryInstancesFromVectorPipe(
								new BinaryTargetNumericParser()))
				.addPipe(
						new SequentialClassifierTrainingPipe<AROWClassifier>(
								model));
		Iterator<Context<OnlineEvaluation>> out = pipeline.process();
		assertTrue(out.hasNext());
		double tot = 0, ct = 0;

		while (out.hasNext()) {
			assertTrue(out.hasNext());
			double perf = out.next().getData().computeAUC();
			if (!Double.isNaN(perf)) {
				tot += perf;
				ct++;
			}
		}
		assertTrue(tot / ct > 0.5);
	}

	/**
	 * Test winnow.
	 * 
	 * @throws IOException
	 *             Signals that an I/O exception has occurred.
	 */
	@Test
	public void testWinnow() throws IOException {
		Pipeline<File, OnlineEvaluation> pipeline;
		Map<String, String> labelMap = Maps.newHashMap();
		labelMap.put("Iris-setosa", 1 + "");
		labelMap.put("Iris-versicolor", 0 + "");
		labelMap.put("Iris-virginica", 0 + "");
		WinnowClassifier model = new WinnowClassifier(bins, true);
		pipeline = Pipeline
				.newPipeline(new FileSource(file))
				.addPipe(new FileToLinesPipe())
				.addPipe(new NumericCSVtoLabeledVectorPipe(-1, 4, labelMap))
				.addPipe(
						new BinaryInstancesFromVectorPipe(
								new BinaryTargetNumericParser()))
				.addPipe(
						new SequentialClassifierTrainingPipe<WinnowClassifier>(
								model));
		Iterator<Context<OnlineEvaluation>> out = pipeline.process();
		assertTrue(out.hasNext());
		double tot = 0, ct = 0;

		while (out.hasNext()) {
			assertTrue(out.hasNext());
			double perf = out.next().getData().computeAUC();
			if (!Double.isNaN(perf)) {
				tot += perf;
				ct++;
			}
		}
		assertTrue(tot / ct > 0.5);
	}
}
