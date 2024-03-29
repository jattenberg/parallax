/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.metaoptimize;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.util.Map;
import java.util.Set;

import org.junit.Test;

import com.dsi.parallax.ml.classifier.Classifiers;
import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.ml.metaoptimize.ClassifierSelection;
import com.dsi.parallax.ml.objective.AccuracyObjective;
import com.dsi.parallax.ml.objective.FoldEvaluator;
import com.dsi.parallax.ml.objective.GeometricMeanScorer;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.target.BinaryTargetNumericParser;
import com.dsi.parallax.pipeline.FileSource;
import com.dsi.parallax.pipeline.Pipeline;
import com.dsi.parallax.pipeline.csv.NumericCSVtoLabeledVectorPipe;
import com.dsi.parallax.pipeline.file.FileToLinesPipe;
import com.dsi.parallax.pipeline.instance.BinaryClassificationInstancesSink;
import com.dsi.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;
import com.dsi.parallax.pipeline.instance.IntroduceLabelNoisePipe;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;

/**
 * The Class TestClassifierSelection.
 */
public class TestClassifierSelection {

	/** The file. */
	File file = new File("data/iris.data");

	/** The bins. */
	int bins = 5;

	/**
	 * Test optimizes.
	 */
	@Test
	public void testOptimizes() {
		Pipeline<File, BinaryClassificationInstance> pipeline;
		Map<String, String> labelMap = Maps.newHashMap();
		labelMap.put("Iris-setosa", 1 + "");
		labelMap.put("Iris-versicolor", 0 + "");
		labelMap.put("Iris-virginica", 0 + "");

		pipeline = Pipeline
				.newPipeline(new FileSource(file))
				.addPipe(new FileToLinesPipe())
				.addPipe(new NumericCSVtoLabeledVectorPipe(-1, 4, labelMap))
				.addPipe(
						new BinaryInstancesFromVectorPipe(
								new BinaryTargetNumericParser()))
				.addPipe(new IntroduceLabelNoisePipe(0.05));
		BinaryClassificationInstancesSink sink = new BinaryClassificationInstancesSink();
		sink.setSource(pipeline);

		assertTrue(sink.hasNext());
		BinaryClassificationInstances insts = sink.next();
		Set<Classifiers> clSet = Sets.newHashSet(Classifiers.AROW,
				Classifiers.PASSIVEAGGRESSIVE);

		ClassifierSelection selector = new ClassifierSelection(clSet);
		AccuracyObjective accy = new AccuracyObjective();
		GeometricMeanScorer<BinaryClassificationTarget> scorer = new GeometricMeanScorer<BinaryClassificationTarget>(
				accy);
		Classifiers classifier = selector.optimize(insts, scorer, true);
		FoldEvaluator eval = new FoldEvaluator();
		Classifiers classifierTrue = null;
		double max = Double.MIN_VALUE;
		for (Classifiers cl : clSet) {
			double obj = eval.evaluate(insts,
					new GeometricMeanScorer<BinaryClassificationTarget>(accy),
					cl.getClassifierBuilder(5, true));
			if (obj > max) {
				max = obj;
				classifierTrue = cl;
			}
		}
		assertEquals(classifierTrue, classifier);
	}

}
