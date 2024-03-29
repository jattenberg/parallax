/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.classifier.rules;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.util.Map;

import org.junit.Test;

import com.dsi.parallax.ml.classifier.rules.MeanClassifier;
import com.dsi.parallax.ml.evaluation.OnlineEvaluation;
import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.ml.target.BinaryTargetNumericParser;
import com.dsi.parallax.pipeline.FileSource;
import com.dsi.parallax.pipeline.Pipeline;
import com.dsi.parallax.pipeline.csv.NumericCSVtoLabeledVectorPipe;
import com.dsi.parallax.pipeline.file.FileToLinesPipe;
import com.dsi.parallax.pipeline.instance.BinaryClassificationInstancesSink;
import com.dsi.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;
import com.google.common.collect.Maps;

/**
 * The Class TestMeanClassifier.
 */
public class TestMeanClassifier {

	/** The file. */
	static File file = new File("data/iris.data");

	/** The bins. */
	static int bins = 4;

	/** The insts. */
	static BinaryClassificationInstances insts = getInstances();

	/**
	 * Test train and predict.
	 */
	@Test
	public void testTrainAndPredict() {

		int folds = 5;
		for (int fold = 0; fold < folds; fold++) {

			MeanClassifier model = new MeanClassifier((int) Math.pow(2, 18),
					true);
			OnlineEvaluation eval = new OnlineEvaluation();

			model.train(insts.getTraining(fold, folds));
			for (BinaryClassificationInstance x : insts.getTesting(fold, folds)) {
				double pred = model.predict(x).getValue();
				double label = x.getLabel().getValue();

				eval.add(label, pred);
			}
			assertTrue(eval.computeAUC() == 0.5);
			assertTrue(eval.computeAccuracy() > 0.5);
		}
	}

	/**
	 * Gets the instances.
	 * 
	 * @return the instances
	 */
	private static BinaryClassificationInstances getInstances() {
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
								new BinaryTargetNumericParser()));
		BinaryClassificationInstancesSink sink = new BinaryClassificationInstancesSink();
		sink.setSource(pipeline);

		assertTrue(sink.hasNext());
		BinaryClassificationInstances insts = sink.next();
		return insts;
	}

}
