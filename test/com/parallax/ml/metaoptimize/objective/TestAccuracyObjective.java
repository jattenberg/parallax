/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.metaoptimize.objective;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.util.Map;

import org.junit.Test;

import com.google.common.collect.Maps;
import com.parallax.ml.classifier.Classifiers;
import com.parallax.ml.classifier.linear.updateable.PerceptronWithMargin;
import com.parallax.ml.evaluation.ConfusionMatrix;
import com.parallax.ml.instance.BinaryClassificationInstance;
import com.parallax.ml.instance.BinaryClassificationInstances;
import com.parallax.ml.objective.AccuracyObjective;
import com.parallax.ml.target.BinaryTargetNumericParser;
import com.parallax.pipeline.FileSource;
import com.parallax.pipeline.Pipeline;
import com.parallax.pipeline.csv.NumericCSVtoLabeledVectorPipe;
import com.parallax.pipeline.file.FileToLinesPipe;
import com.parallax.pipeline.instance.BinaryClassificationInstancesSink;
import com.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;

/**
 * The Class TestAccuracyObjective.
 */
public class TestAccuracyObjective {

	/** The file. */
	File file = new File("data/iris.data");

	/** The bins. */
	int bins = 5;

	/**
	 * Test objective.
	 */
	@Test
	public void testObjective() {
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

		PerceptronWithMargin model = (PerceptronWithMargin) Classifiers
				.getClassifier(PerceptronWithMargin.class, bins, true);
		model.train(insts.getTraining(1, 2));

		AccuracyObjective objective = new AccuracyObjective();
		objective.evaluate(insts.getTesting(1, 2), model);
		ConfusionMatrix conf = new ConfusionMatrix(2);

		for (BinaryClassificationInstance inst : insts.getTesting(1, 2))
			conf.addInfo(inst.getLabel(), model.predict(inst));

		assertEquals(conf.computeAccuracy(),
				objective.evaluate(insts.getTesting(1, 2), model), 0.001);
	}

}
