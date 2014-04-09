/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.metaoptimize.objective;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.util.Map;

import org.junit.Test;

import com.dsi.parallax.ml.classifier.Classifiers;
import com.dsi.parallax.ml.classifier.linear.updateable.WinnowClassifier;
import com.dsi.parallax.ml.evaluation.ReceiverOperatingCharacteristic;
import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.ml.objective.AUCObjective;
import com.dsi.parallax.ml.target.BinaryTargetNumericParser;
import com.dsi.parallax.pipeline.FileSource;
import com.dsi.parallax.pipeline.Pipeline;
import com.dsi.parallax.pipeline.csv.NumericCSVtoLabeledVectorPipe;
import com.dsi.parallax.pipeline.file.FileToLinesPipe;
import com.dsi.parallax.pipeline.instance.BinaryClassificationInstancesSink;
import com.dsi.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;
import com.google.common.collect.Maps;

/**
 * The Class TestAUCObjective.
 */
public class TestAUCObjective {

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

		WinnowClassifier model = (WinnowClassifier) Classifiers.getClassifier(
				WinnowClassifier.class, bins, true);
		model.train(insts.getTraining(1, 2));

		AUCObjective objective = new AUCObjective();
		objective.evaluate(insts.getTesting(1, 2), model);

		ReceiverOperatingCharacteristic roc = new ReceiverOperatingCharacteristic();
		for (BinaryClassificationInstance inst : insts.getTesting(1, 2))
			roc.add(inst.getLabel(), model.predict(inst));

		assertEquals(roc.binaryAUC(),
				objective.evaluate(insts.getTesting(1, 2), model), 0.001);
	}

}
