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
import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.ml.objective.HingeLossObjective;
import com.dsi.parallax.ml.target.BinaryTargetNumericParser;
import com.dsi.parallax.ml.util.MLUtils;
import com.dsi.parallax.pipeline.FileSource;
import com.dsi.parallax.pipeline.Pipeline;
import com.dsi.parallax.pipeline.csv.NumericCSVtoLabeledVectorPipe;
import com.dsi.parallax.pipeline.file.FileToLinesPipe;
import com.dsi.parallax.pipeline.instance.BinaryClassificationInstancesSink;
import com.dsi.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;
import com.google.common.collect.Maps;

/**
 * The Class TestHingeLossObjective.
 */
public class TestHingeLossObjective {

	/** The file. */
	File file = new File("data/iris.data");

	/** The bins. */
	int bins = 5;

	/**
	 * Test.
	 */
	@Test
	public void test() {
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

		HingeLossObjective objective = new HingeLossObjective();

		double count = 0, total = 0;
		for (BinaryClassificationInstance inst : insts.getTesting(1, 2)) {
			count++;
			double prediction = MLUtils.probToSVMInterval(model.predict(inst)
					.getValue());
			double label = MLUtils
					.probToSVMInterval(inst.getLabel().getValue());
			total += Math.max(0, 1 - label * prediction);
		}
		assertEquals(total / count,
				objective.evaluate(insts.getTesting(1, 2), model), 0.001);

	}

}
