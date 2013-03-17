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
import com.parallax.ml.classifier.linear.updateable.WinnowClassifier;
import com.parallax.ml.instance.BinaryClassificationInstance;
import com.parallax.ml.instance.BinaryClassificationInstances;
import com.parallax.ml.objective.HingeLossObjective;
import com.parallax.ml.target.BinaryTargetNumericParser;
import com.parallax.ml.util.MLUtils;
import com.parallax.pipeline.FileSource;
import com.parallax.pipeline.Pipeline;
import com.parallax.pipeline.csv.NumericCSVtoLabeledVectorPipe;
import com.parallax.pipeline.file.FileToLinesPipe;
import com.parallax.pipeline.instance.BinaryClassificationInstancesSink;
import com.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;

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
