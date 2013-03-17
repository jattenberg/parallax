/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.evaluation;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.util.Map;

import org.junit.Test;

import com.google.common.collect.Maps;
import com.parallax.ml.classifier.Classifiers;
import com.parallax.ml.classifier.bayes.NaiveBayes;
import com.parallax.ml.instance.BinaryClassificationInstance;
import com.parallax.ml.instance.BinaryClassificationInstances;
import com.parallax.ml.objective.AUCObjective;
import com.parallax.ml.target.BinaryTargetNumericParser;
import com.parallax.pipeline.FileSource;
import com.parallax.pipeline.Pipeline;
import com.parallax.pipeline.csv.NumericCSVtoLabeledVectorPipe;
import com.parallax.pipeline.file.FileToLinesPipe;
import com.parallax.pipeline.instance.BinaryClassificationInstancesSink;
import com.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;
import com.parallax.pipeline.instance.IntroduceLabelNoisePipe;

/**
 * The Class TestAUCObjective.
 */
public class TestAUCObjective {

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
								new BinaryTargetNumericParser()))
				.addPipe(new IntroduceLabelNoisePipe(0.05));
		BinaryClassificationInstancesSink sink = new BinaryClassificationInstancesSink();
		sink.setSource(pipeline);

		assertTrue(sink.hasNext());
		BinaryClassificationInstances insts = sink.next().shuffle();
		BinaryClassificationInstances training = insts.getTraining(1, 2);
		BinaryClassificationInstances testing = insts.getTesting(1, 2);

		NaiveBayes model = (NaiveBayes) Classifiers.NB.getClassifier(bins,
				false);
		model.train(training);

		AUCObjective obj = new AUCObjective();
		ReceiverOperatingCharacteristic roc = new ReceiverOperatingCharacteristic();

		double guessAUC = obj.evaluate(testing, model);
		for (BinaryClassificationInstance inst : testing)
			roc.add(inst.getLabel(), model.predict(inst));

		assertEquals(roc.binaryAUC(), guessAUC, 0.001);

	}

}
