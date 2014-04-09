/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.supervisedfeture;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;

import org.junit.Test;

import com.dsi.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.LogisticRegressionBuilder;
import com.dsi.parallax.ml.evaluation.OnlineEvaluation;
import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.ml.target.BinaryTargetNumericParser;
import com.dsi.parallax.ml.vector.util.ValueScaling;
import com.dsi.parallax.pipeline.FileSource;
import com.dsi.parallax.pipeline.Pipeline;
import com.dsi.parallax.pipeline.Sink;
import com.dsi.parallax.pipeline.ValueScalingPipe;
import com.dsi.parallax.pipeline.file.FileToLinesPipe;
import com.dsi.parallax.pipeline.instance.BinaryClassificationInstancesSink;
import com.dsi.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;
import com.dsi.parallax.pipeline.supervisedfeture.ClassifierPredictionFeaturePipe;
import com.dsi.parallax.pipeline.text.StringSequenceToNumericDictionaryPipe;
import com.dsi.parallax.pipeline.text.StringToTokenSequencePipe;
import com.dsi.parallax.pipeline.text.StringToVWPipe;
import com.dsi.parallax.pipeline.text.TextSanitizerPipe;
import com.dsi.parallax.pipeline.text.VWtoLabeledStringPipe;

/**
 * The Class TestClassifierPredictionFeaturePipeline.
 */
public class TestClassifierPredictionFeaturePipeline {

	/**
	 * Test.
	 */
	@Test
	public void test() {
		Pipeline<File, BinaryClassificationInstance> pipeline;
		Sink<BinaryClassificationInstance, BinaryClassificationInstances> instancesSink;
		String file = "data/science.small.vw";
		int dimensions = (int) Math.pow(2, 18);

		pipeline = Pipeline
				.newPipeline(new FileSource(file))
				.addPipe(new FileToLinesPipe())
				.addPipe(new StringToVWPipe())
				.addPipe(new VWtoLabeledStringPipe())
				.addPipe(new TextSanitizerPipe())
				.addPipe(new StringToTokenSequencePipe())
				.addPipe(new StringSequenceToNumericDictionaryPipe(dimensions))
				.addPipe(new ValueScalingPipe(ValueScaling.ABS))
				.addPipe(new ValueScalingPipe(ValueScaling.PRESERVING))
				.addPipe(
						new BinaryInstancesFromVectorPipe(
								new BinaryTargetNumericParser()))
				.addPipe(
						new ClassifierPredictionFeaturePipe(
								new LogisticRegressionBuilder((int) Math.pow(2,
										18), true)));
		instancesSink = new BinaryClassificationInstancesSink();
		instancesSink.setSource(pipeline);
		assertTrue(instancesSink.hasNext());

		OnlineEvaluation eval = new OnlineEvaluation(100000);
		BinaryClassificationInstances insts = instancesSink.next();
		for (BinaryClassificationInstance inst : insts) {
			assertEquals(inst.size(), dimensions + 1);
			eval.add(inst.getLabel().getValue(),
					inst.getFeatureValue(dimensions));
		}
		assertTrue(eval.computeAUC() > 0.5);
	}

}
