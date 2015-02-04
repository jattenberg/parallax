/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.examples.pipeline;

import com.dsi.parallax.ml.classifier.linear.updateable.AROWClassifier;
import com.dsi.parallax.ml.evaluation.OnlineEvaluation;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.target.BinaryTargetNumericParser;
import com.dsi.parallax.ml.vector.util.ValueScaling;
import com.dsi.parallax.pipeline.Context;
import com.dsi.parallax.pipeline.FileSource;
import com.dsi.parallax.pipeline.Pipeline;
import com.dsi.parallax.pipeline.ValueScalingPipe;
import com.dsi.parallax.pipeline.classifier.ModelEvaluationSink;
import com.dsi.parallax.pipeline.classifier.PredictionPipe;
import com.dsi.parallax.pipeline.classifier.SequentialClassifierTrainingPipe;
import com.dsi.parallax.pipeline.file.FileToLinesPipe;
import com.dsi.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;
import com.dsi.parallax.pipeline.text.*;

import java.io.File;
import java.util.Iterator;

// TODO: Auto-generated Javadoc
/**
 * The Class PredictionPipeline.
 */
public class PredictionPipeline {

	/**
	 * The main method.
	 *
	 * @param args the arguments
	 */
	public static void main(String[] args) {

		int dimensions = (int) Math.pow(2, 16);
		String filename = "data/science.small.vw";

		AROWClassifier model = new AROWClassifier(dimensions, true);

		ModelEvaluationSink sink = new ModelEvaluationSink();
		Pipeline<File, OnlineEvaluation> pipeline;
		pipeline = Pipeline
				.newPipeline(new FileSource(filename))
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
						new SequentialClassifierTrainingPipe<AROWClassifier>(
								model, 20));

		sink.setSource(pipeline);
		OnlineEvaluation eval = sink.next();
		System.out.println(eval);

		Pipeline<File, BinaryClassificationTarget> testPipeline;
		testPipeline = Pipeline
				.newPipeline(new FileSource(filename))
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
				.addPipe(new PredictionPipe<AROWClassifier>(model));
		Iterator<Context<BinaryClassificationTarget>> iterator = testPipeline
				.process();
		while (iterator.hasNext()) {
			Context<BinaryClassificationTarget> context = iterator.next();
			System.out.println(context.getLabel() + "->" + context.getData());
		}
	}
}
