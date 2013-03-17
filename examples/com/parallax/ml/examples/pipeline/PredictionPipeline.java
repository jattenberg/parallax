/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.examples.pipeline;

import java.io.File;
import java.util.Iterator;

import com.parallax.ml.classifier.linear.updateable.AROWClassifier;
import com.parallax.ml.classifier.smoother.SmootherType;
import com.parallax.ml.evaluation.OnlineEvaluation;
import com.parallax.ml.target.BinaryClassificationTarget;
import com.parallax.ml.target.BinaryTargetNumericParser;
import com.parallax.ml.vector.util.ValueScaling;
import com.parallax.pipeline.Context;
import com.parallax.pipeline.FileSource;
import com.parallax.pipeline.Pipeline;
import com.parallax.pipeline.ValueScalingPipe;
import com.parallax.pipeline.classifier.ModelEvaluationSink;
import com.parallax.pipeline.classifier.PredictionPipe;
import com.parallax.pipeline.classifier.SequentialClassifierTrainingPipe;
import com.parallax.pipeline.file.FileToLinesPipe;
import com.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;
import com.parallax.pipeline.text.StringSequenceToNumericDictionaryPipe;
import com.parallax.pipeline.text.StringToTokenSequencePipe;
import com.parallax.pipeline.text.StringToVWPipe;
import com.parallax.pipeline.text.TextSanitizerPipe;
import com.parallax.pipeline.text.VWtoLabeledStringPipe;

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
			model.setSmoothertype(SmootherType.UPDATEABLEPLATT).initialize();

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
