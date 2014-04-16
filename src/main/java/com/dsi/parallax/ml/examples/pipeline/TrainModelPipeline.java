/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.examples.pipeline;

import java.io.File;

import com.dsi.parallax.ml.classifier.lazy.KNNMixingType;
import com.dsi.parallax.ml.classifier.lazy.SequentialKNN;
import com.dsi.parallax.ml.evaluation.OnlineEvaluation;
import com.dsi.parallax.ml.target.BinaryTargetNumericParser;
import com.dsi.parallax.ml.vector.util.ValueScaling;
import com.dsi.parallax.pipeline.FileSource;
import com.dsi.parallax.pipeline.Pipeline;
import com.dsi.parallax.pipeline.ValueScalingPipe;
import com.dsi.parallax.pipeline.classifier.ModelEvaluationSink;
import com.dsi.parallax.pipeline.classifier.SequentialClassifierTrainingPipe;
import com.dsi.parallax.pipeline.file.FileToLinesPipe;
import com.dsi.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;
import com.dsi.parallax.pipeline.text.StringSequenceToNumericDictionaryPipe;
import com.dsi.parallax.pipeline.text.StringToTokenSequencePipe;
import com.dsi.parallax.pipeline.text.StringToVWPipe;
import com.dsi.parallax.pipeline.text.TextSanitizerPipe;
import com.dsi.parallax.pipeline.text.VWtoLabeledStringPipe;

// TODO: Auto-generated Javadoc
/**
 * The Class TrainModelPipeline.
 */
public class TrainModelPipeline {

	/**
	 * The main method.
	 *
	 * @param args the arguments
	 */
	public static void main(String[] args) {

		int dimensions = (int) Math.pow(2, 16);
		String filename = "data/science.small.vw";

		SequentialKNN model = new SequentialKNN(dimensions, true);
		model.setLabelMizingType(KNNMixingType.MEAN).setK(15).initialize();

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
						new SequentialClassifierTrainingPipe<SequentialKNN>(
								model, 20));

		sink.setSource(pipeline);
		OnlineEvaluation eval = sink.next();
		System.out.println(eval);
	}

}
