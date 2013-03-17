/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline.precompiled;

import java.io.File;

import com.parallax.ml.instance.BinaryClassificationInstance;
import com.parallax.ml.instance.BinaryClassificationInstances;
import com.parallax.ml.target.BinaryTargetNumericParser;
import com.parallax.ml.vector.util.ValueScaling;
import com.parallax.pipeline.FileSource;
import com.parallax.pipeline.Pipeline;
import com.parallax.pipeline.Sink;
import com.parallax.pipeline.ValueScalingPipe;
import com.parallax.pipeline.file.FileToLinesPipe;
import com.parallax.pipeline.instance.BinaryClassificationInstancesSink;
import com.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;
import com.parallax.pipeline.instance.IntroduceLabelNoisePipe;
import com.parallax.pipeline.text.StringSequenceToNumericDictionaryPipe;
import com.parallax.pipeline.text.StringToTokenSequencePipe;
import com.parallax.pipeline.text.StringToVWPipe;
import com.parallax.pipeline.text.TextSanitizerPipe;
import com.parallax.pipeline.text.VWtoLabeledStringPipe;

/**
 * A pre-compiled pipeline that reads a file of text-based machine learning
 * instances in the
 * {@link <a href="https://github.com/JohnLangford/vowpal_wabbit/wiki/Tutorial">Vowpal Wabbit</a>
 * format. and returns a set of binary classification instances. Useful for text
 * classification problems. Labels are perturbed (flipped to the opposite binary
 * label) with some user specified probability *
 */
public class VWtoBinaryInstancesWithLabelNoisePipeline implements
		PipelineSystem<File, BinaryClassificationInstances> {

	/** The pipeline used internally to process data. */
	private Pipeline<File, BinaryClassificationInstance> pipeline;

	/** The instances sink used to extract instances. */
	private Sink<BinaryClassificationInstance, BinaryClassificationInstances> instancesSink;

	/**
	 * Instantiates a new v wto binary instances with label noise pipeline.
	 * 
	 * @param file
	 *            the file to be read
	 * @param dimensions
	 *            the dimensions to be used in the derived instances
	 * @param noiseRatio
	 *            the probability of flipping the binary label
	 */
	public VWtoBinaryInstancesWithLabelNoisePipeline(File file, int dimensions,
			double noiseRatio) {
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
				.addPipe(new IntroduceLabelNoisePipe(noiseRatio));
		instancesSink = new BinaryClassificationInstancesSink();
		instancesSink.setSource(pipeline);
	}

	/**
	 * Instantiates a new v wto binary instances with label noise pipeline.
	 * 
	 * @param file
	 *            the file to be read
	 * @param dimensions
	 *            the dimensions to be used in the derived instances
	 * @param noiseRatio
	 *            the probability of flipping the binary label
	 */
	public VWtoBinaryInstancesWithLabelNoisePipeline(String file,
			int dimensions, double noiseRatio) {
		this(new File(file), dimensions, noiseRatio);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.util.Iterator#hasNext()
	 */
	@Override
	public boolean hasNext() {
		return instancesSink.hasNext();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.util.Iterator#next()
	 */
	@Override
	public BinaryClassificationInstances next() {
		return instancesSink.next();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.util.Iterator#remove()
	 */
	@Override
	public void remove() {
		throw new UnsupportedOperationException(
				"remove is not supported in pipeline system");
	}

}
