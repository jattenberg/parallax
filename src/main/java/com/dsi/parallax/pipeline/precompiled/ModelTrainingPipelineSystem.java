/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.precompiled;

import com.dsi.parallax.ml.classifier.Classifier;
import com.dsi.parallax.ml.classifier.ClassifierBuilder;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;

import java.io.File;

// TODO: Auto-generated Javadoc
/**
 * The Class ModelTrainingPipelineSystem.
 */
public class ModelTrainingPipelineSystem implements
		PipelineSystem<File, Classifier<?>> {

	/** The model. */
	private final Classifier<?> model;
	
	/** The file source. */
	private final VWtoBinaryInstancesPipeline fileSource;

	/**
	 * Instantiates a new model training pipeline system.
	 *
	 * @param file the file
	 * @param dimensions the dimensions
	 * @param builder the builder
	 */
	public ModelTrainingPipelineSystem(File file, int dimensions,
			ClassifierBuilder<?, ?> builder) {
		this(file, dimensions, builder.build());

	}

	/**
	 * Instantiates a new model training pipeline system.
	 *
	 * @param file the file
	 * @param dimensions the dimensions
	 * @param model the model
	 */
	public ModelTrainingPipelineSystem(File file, int dimensions,
			Classifier<?> model) {
		this.model = model;
		fileSource = new VWtoBinaryInstancesPipeline(file, dimensions);
	}

	/* (non-Javadoc)
	 * @see java.util.Iterator#hasNext()
	 */
	@Override
	public boolean hasNext() {
		return fileSource.hasNext();
	}

	/* (non-Javadoc)
	 * @see java.util.Iterator#next()
	 */
	@Override
	public Classifier<?> next() {
		BinaryClassificationInstances insts = fileSource.next();
		model.train(insts);

		return model;
	}

	/* (non-Javadoc)
	 * @see java.util.Iterator#remove()
	 */
	@Override
	public void remove() {
		throw new UnsupportedOperationException(
				"remove isnt supported for ModelTrainingPipelineSystem");
	}

}
