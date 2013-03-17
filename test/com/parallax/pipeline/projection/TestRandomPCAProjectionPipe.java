/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline.projection;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;
import java.util.Iterator;

import org.junit.Test;

import com.parallax.ml.instance.BinaryClassificationInstance;
import com.parallax.ml.projection.RandomPCAProjection;
import com.parallax.pipeline.Context;
import com.parallax.pipeline.FileSource;
import com.parallax.pipeline.Pipeline;
import com.parallax.pipeline.csv.NumericCSVtoLabeledVectorPipe;
import com.parallax.pipeline.file.FileToLinesPipe;
import com.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;

/**
 * The Class TestRandomPCAProjectionPipe.
 */
public class TestRandomPCAProjectionPipe {

	/** The file. */
	File file = new File("data/iris.data");

	/** The bins. */
	int bins = 10000;

	/**
	 * Test projection works.
	 * 
	 * @throws IOException
	 *             Signals that an I/O exception has occurred.
	 */
	@Test
	public void testProjectionWorks() throws IOException {

		for (int i = 1; i < 20; i++) {
			Pipeline<File, BinaryClassificationInstance> pipeline;
			RandomPCAProjection pca = new RandomPCAProjection(5, i + 2, i);
			pipeline = Pipeline.newPipeline(new FileSource(file))
					.addPipe(new FileToLinesPipe())
					.addPipe(new NumericCSVtoLabeledVectorPipe(-1, 4))
					.addPipe(new RandomPCAProjectionPipe(pca))
					.addPipe(new BinaryInstancesFromVectorPipe());

			assertTrue(!pca.isBuilt());
			Iterator<Context<BinaryClassificationInstance>> it = pipeline
					.process();

			while (it.hasNext()) {
				BinaryClassificationInstance inst = it.next().getData();
				assertTrue(inst.L0Norm() <= i);
				assertTrue(pca.isBuilt());
			}

		}
	}

}
