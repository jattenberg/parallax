/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline.projection;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.util.Iterator;

import org.junit.Test;

import com.parallax.ml.instance.BinaryClassificationInstance;
import com.parallax.ml.projection.PolynomialHashProjection;
import com.parallax.pipeline.Context;
import com.parallax.pipeline.FileSource;
import com.parallax.pipeline.Pipeline;
import com.parallax.pipeline.file.FileToLinesPipe;
import com.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;
import com.parallax.pipeline.text.StringSequenceToNGramsPipe;
import com.parallax.pipeline.text.StringSequenceToNumericDictionaryPipe;
import com.parallax.pipeline.text.StringToTokenSequencePipe;
import com.parallax.pipeline.text.TextSanitizerPipe;

/**
 * The Class TestRandomPolynomialHashProjectionPipe.
 */
public class TestRandomPolynomialHashProjectionPipe {

	/** The file. */
	File file = new File(".gitignore");

	/** The bins. */
	int bins = 10000;

	/**
	 * Test.
	 */
	@Test
	public void test() {
		for (int j = 1; j <= 1024; j *= 4) {
			for (int i = 1; i <= 21; i += 10) {
				Pipeline<File, BinaryClassificationInstance> pipeline;
				PolynomialHashProjection pca = new PolynomialHashProjection(
						bins, j, i);
				pipeline = Pipeline
						.newPipeline(new FileSource(file))
						.addPipe(new FileToLinesPipe())
						.addPipe(new TextSanitizerPipe())
						.addPipe(new StringToTokenSequencePipe())
						.addPipe(new StringSequenceToNGramsPipe(2))
						.addPipe(
								new StringSequenceToNumericDictionaryPipe(bins))
						.addPipe(new RandomPolynomialHashProjectionPipe(pca))
						.addPipe(new BinaryInstancesFromVectorPipe());

				Iterator<Context<BinaryClassificationInstance>> it = pipeline
						.process();

				while (it.hasNext()) {
					BinaryClassificationInstance inst = it.next().getData();
					assertTrue(inst.L0Norm() <= j);
				}

			}
		}
	}

}
