/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.projection;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;
import java.util.Iterator;

import org.junit.Test;

import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.projection.PrincipalComponentsAnalysis;
import com.dsi.parallax.pipeline.Context;
import com.dsi.parallax.pipeline.FileSource;
import com.dsi.parallax.pipeline.Pipeline;
import com.dsi.parallax.pipeline.file.FileToLinesPipe;
import com.dsi.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;
import com.dsi.parallax.pipeline.projection.PCAPipe;
import com.dsi.parallax.pipeline.text.StringSequenceToNGramsPipe;
import com.dsi.parallax.pipeline.text.StringSequenceToNumericDictionaryPipe;
import com.dsi.parallax.pipeline.text.StringToTokenSequencePipe;
import com.dsi.parallax.pipeline.text.TextSanitizerPipe;

/**
 * The Class TestPCAPipe.
 */
public class TestPCAPipe {

	/** The file. */
	File file = new File(".classpath");

	/** The bins. */
	int bins = 100;

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
			PrincipalComponentsAnalysis pca = new PrincipalComponentsAnalysis(
					bins, i);
			pipeline = Pipeline.newPipeline(new FileSource(file))
					.addPipe(new FileToLinesPipe())
					.addPipe(new TextSanitizerPipe())
					.addPipe(new StringToTokenSequencePipe())
					.addPipe(new StringSequenceToNGramsPipe(2))
					.addPipe(new StringSequenceToNumericDictionaryPipe(bins))
					.addPipe(new PCAPipe(pca))
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
