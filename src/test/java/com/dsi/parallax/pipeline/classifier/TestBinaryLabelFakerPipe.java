/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.classifier;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.util.Iterator;
import java.util.Map;

import org.junit.Test;

import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.target.BinaryTargetNumericParser;
import com.dsi.parallax.ml.util.MLUtils;
import com.dsi.parallax.pipeline.Context;
import com.dsi.parallax.pipeline.FileSource;
import com.dsi.parallax.pipeline.Pipeline;
import com.dsi.parallax.pipeline.classifier.BinaryLabelFakerPipe;
import com.dsi.parallax.pipeline.csv.NumericCSVtoLabeledVectorPipe;
import com.dsi.parallax.pipeline.file.FileToLinesPipe;
import com.dsi.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;
import com.google.common.collect.Maps;

/**
 * The Class TestBinaryLabelFakerPipe.
 */
public class TestBinaryLabelFakerPipe {

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
		labelMap.put("Iris-setosa", 0.5 + "");
		labelMap.put("Iris-versicolor", 0.5 + "");
		labelMap.put("Iris-virginica", 0.5 + "");

		pipeline = Pipeline
				.newPipeline(new FileSource(file))
				.addPipe(new FileToLinesPipe())
				.addPipe(new NumericCSVtoLabeledVectorPipe(-1, 4, labelMap))
				.addPipe(
						new BinaryInstancesFromVectorPipe(
								new BinaryTargetNumericParser()))
				.addPipe(new BinaryLabelFakerPipe());
		Iterator<Context<BinaryClassificationInstance>> out = pipeline
				.process();
		assertTrue(out.hasNext());

		while (out.hasNext()) {
			assertTrue(!MLUtils.floatingPointEquals(out.next().getData()
					.getLabel().getValue(), 0.5));
		}
	}

}
