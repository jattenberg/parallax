/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.discretization;

import static org.junit.Assert.assertEquals;

import java.io.File;
import java.util.Iterator;
import java.util.Map;

import org.junit.Test;

import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.pipeline.Context;
import com.dsi.parallax.pipeline.FileSource;
import com.dsi.parallax.pipeline.Pipeline;
import com.dsi.parallax.pipeline.csv.NumericCSVtoLabeledVectorPipe;
import com.dsi.parallax.pipeline.discretization.QuantileDiscretizationPipe;
import com.dsi.parallax.pipeline.file.FileToLinesPipe;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;

/**
 * The Class TestQuantileDiscretizationPipe.
 */
public class TestQuantileDiscretizationPipe {

	/** The file. */
	File file = new File("data/iris.data");

	/** The bins. */
	int bins = 4;

	/**
	 * Test.
	 */
	@Test
	public void test() {
		Pipeline<File, LinearVector> pipeline;
		Map<String, String> labelMap = Maps.newHashMap();
		labelMap.put("Iris-setosa", 1 + "");
		labelMap.put("Iris-versicolor", 0 + "");
		labelMap.put("Iris-virginica", 0 + "");

		pipeline = Pipeline.newPipeline(new FileSource(file))
				.addPipe(new FileToLinesPipe())
				.addPipe(new NumericCSVtoLabeledVectorPipe(-1, 4, labelMap))
				.addPipe(new QuantileDiscretizationPipe(5, false));
		Iterator<Context<LinearVector>> pout = pipeline.process();

		while (pout.hasNext()) {
			LinearVector vect = pout.next().getData();
			assertEquals(vect.size(), 4 + 5 * 4);
			assertEquals(vect.L0Norm(), 4, 0.000001);
		}
	}

	/**
	 * Test limited pipe.
	 */
	@Test
	public void testLimitedPipe() {
		Pipeline<File, LinearVector> pipeline;
		Map<String, String> labelMap = Maps.newHashMap();
		labelMap.put("Iris-setosa", 1 + "");
		labelMap.put("Iris-versicolor", 0 + "");
		labelMap.put("Iris-virginica", 0 + "");

		pipeline = Pipeline.newPipeline(new FileSource(file))
				.addPipe(new FileToLinesPipe())
				.addPipe(new NumericCSVtoLabeledVectorPipe(-1, 4, labelMap))
				.addPipe(new QuantileDiscretizationPipe(5, false, 22));
		Iterator<Context<LinearVector>> pout = pipeline.process();

		while (pout.hasNext()) {
			LinearVector vect = pout.next().getData();
			assertEquals(vect.size(), 4 + 5 * 4);
			assertEquals(vect.L0Norm(), 4, 0.000001);
		}
	}

	/**
	 * Test keeps feats.
	 */
	@Test
	public void testKeepsFeats() {
		Pipeline<File, LinearVector> pipeline;
		Map<String, String> labelMap = Maps.newHashMap();
		labelMap.put("Iris-setosa", 1 + "");
		labelMap.put("Iris-versicolor", 0 + "");
		labelMap.put("Iris-virginica", 0 + "");

		pipeline = Pipeline.newPipeline(new FileSource(file))
				.addPipe(new FileToLinesPipe())
				.addPipe(new NumericCSVtoLabeledVectorPipe(-1, 4, labelMap))
				.addPipe(new QuantileDiscretizationPipe(5, true, 22));
		Iterator<Context<LinearVector>> pout = pipeline.process();

		while (pout.hasNext()) {
			LinearVector vect = pout.next().getData();
			assertEquals(vect.size(), 4 + 5 * 4);
			assertEquals(vect.L0Norm(), 4 + 4, 0.000001);
		}
	}

	/**
	 * Test specify continuous.
	 */
	@Test
	public void testSpecifyContinuous() {
		Pipeline<File, LinearVector> pipeline;
		Map<String, String> labelMap = Maps.newHashMap();
		labelMap.put("Iris-setosa", 1 + "");
		labelMap.put("Iris-versicolor", 0 + "");
		labelMap.put("Iris-virginica", 0 + "");

		pipeline = Pipeline
				.newPipeline(new FileSource(file))
				.addPipe(new FileToLinesPipe())
				.addPipe(new NumericCSVtoLabeledVectorPipe(-1, 4, labelMap))
				.addPipe(
						new QuantileDiscretizationPipe(5, true, Sets
								.newHashSet(1)));
		Iterator<Context<LinearVector>> pout = pipeline.process();

		while (pout.hasNext()) {
			LinearVector vect = pout.next().getData();
			assertEquals(vect.size(), 4 + 1 * 5);
			assertEquals(vect.L0Norm(), 1 + 4, 0.000001);
		}

		pipeline = Pipeline
				.newPipeline(new FileSource(file))
				.addPipe(new FileToLinesPipe())
				.addPipe(new NumericCSVtoLabeledVectorPipe(-1, 4, labelMap))
				.addPipe(
						new QuantileDiscretizationPipe(5, true, Sets
								.newHashSet(1, 0)));
		pout = pipeline.process();

		while (pout.hasNext()) {
			LinearVector vect = pout.next().getData();
			assertEquals(vect.size(), 4 + 2 * 5);
			assertEquals(vect.L0Norm(), 2 + 4, 0.000001);
		}
	}
}
