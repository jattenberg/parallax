/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.instance;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.util.Map;

import org.junit.Test;

import com.google.common.collect.Maps;
import com.parallax.ml.projection.HashProjection;
import com.parallax.ml.target.BinaryTargetNumericParser;
import com.parallax.ml.util.pair.GenericPair;
import com.parallax.pipeline.FileSource;
import com.parallax.pipeline.Pipeline;
import com.parallax.pipeline.csv.NumericCSVtoLabeledVectorPipe;
import com.parallax.pipeline.file.FileToLinesPipe;
import com.parallax.pipeline.instance.BinaryClassificationInstancesSink;
import com.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;
import com.parallax.pipeline.instance.IntroduceLabelNoisePipe;

/**
 * The Class TestInstances.
 */
public class TestInstances {

	/** The file. */
	static File file = new File("data/iris.data");

	/** The bins. */
	static int bins = 4;

	/** The insts. */
	static BinaryClassificationInstances insts = getInstances();

	/**
	 * Test project.
	 */
	@Test
	public void testProject() {
		HashProjection projection = new HashProjection(4, 2);
		BinaryClassificationInstances projected = insts.project(projection);

		assertEquals(projected.size(), insts.size());
		for (BinaryClassificationInstance inst : projected) {
			assertEquals(2, inst.getDimension());
		}
	}

	/**
	 * Test split on value.
	 */
	@Test
	public void testSplitOnValue() {
		GenericPair<BinaryClassificationInstances, BinaryClassificationInstances> leftAndRight = insts
				.splitOnValue(1, 3.);
		int lt = 0, gt = 0;
		for (BinaryClassificationInstance inst : insts) {
			if (inst.getFeatureValue(1) <= 3)
				lt++;
			else
				gt++;
		}
		assertEquals(lt, leftAndRight.first.size());
		assertEquals(gt, leftAndRight.second.size());
	}

	@Test
	public void testCrossValidation() {
		int testing = 0, training = 0;
		int folds = 10;
		for(int i = 0; i < folds; i++) {
			training += insts.getTraining(i, folds).size();
			testing += insts.getTesting(i, folds).size();
		}
		assertEquals(training, (folds-1)*insts.size());
		assertEquals(testing, insts.size());
	}
	
	/**
	 * Gets the instances.
	 * 
	 * @return the instances
	 */
	private static BinaryClassificationInstances getInstances() {
		Pipeline<File, BinaryClassificationInstance> pipeline;
		Map<String, String> labelMap = Maps.newHashMap();
		labelMap.put("Iris-setosa", 1 + "");
		labelMap.put("Iris-versicolor", 0 + "");
		labelMap.put("Iris-virginica", 0 + "");

		pipeline = Pipeline
				.newPipeline(new FileSource(file))
				.addPipe(new FileToLinesPipe())
				.addPipe(new NumericCSVtoLabeledVectorPipe(-1, 4, labelMap))
				.addPipe(
						new BinaryInstancesFromVectorPipe(
								new BinaryTargetNumericParser()))
				.addPipe(new IntroduceLabelNoisePipe(0.05));
		BinaryClassificationInstancesSink sink = new BinaryClassificationInstancesSink();
		sink.setSource(pipeline);

		assertTrue(sink.hasNext());
		BinaryClassificationInstances insts = sink.next().shuffle();
		return insts;
	}
}
