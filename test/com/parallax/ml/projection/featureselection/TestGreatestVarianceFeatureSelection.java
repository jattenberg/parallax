/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.projection.featureselection;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import org.apache.commons.math.stat.descriptive.DescriptiveStatistics;
import org.junit.Test;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.parallax.ml.instance.BinaryClassificationInstance;
import com.parallax.ml.instance.BinaryClassificationInstances;
import com.parallax.ml.target.BinaryTargetNumericParser;
import com.parallax.ml.util.pair.PrimitivePair;
import com.parallax.ml.util.pair.SecondDescendingComparator;
import com.parallax.pipeline.FileSource;
import com.parallax.pipeline.Pipeline;
import com.parallax.pipeline.csv.NumericCSVtoLabeledVectorPipe;
import com.parallax.pipeline.file.FileToLinesPipe;
import com.parallax.pipeline.instance.BinaryClassificationInstancesSink;
import com.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;

/**
 * The Class TestGreatestVarianceFeatureSelection.
 */
public class TestGreatestVarianceFeatureSelection {

	/** The file. */
	File file = new File("data/iris.data");

	/** The bins. */
	int bins = 4;

	/**
	 * Test actually projects.
	 */
	@Test
	public void testActuallyProjects() {
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
								new BinaryTargetNumericParser()));

		BinaryClassificationInstancesSink sink = new BinaryClassificationInstancesSink();
		sink.setSource(pipeline);

		assertTrue(sink.hasNext());
		BinaryClassificationInstances insts = sink.next().shuffle();

		GreatestVarianceFeatureSelection selector = new GreatestVarianceFeatureSelection(
				bins, 2);
		selector.build(insts);
		assertEquals(bins, selector.getKeptFeatures().length);

		BinaryClassificationInstances projected = new BinaryClassificationInstances(
				2);
		for (BinaryClassificationInstance inst : insts) {
			BinaryClassificationInstance project = selector.project(inst);
			assertEquals(2, project.getDimension());
			projected.addInstance(project);
		}
		assertEquals(insts.size(), projected.size());
		assertEquals(insts.getNumNeg(), projected.getNumNeg());
		assertEquals(insts.getNumPos(), projected.getNumPos());
	}

	/**
	 * Test greatest variance.
	 */
	@Test
	public void testGreatestVariance() {
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
								new BinaryTargetNumericParser()));

		BinaryClassificationInstancesSink sink = new BinaryClassificationInstancesSink();
		sink.setSource(pipeline);

		assertTrue(sink.hasNext());
		BinaryClassificationInstances insts = sink.next().shuffle();

		GreatestVarianceFeatureSelection selector = new GreatestVarianceFeatureSelection(
				bins, 2);
		selector.build(insts);
		DescriptiveStatistics[] dists = new DescriptiveStatistics[bins];
		for (int i = 0; i < bins; i++)
			dists[i] = new DescriptiveStatistics();
		for (BinaryClassificationInstance inst : insts)
			for (int i = 0; i < bins; i++)
				dists[i].addValue(inst.getFeatureValue(i));

		List<PrimitivePair> featureVariancePairs = Lists.newArrayList();
		for (int i = 0; i < bins; i++)
			featureVariancePairs.add(new PrimitivePair(i, dists[i]
					.getVariance()));
		Collections
				.sort(featureVariancePairs, new SecondDescendingComparator());

		for (int i = 0; i < bins; i++)
			assertEquals(selector.getKeptFeatures()[i],
					(int) featureVariancePairs.get(i).getFirst());
	}

}
