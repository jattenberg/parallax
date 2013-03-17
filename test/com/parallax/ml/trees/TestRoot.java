/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.trees;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.util.Map;

import org.junit.Test;

import com.google.common.collect.Maps;
import com.parallax.ml.instance.BinaryClassificationInstance;
import com.parallax.ml.instance.BinaryClassificationInstances;
import com.parallax.ml.target.BinaryClassificationTarget;
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
 * The Class TestRoot.
 */
public class TestRoot {

	/** The file. */
	static File file = new File("data/iris.data");
	
	/** The bins. */
	static int bins = 5;
	
	/** The insts. */
	static BinaryClassificationInstances insts = getInstances();

	/**
	 * Test.
	 */
	@Test
	public void test() {
		Root<BinaryClassificationTarget> root = Root.buildRoot(insts);
		assertTrue(root.isRoot());
		assertTrue(root.isLeaf());

		Tree<BinaryClassificationTarget> tree = new Tree<BinaryClassificationTarget>(
				insts, new GreaterThanSplitCondition(1, 3), root);
		assertFalse(root.isLeaf());
		assertNull(root.idForLeaf(tree));

		root.addLeaf(tree);
		assertEquals(0, (int) root.idForLeaf(tree));

		tree = new Tree<BinaryClassificationTarget>(null,
				new LessThanOrEqualsToSplitCondition(1, 3), root);
		root.addLeaf(tree);
		assertEquals(1, (int) root.idForLeaf(tree));
	}

	/**
	 * Finds leaves.
	 */
	@Test
	public void findsLeaves() {
		Root<BinaryClassificationTarget> root = Root.buildRoot(insts);
		assertTrue(root.isRoot());
		assertTrue(root.isLeaf());

		GenericPair<BinaryClassificationInstances, BinaryClassificationInstances> leftAndRight = insts
				.splitOnValue(1, 3.);

		Tree<BinaryClassificationTarget> leftTree = new Tree<BinaryClassificationTarget>(
				leftAndRight.first, new LessThanOrEqualsToSplitCondition(1, 3),
				root);
		Tree<BinaryClassificationTarget> rightTree = new Tree<BinaryClassificationTarget>(
				leftAndRight.second, new GreaterThanSplitCondition(1, 3), root);

		int[] sizes = new int[2];
		for (BinaryClassificationInstance inst : insts) {
			if (inst.getFeatureValue(1) <= 3)
				sizes[0]++;
			else
				sizes[1]++;
		}

		root.addLeaf(leftTree);
		root.addLeaf(rightTree);

		assertEquals(0, (int) root.idForLeaf(leftTree));
		assertEquals(1, (int) root.idForLeaf(rightTree));

		int[] counts = new int[2];
		for (BinaryClassificationInstance inst : insts) {
			int id = root.idForInstance(inst);
			counts[id]++;
		}
		assertEquals(counts[0], sizes[0]);
		assertEquals(counts[1], sizes[1]);
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
