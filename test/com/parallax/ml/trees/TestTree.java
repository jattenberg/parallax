/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.trees;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.util.Map;
import java.util.Set;

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
 * The Class TestTree.
 */
public class TestTree {

	/** The file. */
	static File file = new File("data/iris.data");

	/** The bins. */
	static int bins = 5;

	/** The insts. */
	static BinaryClassificationInstances insts = getInstances();

	/**
	 * Test root and leaf.
	 */
	@Test
	public void testRootAndLeaf() {
		Tree<BinaryClassificationTarget> tree = new Tree<BinaryClassificationTarget>(
				null);
		assertTrue(tree.isRoot());
		assertTrue(tree.isLeaf());

		Tree<BinaryClassificationTarget> node = new Tree<BinaryClassificationTarget>(
				null, new GreaterThanSplitCondition(0, 0), tree);
		assertTrue(node.isLeaf());
		assertFalse(tree.isLeaf());
		assertFalse(node.isRoot());
	}

	/**
	 * Test children.
	 */
	@Test
	public void testChildren() {
		Tree<BinaryClassificationTarget> tree = new Tree<BinaryClassificationTarget>(
				insts);
		assertNotNull(tree.getInstances());
	}

	/**
	 * Test get children.
	 */
	@Test
	public void testGetChildren() {
		Tree<BinaryClassificationTarget> tree = new Tree<BinaryClassificationTarget>(
				insts);

		GenericPair<BinaryClassificationInstances, BinaryClassificationInstances> leftAndRight = insts
				.splitOnValue(1, 3.);

		Tree<BinaryClassificationTarget> leftTree = new Tree<BinaryClassificationTarget>(
				leftAndRight.first, new LessThanOrEqualsToSplitCondition(1, 3),
				tree);
		Tree<BinaryClassificationTarget> rightTree = new Tree<BinaryClassificationTarget>(
				leftAndRight.second, new GreaterThanSplitCondition(1, 3), tree);

		assertNotNull(tree.getChildren());
		assertEquals(tree.getChildren().size(), 2);
		Set<Tree<BinaryClassificationTarget>> children = tree.getChildren();

		assertTrue(children.contains(leftTree));
		assertTrue(children.contains(rightTree));

		int[] sizes = new int[2];
		for (BinaryClassificationInstance inst : insts) {
			if (inst.getFeatureValue(1) <= 3)
				sizes[0]++;
			else
				sizes[1]++;
		}

		for (Tree<BinaryClassificationTarget> child : children) {
			assertTrue(child.getInstances().size() == sizes[0]
					|| child.getInstances().size() == sizes[1]);
		}
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
