/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.trees;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.util.Map;

import org.junit.Test;

import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.target.BinaryTargetNumericParser;
import com.dsi.parallax.ml.trees.SingleLabelTerminator;
import com.dsi.parallax.pipeline.FileSource;
import com.dsi.parallax.pipeline.Pipeline;
import com.dsi.parallax.pipeline.csv.NumericCSVtoLabeledVectorPipe;
import com.dsi.parallax.pipeline.file.FileToLinesPipe;
import com.dsi.parallax.pipeline.instance.BinaryClassificationInstancesSink;
import com.dsi.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;
import com.dsi.parallax.pipeline.instance.IntroduceLabelNoisePipe;
import com.google.common.collect.Maps;

/**
 * The Class TestSingleLabelTerminator.
 */
public class TestSingleLabelTerminator {

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
		SingleLabelTerminator<BinaryClassificationTarget> term = new SingleLabelTerminator<BinaryClassificationTarget>();
		assertFalse(term.terminate(insts, 5));
		BinaryClassificationInstances tinsts = new BinaryClassificationInstances(
				5);
		assertTrue(term.terminate(tinsts, 5));

		tinsts.addInstance(insts.iterator().next());
		assertTrue(term.terminate(tinsts, 5));

		tinsts = new BinaryClassificationInstances(5);

		for (BinaryClassificationInstance inst : insts)
			if (inst.getLabel().getValue() > 0.5)
				tinsts.addInstance(inst);

		assertTrue(term.terminate(tinsts, 5));
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
