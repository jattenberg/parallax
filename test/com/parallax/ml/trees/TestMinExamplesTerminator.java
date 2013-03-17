/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.trees;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.util.Map;

import org.junit.Test;

import com.google.common.collect.Maps;
import com.parallax.ml.instance.BinaryClassificationInstance;
import com.parallax.ml.instance.BinaryClassificationInstances;
import com.parallax.ml.target.BinaryClassificationTarget;
import com.parallax.ml.target.BinaryTargetNumericParser;
import com.parallax.pipeline.FileSource;
import com.parallax.pipeline.Pipeline;
import com.parallax.pipeline.csv.NumericCSVtoLabeledVectorPipe;
import com.parallax.pipeline.file.FileToLinesPipe;
import com.parallax.pipeline.instance.BinaryClassificationInstancesSink;
import com.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;
import com.parallax.pipeline.instance.IntroduceLabelNoisePipe;

/**
 * The Class TestMinExamplesTerminator.
 */
public class TestMinExamplesTerminator {

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
		MinExamplesTerminator<BinaryClassificationTarget> term = new MinExamplesTerminator<BinaryClassificationTarget>(
				50);
		BinaryClassificationInstances nists = new BinaryClassificationInstances(
				15);

		for (BinaryClassificationInstance inst : insts) {
			nists.addInstance(inst);
			assertEquals(nists.size() < 50, term.terminate(nists, 13));
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
