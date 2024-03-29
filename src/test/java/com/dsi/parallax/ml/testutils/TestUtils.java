/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.testutils;

import java.io.File;
import java.util.Map;

import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.ml.target.BinaryTargetNumericParser;
import com.dsi.parallax.pipeline.FileSource;
import com.dsi.parallax.pipeline.Pipeline;
import com.dsi.parallax.pipeline.csv.NumericCSVtoLabeledVectorPipe;
import com.dsi.parallax.pipeline.file.FileToLinesPipe;
import com.dsi.parallax.pipeline.instance.BinaryClassificationInstancesSink;
import com.dsi.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;
import com.dsi.parallax.pipeline.precompiled.VWtoBinaryInstancesPipeline;
import com.google.common.collect.Maps;

/**
 * The Class TestUtils.
 */
public class TestUtils {

	/** The file. */
	private static File file = new File("data/iris.data");

	/** The iris dim. */
	public static int irisDim = 4;

	/** The iris instances. */
	public static BinaryClassificationInstances irisInstances = getIrisInstances();

	/** The text file. */
	private static File textFile = new File("data/science.small.vw");

	/** The test dim. */
	public int testDim = (int) Math.pow(2, 14);

	/** The text instances. */
	public static BinaryClassificationInstances textInstances = getTextInstances();

	/**
	 * Gets the iris instances.
	 * 
	 * @return the iris instances
	 */
	public static BinaryClassificationInstances getIrisInstances() {
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

		BinaryClassificationInstances insts = sink.next();
		return insts;
	}

	/**
	 * Gets the text instances.
	 * 
	 * @return the text instances
	 */
	public static BinaryClassificationInstances getTextInstances() {
		VWtoBinaryInstancesPipeline pipe = new VWtoBinaryInstancesPipeline(
				textFile, (int) Math.pow(2, 18));
		BinaryClassificationInstances insts = pipe.next();
		return insts;
	}

}
