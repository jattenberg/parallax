package com.dsi.parallax.ml.utils;

import static com.google.common.base.Preconditions.checkArgument;

import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.pipeline.precompiled.VWtoBinaryInstancesPipeline;
import com.dsi.parallax.pipeline.precompiled.VWtoBinaryInstancesWithLabelNoisePipeline;

public class ScienceReader {

	public static final int DIMENSION = (int) Math.pow(2, 14);
	public static final String FILE = "data/science.small.vw";

	public static BinaryClassificationInstances readScience() {
		return readScience(DIMENSION);
	}

	public static BinaryClassificationInstances readScience(int dimension) {
		VWtoBinaryInstancesPipeline pipe = new VWtoBinaryInstancesPipeline(
				FILE, dimension);
		BinaryClassificationInstances insts = pipe.next();
		return insts;
	}

	public static BinaryClassificationInstances readScienceWithLabelNoise(
			int dimension, double pct) {
		checkArgument(pct >= 0 && pct <= 1,
				"noise percent must be in [0, 1], given: %s", pct);
		VWtoBinaryInstancesWithLabelNoisePipeline pipe = new VWtoBinaryInstancesWithLabelNoisePipeline(
				FILE, dimension, pct);
		return pipe.next();
	}
}
