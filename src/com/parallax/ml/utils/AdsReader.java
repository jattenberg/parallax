package com.parallax.ml.utils;

import java.io.File;
import java.util.Map;

import com.google.common.collect.Maps;
import com.parallax.ml.instance.BinaryClassificationInstance;
import com.parallax.ml.instance.BinaryClassificationInstances;
import com.parallax.ml.target.BinaryTargetNumericParser;
import com.parallax.pipeline.FileSource;
import com.parallax.pipeline.Pipeline;
import com.parallax.pipeline.csv.NumericCSVtoLabeledVectorPipe;
import com.parallax.pipeline.file.FileToLinesPipe;
import com.parallax.pipeline.instance.BinaryClassificationInstancesSink;
import com.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;
import com.parallax.pipeline.text.RegexStringFilterPipe;

public class AdsReader {

	private static final String file = "data/ad.data";
	public static final int DIMENSION = 1558;

	public static BinaryClassificationInstances readAds() {
		Pipeline<File, BinaryClassificationInstance> pipeline;

		Map<String, String> labelMap = Maps.newHashMap();
		labelMap.put("ad.", "" + 1);
		labelMap.put("nonad.", "" + 0);

		pipeline = Pipeline
				.newPipeline(new FileSource(file))
				.addPipe(new FileToLinesPipe())
				.addPipe(new RegexStringFilterPipe("\\?"))
				.addPipe(
						new NumericCSVtoLabeledVectorPipe(" *, *", DIMENSION,
								labelMap))
				.addPipe(
						new BinaryInstancesFromVectorPipe(
								new BinaryTargetNumericParser()));

		BinaryClassificationInstancesSink sink = new BinaryClassificationInstancesSink();
		sink.setSource(pipeline.process());

		return sink.next();
	}
}
