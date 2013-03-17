package com.parallax.pipeline.projection;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.util.Map;

import org.junit.Test;

import com.google.common.collect.Maps;
import com.parallax.ml.instance.BinaryClassificationInstance;
import com.parallax.ml.target.BinaryTargetNumericParser;
import com.parallax.pipeline.FileSource;
import com.parallax.pipeline.Pipeline;
import com.parallax.pipeline.csv.NumericCSVtoLabeledVectorPipe;
import com.parallax.pipeline.file.FileToLinesPipe;
import com.parallax.pipeline.instance.BinaryClassificationInstancesSink;
import com.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;

public class TestIntervalNormalizationPipe {

	@Test
	public void testOnIris() {
		Pipeline<File, BinaryClassificationInstance> pipeline;
		Map<String, String> labelMap = Maps.newHashMap();
		labelMap.put("Iris-setosa", 1 + "");
		labelMap.put("Iris-versicolor", 0 + "");
		labelMap.put("Iris-virginica", 0 + "");

		IntervalNormalizationPipe ipp = new IntervalNormalizationPipe();

		pipeline = Pipeline
				.newPipeline(new FileSource(new File("data/iris.data")))
				.addPipe(new FileToLinesPipe())
				.addPipe(new NumericCSVtoLabeledVectorPipe(-1, 4, labelMap))
				.addPipe(ipp)
				.addPipe(
						new BinaryInstancesFromVectorPipe(
								new BinaryTargetNumericParser()));
		BinaryClassificationInstancesSink sink = new BinaryClassificationInstancesSink();
		sink.setSource(pipeline);
		assertTrue(ipp.isTrained());
		assertTrue(sink.hasNext());

		for (BinaryClassificationInstance inst : sink.next()) {
			for (int x_i : inst) {
				double y_i = inst.getFeatureValue(x_i);
				assertTrue(y_i >= 0 && y_i <= 1);
			}
		}
	}

	@Test
	public void testChangeRange() {

		double[] mins = new double[] { -.5, -.1, 100, 400 };
		double[] maxs = new double[] { -.4, 10, 102, 1000 };

		for (int i = 0; i < 4; i++) {
			double min = mins[i];
			double max = maxs[i];

			Pipeline<File, BinaryClassificationInstance> pipeline;
			Map<String, String> labelMap = Maps.newHashMap();
			labelMap.put("Iris-setosa", 1 + "");
			labelMap.put("Iris-versicolor", 0 + "");
			labelMap.put("Iris-virginica", 0 + "");

			IntervalNormalizationPipe ipp = new IntervalNormalizationPipe(min, max);

			pipeline = Pipeline
					.newPipeline(new FileSource(new File("data/iris.data")))
					.addPipe(new FileToLinesPipe())
					.addPipe(new NumericCSVtoLabeledVectorPipe(-1, 4, labelMap))
					.addPipe(ipp)
					.addPipe(
							new BinaryInstancesFromVectorPipe(
									new BinaryTargetNumericParser()));
			BinaryClassificationInstancesSink sink = new BinaryClassificationInstancesSink();
			sink.setSource(pipeline);
			assertTrue(ipp.isTrained());
			assertTrue(sink.hasNext());

			for (BinaryClassificationInstance inst : sink.next()) {
				for (int x_i : inst) {
					double y_i = inst.getFeatureValue(x_i);
					assertTrue(y_i >= min && y_i <= max);
				}
			}
		}
	}
}
