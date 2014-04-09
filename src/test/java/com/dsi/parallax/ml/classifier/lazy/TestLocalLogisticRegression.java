package com.dsi.parallax.ml.classifier.lazy;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.util.Map;

import org.junit.Test;

import com.dsi.parallax.ml.classifier.lazy.KDType;
import com.dsi.parallax.ml.classifier.lazy.KNNMixingType;
import com.dsi.parallax.ml.classifier.lazy.LocalLogisticRegression;
import com.dsi.parallax.ml.classifier.lazy.LocalLogisticRegressionBuilder;
import com.dsi.parallax.ml.classifier.smoother.SmootherType;
import com.dsi.parallax.ml.evaluation.OnlineEvaluation;
import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.ml.target.BinaryTargetNumericParser;
import com.dsi.parallax.pipeline.FileSource;
import com.dsi.parallax.pipeline.Pipeline;
import com.dsi.parallax.pipeline.csv.NumericCSVtoLabeledVectorPipe;
import com.dsi.parallax.pipeline.file.FileToLinesPipe;
import com.dsi.parallax.pipeline.instance.BinaryClassificationInstancesSink;
import com.dsi.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;
import com.google.common.collect.Maps;

public class TestLocalLogisticRegression {

	/** The dim. */
	int dim = 4;

	/** The file. */
	File file = new File("data/iris.data");

	/**
	 * Test.
	 */
	@Test
	public void test() {
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
		BinaryClassificationInstances insts = sink.next();

		int folds = 3;
		for (int fold = 0; fold < folds; fold++) {

			LocalLogisticRegression model = new LocalLogisticRegression(dim,
					true);
			OnlineEvaluation eval = new OnlineEvaluation();

			model.train(insts.getTraining(fold, folds));
			for (BinaryClassificationInstance x : insts.getTesting(fold, folds)) {
				double pred = model.predict(x).getValue();
				double label = x.getLabel().getValue();
				eval.add(label, pred);
			}
			assertTrue(eval.computeAUC() > 0.5);
		}
	}

	/**
	 * Test options.
	 */
	@Test
	public void testOptions() {
		LocalLogisticRegressionBuilder builder = new LocalLogisticRegressionBuilder(
				dim, false);
		builder.setK(33).setKDTreeType(KDType.MANHATTAN)
				.setLabelMizingType(KNNMixingType.MODE)
				.setRegulizerType(SmootherType.UPDATEABLEPLATT)
				.setSizeLimit(100);

		LocalLogisticRegression model = builder.build();
		assertEquals(33, model.getK());
		assertEquals(100, model.getSizeLimit());
		assertEquals(KDType.MANHATTAN, model.getKdType());
		assertEquals(KNNMixingType.MODE, model.getMixingType());
		assertEquals(false, model.usesBiasTerm());
		assertEquals(dim, model.getModelDimension());
	}

	/**
	 * Test configuration training.
	 */
	@Test
	public void testConfigurationTraining() {
		LocalLogisticRegressionBuilder builder = new LocalLogisticRegressionBuilder(
				dim, false);
		builder.setK(33).setKDTreeType(KDType.MANHATTAN)
				.setLabelMizingType(KNNMixingType.MODE)
				.setRegulizerType(SmootherType.ISOTONIC);

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
		BinaryClassificationInstances insts = sink.next();

		int folds = 3;
		for (int fold = 0; fold < folds; fold++) {
			BinaryClassificationInstances train = insts
					.getTraining(fold, folds);
			BinaryClassificationInstances test = insts.getTesting(fold, folds);
			assertTrue(train.getNumNeg() > 0);
			assertTrue(test.getNumNeg() > 0);
			assertTrue(train.getNumPos() > 0);
			assertTrue(train.getNumPos() > 0);
			LocalLogisticRegression model = builder.build();
			OnlineEvaluation eval = new OnlineEvaluation();
			model.train(train);
			for (BinaryClassificationInstance x : test) {
				double pred = model.predict(x).getValue();
				double label = x.getLabel().getValue();
				eval.add(label, pred);
			}
			assertTrue(eval.computeAUC() > 0.5);
		}
	}

}
