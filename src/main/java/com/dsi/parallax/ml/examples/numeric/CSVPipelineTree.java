/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.examples.numeric;

import com.dsi.parallax.ml.classifier.trees.ID3Builder;
import com.dsi.parallax.ml.classifier.trees.ID3TreeClassifier;
import com.dsi.parallax.ml.evaluation.OnlineEvaluation;
import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.target.BinaryTargetNumericParser;
import com.dsi.parallax.pipeline.FileSource;
import com.dsi.parallax.pipeline.Pipeline;
import com.dsi.parallax.pipeline.csv.NumericCSVtoLabeledVectorPipe;
import com.dsi.parallax.pipeline.file.FileToLinesPipe;
import com.dsi.parallax.pipeline.instance.BinaryClassificationInstancesSink;
import com.dsi.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;
import com.dsi.parallax.pipeline.text.RegexStringFilterPipe;
import com.google.common.collect.Maps;

import java.io.File;
import java.util.Map;

/**
 * An example where examples are represented by numeric features in a CSV file.
 * These examples are slurped in and transformed to instances. These instances
 * are then used to do 5 fold x-validation on an ID3 classifier.
 */
public class CSVPipelineTree {

	/**
	 * The main method.
	 * 
	 * @param args
	 *            the arguments
	 */
	public static void main(String[] args) {

		String file = "data/ad.data";
		int dimension = 1558;

		Pipeline<File, BinaryClassificationInstance> pipeline;

		Map<String, String> labelMap = Maps.newHashMap();
		labelMap.put("ad.", "" + 1);
		labelMap.put("nonad.", "" + 0);

		pipeline = Pipeline
				.newPipeline(new FileSource(file))
				.addPipe(new FileToLinesPipe())
				.addPipe(new RegexStringFilterPipe("\\?"))
				.addPipe(
						new NumericCSVtoLabeledVectorPipe(" *, *", dimension,
								labelMap))
				.addPipe(
						new BinaryInstancesFromVectorPipe(
								new BinaryTargetNumericParser()));

		BinaryClassificationInstancesSink sink = new BinaryClassificationInstancesSink();
		sink.setSource(pipeline.process());

		BinaryClassificationInstances instances = sink.next().shuffle();

		int folds = 5;

		ID3Builder builder = new ID3Builder(dimension, false).setMaxDepth(500);

		for (int fold = 0; fold < folds; fold++) {
			OnlineEvaluation eval = new OnlineEvaluation();
			BinaryClassificationInstances training = instances.getTraining(
					fold, folds);
			ID3TreeClassifier model = builder.build();
			model.train(training);
			for (Instance<BinaryClassificationTarget> inst : instances
					.getTesting(fold, folds)) {
				BinaryClassificationTarget label = inst.getLabel();
				BinaryClassificationTarget prediction = model.predict(inst);
				eval.add(label.getValue(), prediction.getValue());
			}
			System.out.println(eval.toString());
		}
	}
}
