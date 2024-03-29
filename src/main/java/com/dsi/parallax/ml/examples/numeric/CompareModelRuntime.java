package com.dsi.parallax.ml.examples.numeric;

import com.dsi.parallax.ml.classifier.Classifier;
import com.dsi.parallax.ml.classifier.Classifiers;
import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.ml.target.BinaryTargetNumericParser;
import com.dsi.parallax.pipeline.FileSource;
import com.dsi.parallax.pipeline.Pipeline;
import com.dsi.parallax.pipeline.csv.NumericCSVtoLabeledVectorPipe;
import com.dsi.parallax.pipeline.file.FileToLinesPipe;
import com.dsi.parallax.pipeline.instance.BinaryClassificationInstancesSink;
import com.dsi.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;
import com.dsi.parallax.pipeline.text.RegexStringFilterPipe;
import com.google.common.collect.Maps;
import org.apache.commons.lang.time.StopWatch;
import org.apache.commons.math.stat.descriptive.DescriptiveStatistics;

import java.io.File;
import java.util.Map;

/**
 * An example for comparing the training time for a variety of classifier types.
 */
public class CompareModelRuntime {

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
		sink.setSource(pipeline);

		BinaryClassificationInstances instances = sink.next();
		StopWatch sw = new StopWatch();
		int folds = 5;

		for (Classifiers cl : Classifiers.values()) {
			if (cl == Classifiers.PEGASOS || cl == Classifiers.KNN
					|| cl == Classifiers.FORGETRON
					|| cl == Classifiers.BUDGETKERNELPERCEPTRON) {
				continue;
			}
			System.out.println("buildnig a: " + cl);
			DescriptiveStatistics stats = new DescriptiveStatistics();
			for (int fold = 0; fold < folds; fold++) {
				sw.reset();
				Classifier<?> model = cl.getClassifier(dimension, true);
				sw.start();
				model.train(instances.getTraining(fold, folds));
				sw.stop();
				stats.addValue(sw.getTime());
			}
			System.out.println(cl.toString() + "- mean:" + stats.getMean()
					+ " var: " + stats.getVariance());
		}

	}
}
