package com.dsi.parallax.ml.examples.text;

import com.dsi.parallax.ml.classifier.kernelmethods.KernelMethodBuilder.KernelSGDBuilder;
import com.dsi.parallax.ml.classifier.kernelmethods.KernelSGD;
import com.dsi.parallax.ml.classifier.smoother.SmootherType;
import com.dsi.parallax.ml.evaluation.OnlineEvaluation;
import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.ml.mercerkernels.KernelConfigurableBuilder;
import com.dsi.parallax.ml.mercerkernels.KernelType;
import com.dsi.parallax.ml.target.BinaryTargetNumericParser;
import com.dsi.parallax.ml.util.csv.CSV;
import com.dsi.parallax.pipeline.Pipeline;
import com.dsi.parallax.pipeline.csv.CSVPipe;
import com.dsi.parallax.pipeline.file.FileReaderSource;
import com.dsi.parallax.pipeline.file.ReaderToLinesPipe;
import com.dsi.parallax.pipeline.instance.BinaryClassificationInstancesSink;
import com.dsi.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;
import com.dsi.parallax.pipeline.text.StringSplitPipe;
import com.google.common.collect.Sets;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

public class CategoricalTextCSV {

	static File file = new File("./data/train.csv");
	static int folds = 5;

	public static void main(String[] args) throws Exception {
		BufferedReader reader = new BufferedReader(new FileReader(file));
		List<String> columnNames = Arrays.asList(reader.readLine().split(","));

		String labelColumn = "survived";
		String idColumn = "name";
		Set<String> sexes = Sets.newHashSet("male", "female");
		Set<String> embarked = Sets.newHashSet("Q", "C", "S");
		Set<String> sibsp = Sets.newHashSet("0", "1", "2", "3", "4", "5", "6",
				"7", "8");
		Set<String> parch = Sets.newHashSet("0", "1", "2", "3", "4", "5", "6");

		CSV csv = CSV.numericColumnCSV(columnNames).setIdColumn(idColumn)
				.setLabelColumn(labelColumn)
				.setValuesForCategoricalColumn("sex", sexes)
				.setValuesForCategoricalColumn("embarked", embarked)
				.setValuesForCategoricalColumn("sibsp", sibsp)
				.setValuesForCategoricalColumn("parch", parch)
				.ignoreColumn("cabin").ignoreColumn("ticket").initialize();

		Pipeline<BufferedReader, BinaryClassificationInstance> pipeline = Pipeline
				.newPipeline(new FileReaderSource(reader))
				.addPipe(new ReaderToLinesPipe())
				.addPipe(new StringSplitPipe(','))
				.addPipe(new CSVPipe(csv))
				.addPipe(
						new BinaryInstancesFromVectorPipe(
								new BinaryTargetNumericParser()));

		BinaryClassificationInstancesSink sink = new BinaryClassificationInstancesSink();
		sink.setSource(pipeline);

		BinaryClassificationInstances insts = sink.next();
		int dimension = insts.getDimensions();
		int size = insts.size();

		System.out.println("got: " + size + " instances of dimension: "
				+ dimension);

		KernelSGDBuilder builder = new KernelSGDBuilder(dimension, false);
		builder.setCrossvalidateSmootherTraining(5)
				.setRegulizerType(SmootherType.ISOTONIC)
				.setKernelBuilder(
						new KernelConfigurableBuilder()
								.setKernelType(KernelType.RBF));
		OnlineEvaluation eval = new OnlineEvaluation();

		for (int fold = 0; fold < folds; fold++) {
			KernelSGD model = builder.build();
			model.train(insts.getTraining(fold, folds));
			eval.add(insts.getTesting(fold, folds), model);
		}

		System.out.println(eval);
	}
}
