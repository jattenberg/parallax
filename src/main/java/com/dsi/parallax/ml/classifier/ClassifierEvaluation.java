/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.classifier;

import com.dsi.parallax.ml.evaluation.ConfusionMatrix;
import com.dsi.parallax.ml.evaluation.ReceiverOperatingCharacteristic;
import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.ml.target.BinaryTargetNumericParser;
import com.dsi.parallax.ml.vector.util.ValueScaling;
import com.dsi.parallax.pipeline.FileSource;
import com.dsi.parallax.pipeline.Pipeline;
import com.dsi.parallax.pipeline.Sink;
import com.dsi.parallax.pipeline.ValueScalingPipe;
import com.dsi.parallax.pipeline.csv.NumericCSVtoLabeledVectorPipe;
import com.dsi.parallax.pipeline.file.FileToLinesPipe;
import com.dsi.parallax.pipeline.instance.BinaryClassificationInstancesSink;
import com.dsi.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;
import com.dsi.parallax.pipeline.text.*;
import org.apache.commons.cli.*;
import org.apache.commons.lang.StringUtils;
import org.apache.commons.lang.time.StopWatch;

import java.io.File;
import java.lang.reflect.InvocationTargetException;

/**
 * "main" class facilitating convenient command line usage for classifier types.
 * constituent evaluation methods are called by main methods of individual
 * classifier types.
 */
@SuppressWarnings("static-access")
public final class ClassifierEvaluation {

	private static Options options = new Options();
	private static String usageString = "[Classifier] [options]";

	static {
		options.addOption(OptionBuilder.withArgName("help").isRequired(false)
				.hasArgs(0).withDescription("print this help message")
				.create('h'));
		options.addOption(OptionBuilder.withArgName("file")
				.withDescription("name of file used for cross validation")
				.isRequired(false).hasArg().create('f'));
		options.addOption(OptionBuilder
				.withArgName("folds")
				.isRequired(false)
				.hasArgs(0)
				.withDescription(
						"number of folds used for cross validation (default is 5)")
				.create('x'));
		options.addOption(OptionBuilder
				.withArgName("modelopts")
				.withDescription(
						"options used to set hyper-parameters of calling model type")
				.isRequired(false).create('m'));
		options.addOption(OptionBuilder
				.withArgName("grams")
				.isRequired(false)
				.withDescription(
						"use gram features if processing text files (vw format)")
				.hasArgs().create("g"));
		options.addOption(OptionBuilder.withArgName("verbose")
				.isRequired(false).hasArgs(0)
				.withDescription("prints more verbose output").create("v"));
		options.addOption(OptionBuilder
				.withArgName("bias")
				.isRequired(false)
				.hasArg(false)
				.withDescription(
						"should models have a bias term to avoid passing through the origin")
				.create("b"));
	}

	/**
	 * print a usage message
	 */
	private static void getHelp() {
		HelpFormatter helpFormatter = new HelpFormatter();
		helpFormatter.printHelp(usageString, options);
	}

	/**
	 * generate instances from the supplied training file
	 * @param filename
	 * @param dimensions - number of covariates to be used in the ML model
	 * @param grams - num of n-grams if using text features. 
	 * @return
	 */
	private static BinaryClassificationInstances buildInstances(
			String filename, int dimensions, int grams) {
		Pipeline<File, BinaryClassificationInstance> pipeline = null;
		Sink<BinaryClassificationInstance, BinaryClassificationInstances> instancesSink = new BinaryClassificationInstancesSink();

		if (filename.endsWith("csv")) {
			Pipeline.newPipeline(new FileSource(filename))
					.addPipe(new FileToLinesPipe())
					.addPipe(new RegexStringFilterPipe("\\?"))
					.addPipe(new NumericCSVtoLabeledVectorPipe())
					.addPipe(
							new BinaryInstancesFromVectorPipe(
									new BinaryTargetNumericParser()));
		} else if (filename.endsWith("vw")) {
			pipeline = Pipeline
					.newPipeline(new FileSource(filename))
					.addPipe(new FileToLinesPipe())
					.addPipe(new StringToVWPipe())
					.addPipe(new VWtoLabeledStringPipe())
					.addPipe(new TextSanitizerPipe())
					.addPipe(new StringToTokenSequencePipe())
					.addPipe(
							new StringSequenceToNGramsPipe(new int[] { grams }))
					.addPipe(
							new StringSequenceToNumericDictionaryPipe(
									dimensions))
					.addPipe(new ValueScalingPipe(ValueScaling.ABS))
					.addPipe(new ValueScalingPipe(ValueScaling.PRESERVING))
					.addPipe(
							new BinaryInstancesFromVectorPipe(
									new BinaryTargetNumericParser()));
		} else {
			throw new IllegalArgumentException(
					"supported file formats are csv and vw");
		}

		instancesSink.setSource(pipeline);

		return instancesSink.next().shuffle();
	}

	/**
	 * Evaluate; the "main" method classed by various classifier types
	 * 
	 * @param args
	 *            command line arguments passed from calling main method
	 * @param the
	 *            class of the Classifier calling this
	 * @throws NoSuchMethodException
	 * @throws InvocationTargetException
	 * @throws IllegalAccessException
	 * @throws ParseException
	 * @throws SecurityException
	 * @throws IllegalArgumentException
	 */
	public static void evaluate(String[] args, Class<?> regClass)
			throws IllegalArgumentException, SecurityException, ParseException,
			IllegalAccessException, InvocationTargetException,
			NoSuchMethodException {
		int dimensions = (int) Math.pow(2, 16);
		String trainingFile = null;

		boolean verbose = true, bias = false;
		String[] modelopts = new String[] {};

		int folds = 10, grams = 1;

		CommandLineParser parser = new GnuParser();
		CommandLine commandLine = parser.parse(options, args);

		if (commandLine.hasOption("h")) {
			getHelp();
			System.exit(0);
		} else {
			if (commandLine.hasOption("f")) {
				trainingFile = commandLine.getOptionValue("f");
			} else {
				getHelp();
				throw new IllegalArgumentException("must specify training file");
			}

			if (commandLine.hasOption("x")) {
				folds = Integer.parseInt(commandLine.getOptionValue("x"));
			}

			if (commandLine.hasOption("m")) {
				modelopts = StringUtils.split(commandLine.getOptionValue("m"));
			}

			if (commandLine.hasOption("g")) {
				grams = Integer.parseInt(commandLine.getOptionValue("g"));
			}

			verbose = commandLine.hasOption("v");
			bias = commandLine.hasOption("b");
		}

		if(verbose) 
			System.out.println("loading data");
		StopWatch tsw = new StopWatch();
		tsw.start();
		BinaryClassificationInstances instances = buildInstances(trainingFile,
				dimensions, grams);
		if(verbose) {
			System.out.println("done, took:" + tsw.getTime() + "ms, " + (double) tsw.getTime() / (double)instances.size() + "ms per instance");
		}
			
		
		ClassifierBuilder<?, ?> builder = Classifiers.getClassifierType(
				regClass).getClassifierBuilder(dimensions, bias);

		builder.configureFromArguments(modelopts);

		ConfusionMatrix conf = new ConfusionMatrix(2);
		ReceiverOperatingCharacteristic roc = new ReceiverOperatingCharacteristic();

		long totTrain = 0, totTest = 0;
		for (int fold = 0; fold < folds; fold++) {
			if (verbose) {
				System.out.println("on fold: " + (fold + 1));
			}

			BinaryClassificationInstances training = instances.getTraining(
					fold, folds);
			BinaryClassificationInstances testing = instances.getTesting(fold,
					folds);

			StopWatch sw = new StopWatch();
			sw.start();

			if (verbose) {
				System.out.println("training");
			}

			Classifier<?> model = builder.build();
			model.train(training);

			if (verbose)
				System.out.println("\ndone, total time: " + sw.getTime()
						+ "ms, time per instance: " + (double) (sw.getTime())
						/ (double) training.size() + "ms, # instances: "
						+ training.size());
			totTrain += sw.getTime();

			for (BinaryClassificationInstance instance : testing) {
				double label = instance.getLabel().getValue();
				sw.reset();
				sw.start();
				double prediction = model.predict(instance).getValue();
				totTest += sw.getTime();

				if (verbose) {
					System.out.println("label: " + label + ", prediction: "
							+ prediction);
				}
				conf.addInfo(label, prediction);
				roc.add(label, prediction);
			}

		}

		System.out.println("\n\nprediction time: " + totTest + "ms, or "
				+ (double) totTest / (double) instances.size()
				+ "ms per instance over " + instances.size() + " instancs");
		System.out.println("train time: " + totTrain + "ms, or "
				+ (double) totTrain / (double) instances.size()
				+ "ms per instance over " + instances.size() + " instancs");
		System.out.print(conf.toString());
		System.out.println("\n-------\n\nIR Metrics:\n" + conf.getIR());
		System.out.println("\n-------\nAUC: " + roc.binaryAUC());
		System.out.println("Brier Score: " + roc.brierScore());
	}

}
