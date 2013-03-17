package com.parallax.ml.classifier;

import static org.junit.Assert.assertTrue;

import com.parallax.ml.evaluation.OnlineEvaluation;
import com.parallax.ml.instance.BinaryClassificationInstances;
import com.parallax.ml.utils.AdsReader;
import com.parallax.ml.utils.IrisReader;
import com.parallax.ml.utils.ScienceReader;

public class ClassifierTestUtilities {

	private final static int FOLDS = 5;

	public static <C extends Classifier<C>, B extends ClassifierBuilder<C, B>> void testCanClassify(
			B builder) {

		BinaryClassificationInstances insts = IrisReader.readIris();
		builder.setDimension(IrisReader.DIMENSION);
		OnlineEvaluation eval = new OnlineEvaluation();

		for (int fold = 0; fold < FOLDS; fold++) {
			C model = builder.build();
			model.train(insts.getTraining(fold, FOLDS));
			eval.add(insts.getTesting(fold, FOLDS), model);
		}
		System.out.println(eval);
		assertTrue(eval.computeAccuracy() > 0.5);
		assertTrue(eval.computeAUC() > 0.5);
	}

	public static <C extends Classifier<C>, B extends ClassifierBuilder<C, B>> void testCanClassifyAds(
			B builder) {

		BinaryClassificationInstances insts = AdsReader.readAds();
		builder.setDimension(AdsReader.DIMENSION);
		OnlineEvaluation eval = new OnlineEvaluation();

		for (int fold = 0; fold < FOLDS; fold++) {
			C model = builder.build();
			model.train(insts.getTraining(fold, FOLDS));
			eval.add(insts.getTesting(fold, FOLDS), model);
		}

		assertTrue(eval.computeAccuracy() > 0.5);
		assertTrue(eval.computeAUC() > 0.5);
	}

	public static <C extends Classifier<C>, B extends ClassifierBuilder<C, B>> void testCanClassifyText(
			B builder) {

		BinaryClassificationInstances insts = ScienceReader.readScience();
		builder.setDimension(ScienceReader.DIMENSION);
		OnlineEvaluation eval = new OnlineEvaluation();

		for (int fold = 0; fold < FOLDS; fold++) {
			C model = builder.build();
			model.train(insts.getTraining(fold, FOLDS));
			eval.add(insts.getTesting(fold, FOLDS), model);
		}

		System.out.println("TEXT" + eval);
		assertTrue(eval.computeAccuracy() > 0.5);
		assertTrue(eval.computeAUC() > 0.5);
	}

}
