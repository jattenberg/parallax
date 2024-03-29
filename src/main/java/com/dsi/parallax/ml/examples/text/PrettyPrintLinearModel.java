package com.dsi.parallax.ml.examples.text;

import com.dsi.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.LogisticRegressionBuilder;
import com.dsi.parallax.ml.classifier.linear.updateable.LogisticRegression;
import com.dsi.parallax.ml.evaluation.OnlineEvaluation;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.ml.utils.IrisReader;

public class PrettyPrintLinearModel {

	public static void main(String[] args) {
		BinaryClassificationInstances insts = IrisReader.readIris().shuffle();
		LogisticRegressionBuilder builder = new LogisticRegressionBuilder(
				IrisReader.DIMENSION, true).setCauchyWeight(3d);

		int folds = 10;
		OnlineEvaluation eval = new OnlineEvaluation();

		for (int fold = 0; fold < folds; fold++) {
			LogisticRegression lr = builder.build();
			lr.train(insts.getTraining(fold, folds));
			eval.add(insts.getTesting(fold, folds), lr);
			System.out.println("fold: " + fold + ", model: " + lr);
		}

		System.out.println(eval);
	}

}
