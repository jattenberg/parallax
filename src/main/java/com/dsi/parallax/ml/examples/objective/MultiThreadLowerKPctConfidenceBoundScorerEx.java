package com.dsi.parallax.ml.examples.objective;

import java.io.File;

import com.dsi.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.LogisticRegressionBuilder;
import com.dsi.parallax.ml.classifier.linear.updateable.LogisticRegression;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.ml.objective.AUCObjective;
import com.dsi.parallax.ml.objective.FoldEvaluator;
import com.dsi.parallax.ml.objective.LowerKPctConfidenceBoundScorer;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.pipeline.precompiled.VWtoBinaryInstancesPipeline;

public class MultiThreadLowerKPctConfidenceBoundScorerEx {

	private static File textFile = new File("data/science.small.vw");
	
	public static BinaryClassificationInstances getTextInstances() {
		VWtoBinaryInstancesPipeline pipe = new VWtoBinaryInstancesPipeline(
				textFile, (int) Math.pow(2, 18));
		BinaryClassificationInstances insts = pipe.next();
		return insts;
	}
	
	public static void main(String[] args) {
		BinaryClassificationInstances insts = getTextInstances();
		LogisticRegressionBuilder builder = new LogisticRegressionBuilder(
				insts.getDimensions(), true);

		AUCObjective obj = new AUCObjective();
		LowerKPctConfidenceBoundScorer<BinaryClassificationTarget> scorer = new LowerKPctConfidenceBoundScorer<BinaryClassificationTarget>(
				obj);

		int folds = 5;
		for (int fold = 0; fold < folds; fold++) {
			BinaryClassificationInstances training = insts.getTraining(fold,
					folds);
			BinaryClassificationInstances testing = insts.getTesting(fold,
					folds);
			LogisticRegression model = builder.build();
			model.train(training);
			scorer.evaluate(testing, model);
		}
		double score = scorer.getScore();
		scorer = new LowerKPctConfidenceBoundScorer<BinaryClassificationTarget>(
				obj);
		FoldEvaluator eval = new FoldEvaluator(folds, 15);

		double mtscore = eval.evaluate(insts,
				new LowerKPctConfidenceBoundScorer<BinaryClassificationTarget>(
						new AUCObjective()), builder);
		
		System.out.println("score: " + score);
		System.out.println("mtscore: " + mtscore);
	}
}
