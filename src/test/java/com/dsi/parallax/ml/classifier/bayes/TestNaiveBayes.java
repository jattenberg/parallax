/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.classifier.bayes;

import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.dsi.parallax.ml.classifier.bayes.NaiveBayes;
import com.dsi.parallax.ml.distributions.DistributionConfigurableBuilder;
import com.dsi.parallax.ml.distributions.DistributionType;
import com.dsi.parallax.ml.distributions.kde.KDEConfigurableBuilder;
import com.dsi.parallax.ml.distributions.kde.KDEKernel;
import com.dsi.parallax.ml.evaluation.OnlineEvaluation;
import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.pipeline.precompiled.VWtoBinaryInstancesPipeline;

/**
 * The Class TestNaiveBayes.
 */
public class TestNaiveBayes {

	/**
	 * Test.
	 */
	@Test
	public void test() {
		VWtoBinaryInstancesPipeline pipe = new VWtoBinaryInstancesPipeline(
				"data/science.small.vw", (int) Math.pow(2, 18));
		assertTrue(pipe.hasNext());
		BinaryClassificationInstances insts = pipe.next().shuffle();

		int folds = 3;
		for (int fold = 0; fold < folds; fold++) {
			BinaryClassificationInstances train = insts
					.getTraining(fold, folds);
			BinaryClassificationInstances test = insts.getTesting(fold, folds);
			assertTrue(train.getNumNeg() > 0);
			assertTrue(test.getNumNeg() > 0);
			assertTrue(train.getNumPos() > 0);
			assertTrue(train.getNumPos() > 0);
			NaiveBayes model = new NaiveBayes((int) Math.pow(2, 18), true);
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

	/**
	 * Test other configurations.
	 */
	@Test
	public void testOtherConfigurations() {
		VWtoBinaryInstancesPipeline pipe = new VWtoBinaryInstancesPipeline(
				"data/science.small.vw", (int) Math.pow(2, 18));
		assertTrue(pipe.hasNext());
		BinaryClassificationInstances insts = pipe.next().shuffle();

		DistributionConfigurableBuilder distBuilder = new DistributionConfigurableBuilder();
		distBuilder.setAlpha(2.)
			.setBins(33);
		for (DistributionType dtype : DistributionType.values()) {
			distBuilder.setDistributionType(dtype);

			int folds = 3;
			for (int fold = 0; fold < folds; fold++) {
				BinaryClassificationInstances train = insts.getTraining(fold,
						folds);
				BinaryClassificationInstances test = insts.getTesting(fold,
						folds);
				assertTrue(train.getNumNeg() > 0);
				assertTrue(test.getNumNeg() > 0);
				assertTrue(train.getNumPos() > 0);
				assertTrue(train.getNumPos() > 0);
				NaiveBayes model = new NaiveBayes((int) Math.pow(2, 18), true)
						.setDistributionBuilder(distBuilder);

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
	
	/**
	 * Test kde distributions.
	 */
	@Test
	public void testKDEDistributions() {
		VWtoBinaryInstancesPipeline pipe = new VWtoBinaryInstancesPipeline(
				"data/science.small.vw", (int) Math.pow(2, 18));
		assertTrue(pipe.hasNext());
		BinaryClassificationInstances insts = pipe.next().shuffle();

		DistributionConfigurableBuilder distBuilder = new DistributionConfigurableBuilder();
		distBuilder.setDistributionType(DistributionType.KDE);
		KDEConfigurableBuilder kdeBuilder = new KDEConfigurableBuilder();
		kdeBuilder.setBandWidth(33)
			.setDistanceDamping(2);
		
		for (KDEKernel ktype : KDEKernel.values()) {
			kdeBuilder.setKernelType(ktype);
			distBuilder.setKDEBuilder(kdeBuilder);

			int folds = 3;
			for (int fold = 0; fold < folds; fold++) {
				BinaryClassificationInstances train = insts.getTraining(fold,
						folds);
				BinaryClassificationInstances test = insts.getTesting(fold,
						folds);
				assertTrue(train.getNumNeg() > 0);
				assertTrue(test.getNumNeg() > 0);
				assertTrue(train.getNumPos() > 0);
				assertTrue(train.getNumPos() > 0);
				NaiveBayes model = new NaiveBayes((int) Math.pow(2, 18), true)
						.setDistributionBuilder(distBuilder);

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
}
