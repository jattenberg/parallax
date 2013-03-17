package com.parallax.ml.instance;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import com.parallax.ml.target.BinaryClassificationTarget;

public class TestBinaryClassificationInstances {

	@Test
	public void testInduceSkew() {
		BinaryClassificationInstances insts = new BinaryClassificationInstances(
				0);
		for (int i = 0; i < 307; i++) {
			BinaryClassificationInstance inst = new BinaryClassificationInstance(
					0);
			inst.setLabel(new BinaryClassificationTarget(1));
			insts.addInstance(inst);
		}
		for (int i = 0; i < 383; i++) {
			BinaryClassificationInstance inst = new BinaryClassificationInstance(
					0);
			inst.setLabel(new BinaryClassificationTarget(0));
			insts.addInstance(inst);
		}
		assertEquals(insts.getPositiveRatio(), 307. / (307. + 383.), 0.00001);

		for (double r = 0; r <= 1; r += 0.1) {
			BinaryClassificationInstances skewed = insts.induceSkew(r);
			assertEquals(skewed.getPositiveRatio(), r, 0.1);
		}
	}

	@Test
	public void testStratification() {
		int pos = 900, neg = 100;
		BinaryClassificationInstances insts = new BinaryClassificationInstances(
				1);
		for (int i = 0; i < 1000; i++) {
			if (i < pos) {
				BinaryClassificationInstance inst = new BinaryClassificationInstance(
						1);
				inst.setLabel(new BinaryClassificationTarget(1));
				insts.add(inst);
			} else {
				BinaryClassificationInstance inst = new BinaryClassificationInstance(
						1);
				inst.setLabel(new BinaryClassificationTarget(0));
				insts.add(inst);
			}
		}
		int folds = 20;
		for (int fold = 0; fold < folds; fold++) {
			BinaryClassificationInstances training = insts
					.getStratifiedTraining(fold, folds);
			BinaryClassificationInstances testing = insts.getStratifiedTesting(
					fold, folds);

			assertEquals(training.getPositiveRatio(), pos
					/ (double) (pos + neg), 0.00001);
			assertEquals(1 - training.getPositiveRatio(), neg
					/ (double) (pos + neg), 0.00001);

			
			assertEquals(testing.getPositiveRatio(), pos
					/ (double) (pos + neg), 0.00001);
			assertEquals(1 - testing.getPositiveRatio(), neg
					/ (double) (pos + neg), 0.00001);
			
			
			
		}
	}

}
