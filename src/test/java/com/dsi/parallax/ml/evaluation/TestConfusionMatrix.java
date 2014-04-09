/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.evaluation;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import com.dsi.parallax.ml.classifier.trees.ID3TreeClassifier;
import com.dsi.parallax.ml.evaluation.ConfusionMatrix;
import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.ml.utils.IrisReader;


/**
 * The Class TestConfusionMatrix.
 */
public class TestConfusionMatrix {

	static BinaryClassificationInstances insts = IrisReader.readIris();
	
	/**
	 * Load valid matrix.
	 */
	@Test
	public void loadValidMatrix() {
		ConfusionMatrix conf = new ConfusionMatrix(2);
		for(int i = 0; i < 10; i++)
			for(int j = 0; j < 10; j++)
				conf.addInfo(i%2, j%2);
		
		assertEquals(conf.computeAccuracy(), 0.5, 0.001);

		
		double[][] matrix = conf.getMatrix();
		for(int i = 0; i < 2; i++)
			for(int j = 0; j < 2; j++) {
				assertEquals(matrix[i][j], 25, 0.001);
				assertEquals(conf.computePrecision(i), 0.5, 0.001);
				assertEquals(conf.computeRecall(i), 0.5, 0.001);
				assertEquals(conf.computeFmeasure(i), 0.5, 0.001);
			}
	}

	@Test
	public void testMatrixSize() {
		int folds = 3;
		int seen = 0;
		ConfusionMatrix conf = new ConfusionMatrix(2), hconf = new ConfusionMatrix(2);
		
		for(int i = 0; i < folds; i++) {
			ID3TreeClassifier model = new ID3TreeClassifier(insts.getDimensions(), false);

			BinaryClassificationInstances training = insts.getTraining(i, folds);
			BinaryClassificationInstances testing = insts.getTesting(i, folds);
			seen += testing.size();
			
			model.train(training);
			
			for(BinaryClassificationInstance inst : testing) {
				conf.addInfo(inst.getLabel(), model.predict(inst));
				hconf.addHard(inst.getLabel(), model.predict(inst));
			}
		}
		assertEquals(insts.size(), seen);
		assertEquals(conf.obs, insts.size(), 0.0001);
		assertEquals(hconf.obs, insts.size(), 0.0001);
		double size = matrixSum(conf);
		double hsize = matrixSum(hconf);
		assertEquals(size, insts.size(), 0.0001);
		assertEquals(hsize, insts.size(), 0.0001);
	}

	private double matrixSum(ConfusionMatrix conf) {
		double sum = 0;
		for(int i = 0; i < conf.getMatrix().length; i++) {
			for(int j = 0; j < conf.getMatrix()[i].length; j++) {
				sum += conf.getMatrix()[i][j];
			}
		}
		return sum;
	}
}
