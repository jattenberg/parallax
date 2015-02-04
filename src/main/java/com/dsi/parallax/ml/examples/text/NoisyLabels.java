/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.examples.text;

import com.dsi.parallax.ml.classifier.UpdateableType;
import com.dsi.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.LogisticRegressionBuilder;
import com.dsi.parallax.ml.classifier.linear.updateable.LogisticRegression;
import com.dsi.parallax.ml.evaluation.OnlineEvaluation;
import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.ml.target.BinaryTargetNumericParser;
import com.dsi.parallax.pipeline.Context;
import com.dsi.parallax.pipeline.ContextAddingFunction;
import com.dsi.parallax.pipeline.FileSource;
import com.dsi.parallax.pipeline.Pipeline;
import com.dsi.parallax.pipeline.csv.NumericCSVtoLabeledVectorPipe;
import com.dsi.parallax.pipeline.file.FileToLinesPipe;
import com.dsi.parallax.pipeline.instance.BinaryClassificationInstancesSink;
import com.dsi.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;
import com.dsi.parallax.pipeline.instance.IntroduceLabelNoisePipe;
import com.google.common.collect.Iterators;
import com.google.common.collect.Maps;

import java.io.File;
import java.util.Iterator;
import java.util.Map;

// TODO: Auto-generated Javadoc
/**
 * The Class NoisyLabels.
 */
public class NoisyLabels {
	
	/**
	 * The main method.
	 *
	 * @param args the arguments
	 */
	public static void main(String[] args) {
		String filename = "data/iris.data";

        Map<String, String> labelMap = Maps.newHashMap();
		labelMap.put("Iris-setosa", 1+"");
		labelMap.put("Iris-versicolor", 0+"");
		labelMap.put("Iris-virginica", 0+"");
		
		System.out.println("noise\tL\tL2\tAUC");
		for(double ratio = 0; ratio <= 1; ratio += 0.05) {
			
	        BinaryClassificationInstancesSink sink = new BinaryClassificationInstancesSink();
	        Pipeline<File, BinaryClassificationInstance> pipeline;
	        pipeline = Pipeline.newPipeline(new FileSource(filename))
	                .addPipe(new FileToLinesPipe())
	                .addPipe(new NumericCSVtoLabeledVectorPipe(-1,4, labelMap))
	                .addPipe(new BinaryInstancesFromVectorPipe(new BinaryTargetNumericParser()));
	        sink.setSource(pipeline);
	        
			//VWtoBinaryInstancesWithLabelNoisePipeline pipe = new VWtoBinaryInstancesWithLabelNoisePipeline(filename, (int)Math.pow(2, 18), ratio);
			BinaryClassificationInstances insts = (BinaryClassificationInstances)sink.next().shuffle();
			
			
			for(double l = 0; l <= 100000; l = (l==0?1:l*10)) {
				double vectorSize = 0;
				double AUC = 0;
				
				for(int fold = 0; fold < 10; fold++) {
					BinaryClassificationInstances trainingT = insts.getTraining(fold, 10);

					Iterator<Context<BinaryClassificationInstance>> binIt = Iterators.transform(trainingT.iterator(), new ContextAddingFunction<BinaryClassificationInstance>());
					IntroduceLabelNoisePipe noiser = new IntroduceLabelNoisePipe(ratio);
					Iterator<Context<BinaryClassificationInstance>> binIt2 = noiser.processIterator(binIt);
					sink.setSource(binIt2);
					BinaryClassificationInstances training = sink.next();
					
					
					BinaryClassificationInstances testing = insts.getTesting(fold, 10);

					LogisticRegressionBuilder builder = new LogisticRegressionBuilder((int)Math.pow(2, 18), true)
						.setUpdateableType(UpdateableType.WINNOWING)
						.setPasses(1)
						.setSquaredWeight(l);
						
					
					LogisticRegression model = builder.build();
					model.train(training);
					vectorSize += model.getVector().L2Norm();
					
					OnlineEvaluation eval = new OnlineEvaluation();
					for(BinaryClassificationInstance inst : testing){
						eval.add(inst.getLabel(), model.predict(inst));
					}
					AUC += eval.computeAUC();	
				}
				vectorSize/=10.;
				AUC/=10.;
				System.out.println(ratio +"\t"+ l + "\t" + vectorSize + "\t" + AUC);
			}
		}
	}

	

}
