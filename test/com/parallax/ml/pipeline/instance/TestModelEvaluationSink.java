/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.pipeline.instance;

import static org.junit.Assert.assertEquals;

import java.io.File;

import org.junit.Test;

import com.parallax.ml.classifier.linear.updateable.AROWClassifier;
import com.parallax.ml.evaluation.OnlineEvaluation;
import com.parallax.ml.instance.BinaryClassificationInstance;
import com.parallax.ml.instance.BinaryClassificationInstances;
import com.parallax.ml.target.BinaryClassificationTarget;
import com.parallax.ml.target.BinaryTargetNumericParser;
import com.parallax.ml.vector.util.ValueScaling;
import com.parallax.pipeline.FileSource;
import com.parallax.pipeline.Pipeline;
import com.parallax.pipeline.ValueScalingPipe;
import com.parallax.pipeline.classifier.ModelEvaluationSink;
import com.parallax.pipeline.classifier.SequentialClassifierTrainingPipe;
import com.parallax.pipeline.file.FileToLinesPipe;
import com.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;
import com.parallax.pipeline.precompiled.VWtoBinaryInstancesPipeline;
import com.parallax.pipeline.text.StringSequenceToNumericDictionaryPipe;
import com.parallax.pipeline.text.StringToTokenSequencePipe;
import com.parallax.pipeline.text.StringToVWPipe;
import com.parallax.pipeline.text.TextSanitizerPipe;
import com.parallax.pipeline.text.VWtoLabeledStringPipe;

/**
 * The Class TestModelEvaluationSink.
 */
public class TestModelEvaluationSink {

	

	
	/**
	 * Test.
	 */
	@Test
	public void test() {
		int dimensions = (int)Math.pow(2, 16);
		String filename = "data/science.small.vw";
		
		AROWClassifier model = new AROWClassifier(dimensions, true);
		
		
		ModelEvaluationSink sink = new ModelEvaluationSink();
        Pipeline<File, OnlineEvaluation> pipeline;
        pipeline = Pipeline.newPipeline(new FileSource(filename))
        		.addPipe(new FileToLinesPipe()).addPipe(new StringToVWPipe())
				.addPipe(new VWtoLabeledStringPipe())
				.addPipe(new TextSanitizerPipe())
				.addPipe(new StringToTokenSequencePipe())
				.addPipe(new StringSequenceToNumericDictionaryPipe(dimensions))
				.addPipe(new ValueScalingPipe(ValueScaling.ABS))
				.addPipe(new ValueScalingPipe(ValueScaling.PRESERVING))
				.addPipe(new BinaryInstancesFromVectorPipe(new BinaryTargetNumericParser()))
				.addPipe(new SequentialClassifierTrainingPipe<AROWClassifier>(model, 20));
        
        sink.setSource(pipeline);
        OnlineEvaluation eval = sink.next();
        
        VWtoBinaryInstancesPipeline instPipe = new VWtoBinaryInstancesPipeline(filename, dimensions);
        BinaryClassificationInstances binInsts = instPipe.next();
        OnlineEvaluation eval2 = new OnlineEvaluation(20);
        model = new AROWClassifier(dimensions, true);

        for(BinaryClassificationInstance inst : binInsts) {
            if(null != inst.getLabel()) {
            	BinaryClassificationTarget target = model.predict(inst);
                eval2.add(inst.getLabel().getValue(), target.getValue());
                model.update(inst);
            }
        }
        
        assertEquals(eval2.computeAUC(), eval.computeAUC(), 0.01);
        assertEquals(eval2.computeAccuracy(), eval.computeAccuracy(), 0.01);
        assertEquals(eval2.computeBrierScore(), eval.computeBrierScore(), 0.01);
        
	}

}
