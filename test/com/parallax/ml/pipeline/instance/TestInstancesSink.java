/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.pipeline.instance;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Map;

import org.junit.Test;

import com.google.common.collect.Maps;
import com.parallax.ml.instance.BinaryClassificationInstance;
import com.parallax.ml.instance.BinaryClassificationInstances;
import com.parallax.ml.target.BinaryTargetNumericParser;
import com.parallax.pipeline.FileSource;
import com.parallax.pipeline.Pipeline;
import com.parallax.pipeline.csv.NumericCSVtoLabeledVectorPipe;
import com.parallax.pipeline.file.FileToLinesPipe;
import com.parallax.pipeline.instance.BinaryClassificationInstancesSink;
import com.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;


/**
 * The Class TestInstancesSink.
 */
public class TestInstancesSink {

    /** The file. */
    File file = new File("data/iris.data");
    
    /** The bins. */
    int bins = 10000;

    /**
     * Test instance sinking.
     *
     * @throws IOException Signals that an I/O exception has occurred.
     */
    @Test
    public void testInstanceSinking() throws IOException {
        Map<String, String> labelMap = Maps.newHashMap();
		labelMap.put("Iris-setosa", 1+"");
		labelMap.put("Iris-versicolor", 0+"");
		labelMap.put("Iris-virginica", 0+"");
        BinaryClassificationInstancesSink sink = new BinaryClassificationInstancesSink();
        Pipeline<File, BinaryClassificationInstance> pipeline;
        pipeline = Pipeline.newPipeline(new FileSource(file))
                .addPipe(new FileToLinesPipe())
                .addPipe(new NumericCSVtoLabeledVectorPipe(-1,4, labelMap))
                .addPipe(new BinaryInstancesFromVectorPipe(new BinaryTargetNumericParser()));
        
        assertTrue(!sink.hasNext() );
        sink.setSource(pipeline);
        assertTrue( sink.hasNext() );
        BinaryClassificationInstances insts = sink.next();
        assertTrue(!sink.hasNext());
        
        int ct2 = 0;
        BufferedReader reader = new BufferedReader(new FileReader(file));
        @SuppressWarnings("unused")
        String line;
        while (null != (line = reader.readLine()))
            ct2++;
        assertEquals(ct2, insts.size());
       
    }

}
