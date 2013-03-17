/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Iterator;

import org.junit.Test;

import com.parallax.ml.vector.LinearVector;
import com.parallax.pipeline.file.FileToLinesPipe;
import com.parallax.pipeline.text.StringSequenceToNGramsPipe;
import com.parallax.pipeline.text.StringSequenceToNumericDictionaryPipe;
import com.parallax.pipeline.text.StringToTokenSequencePipe;
import com.parallax.pipeline.text.TextSanitizerPipe;

public class TestPipeline {

    File file = new File("README");
    int bins = 10000;

    @Test
    public void testPipeWorks() throws IOException {
        Pipeline<File, LinearVector> pipeline;
        pipeline = Pipeline.newPipeline(new FileSource(file))
                .addPipe(new FileToLinesPipe())
                .addPipe(new TextSanitizerPipe())
                .addPipe(new StringToTokenSequencePipe())
                .addPipe(new StringSequenceToNGramsPipe(2))
                .addPipe(new StringSequenceToNumericDictionaryPipe(bins));
        // .addPipe(new LinearVectorLengthFilterPipe(2));

        Iterator<Context<LinearVector>> it = pipeline.process();
        int ct = 0;
        while (it.hasNext()) {
            ++ct;
            @SuppressWarnings("unused")
            LinearVector inst = it.next().getData();
        }
        int ct2 = 0;
        BufferedReader reader = new BufferedReader(new FileReader(file));
        @SuppressWarnings("unused")
        String line;
        while (null != (line = reader.readLine()))
            ct2++;
        assertEquals(ct2, ct);
        
        pipeline = Pipeline.newPipeline(new FileSource(file))
                .addPipe(new FileToLinesPipe())
                .addPipe(new TextSanitizerPipe())
                .addPipe(new StringToTokenSequencePipe())
                .addPipe(new StringSequenceToNGramsPipe(2))
                .addPipe(new StringSequenceToNumericDictionaryPipe(bins))
                .addPipe(new LinearVectorLengthFilterPipe(2));  
        it = pipeline.process();
        ct = 0;
        while (it.hasNext()) {
            ++ct;
            @SuppressWarnings("unused")
            LinearVector inst = it.next().getData();
        }
        assertTrue(ct2 != ct);
    }
    
}
