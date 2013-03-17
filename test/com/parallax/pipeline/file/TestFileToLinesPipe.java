/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline.file;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Iterator;

import org.junit.Test;

import com.parallax.pipeline.Context;
import com.parallax.pipeline.FileSource;
import com.parallax.pipeline.Pipeline;

public class TestFileToLinesPipe {

	File file = new File(".gitignore");
	@Test
	public void test() throws IOException {
		
        Pipeline<File, String> pipeline;
        pipeline = Pipeline.newPipeline(new FileSource(file))
                .addPipe(new FileToLinesPipe());
        
        Iterator<Context<String>> it = pipeline.process();
        assertTrue(it.hasNext());
        int count = 0;
        while(it.hasNext()) {
        	it.next();
        	count++;
        }
        assertTrue(!it.hasNext());

        int ct2 = 0;
        BufferedReader reader = new BufferedReader(new FileReader(file));
        @SuppressWarnings("unused")
        String line;
        while (null != (line = reader.readLine()))
            ct2++;
        assertEquals(ct2, count);
	}

}
