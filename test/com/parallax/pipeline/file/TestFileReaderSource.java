package com.parallax.pipeline.file;

import static org.junit.Assert.assertEquals;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;

import org.junit.Test;

public class TestFileReaderSource {

    File file = new File("./data/iris.data");
    
    @Test
    public void testSingleFile() throws IOException {
        FileReaderSource source = new FileReaderSource(file);
        while(source.provideData().hasNext()) {
            BufferedReader f = source.provideData().next().getData();
            int lines = 0;
            while( null != f.readLine()) {
            	lines++;
            }
            assertEquals(lines, 150);
        }
    }

}
