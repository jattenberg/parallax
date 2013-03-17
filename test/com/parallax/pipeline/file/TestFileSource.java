/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline.file;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.util.Set;

import org.junit.Test;

import com.google.common.collect.Sets;
import com.parallax.pipeline.FileSource;

public class TestFileSource {



        File file = new File(".");
        
        @Test
        public void testSingleFile() {
            FileSource source = new FileSource(file);
            Set<String> contains = Sets.newHashSet();
            while(source.provideData().hasNext()) {
                File f = (File)source.provideData().next().getData();
                contains.add(f.getAbsolutePath());
            }
            assertTrue(contains.contains(file.getAbsolutePath()));
            contains.remove(file.getAbsolutePath());
            assertTrue(contains.isEmpty());
        }

        public void testMultipleFiles() {
            FileSource source = new FileSource(file.listFiles());
            Set<File> shouldHave = Sets.newHashSet(file.listFiles());
            Set<File> contains = Sets.newHashSet();
            while(source.provideData().hasNext()) {
                File f = (File)source.provideData().next().getData();
                contains.add(f);
            }
            assertTrue(contains.containsAll(shouldHave));
            assertTrue(shouldHave.containsAll(contains));
        }

}
