/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.file;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.File;
import java.util.Iterator;
import java.util.Set;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.filefilter.TrueFileFilter;
import org.junit.Test;

import com.dsi.parallax.pipeline.Context;
import com.dsi.parallax.pipeline.file.DirectoryToFilesPipe;
import com.google.common.collect.Iterators;
import com.google.common.collect.Sets;

public class TestDirectoryToFilesPipe {

    File file = new File(".");

    @Test
    public void checkFindsAll() {
        if (!file.isDirectory())
            fail("cant find a file: " + file.getAbsoluteFile());
        else {
            Set<String> contains = Sets.newHashSet();
            Context<File> fileContext = new Context<File>(file);
            Iterator<File> it = FileUtils.iterateFiles(file,
                    TrueFileFilter.INSTANCE, TrueFileFilter.INSTANCE);
            while (it.hasNext()) {
                File f = it.next();
                contains.add(f.toString());
            }
            DirectoryToFilesPipe pipe = new DirectoryToFilesPipe();
            @SuppressWarnings("unchecked")
			Iterator<Context<File>> it2 = pipe.processIterator(Iterators.forArray(fileContext));
            Set<String> pipeContains = Sets.newHashSet();
            while (it2.hasNext()) {
                File f = (File) it2.next().getData();
                pipeContains.add(f.toString());
            }
            assertTrue(contains.containsAll(pipeContains));
            assertTrue(pipeContains.containsAll(contains));
        }
    }

    @Test
    public void checkFindsAllIterator() {
        Set<File> files = Sets.newHashSet(file.listFiles());
        Set<String> contains = Sets.newHashSet();
        Set<String> pipeContains = Sets.newHashSet();
        Set<Context<File>> contexts = Sets.newHashSet();

        for (File f : files) {
            if (f.isDirectory()) {
                Iterator<File> it = FileUtils.iterateFiles(f,
                        TrueFileFilter.INSTANCE, TrueFileFilter.INSTANCE);
                while (it.hasNext()) {
                    File v = it.next();
                    contains.add(v.toString());
                }

            } else {
                contains.add(f.toString());
            }
            contexts.add(new Context<File>(f));
        }
        DirectoryToFilesPipe pipe = new DirectoryToFilesPipe();
        Iterator<Context<File>> it2 = pipe.processIterator(contexts.iterator());
        while (it2.hasNext()) {
            File f = (File) it2.next().getData();
            pipeContains.add(f.toString());
        }
        assertTrue(contains.containsAll(pipeContains));
        assertTrue(pipeContains.containsAll(contains));
    }

}
